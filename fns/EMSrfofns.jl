### EMS Robust functions
# This file contains the functions used for building a EMS model object.
# The functions are mainly the device models (solar pv, bess, tess, etc.) and other elements (grid balances) of the EMS problem.

# By: Darío Slaifstein, PhD-student @TU Delft, DCES.
# Branch:
# Version: 0.1
# Date: 17/04/2024

## Modeling functions
# These functions are used to build the EMS model object. They include the device models, the grid balances and cost function.
cd(@__DIR__)
using Parameters, Serialization

################ MODEL CREATION ################

# Electric devices
"""
    spvRFO!(model, data)

The PV panel is composed of a measurement (MPPT power), actual P out, and rolling forecast.
The function translates the measurement into a forecast and adds the forecast to the model.
# Arguments
- model::InfiniteModel: the model object containing the EMS problem.
- data::Dict: the data dictionary containing the Exogenous information (MPPT measurement).
# Returns
- model::InfiniteModel: updated with the added deterministic forecast.
"""
function spvRFO!(model::InfiniteModel, data::Dict) # solar pv panel
    # MPPT measurement
    t = model[:t];
    Dt = supports(t);
    t0 = supports(t)[1]; Δt=supports(t)[2]-supports(t)[1];
    # make new time window
    it0 = round(Int,(t0/Δt));
    itend = it0+length(supports(t))-1;

    MPPTmeas = data["SPV"].MPPTData[it0:itend];
    @parameter_function(model, PpvMPPT == (t) -> MPPTmeas[zero_order_time_index(Dt, t)])
    return model;
end;

function bessRFO!(model::InfiniteModel, sets::modelSettingsRFO, data::Dict) # stationary battery pack
    t=model[:t];
    # Pdrive = model[:Pdrive];
    num_samples = sets.num_samples;
    t0=supports(t)[1];
    # Extract data
    # General Info
    @unpack GenInfo, PerfParameters = data["BESS"]
    @unpack PowerLim, P0, SoCLim, SoC0, vLim = GenInfo
    PbessMax = PowerLim[2]; # Max power [kW]
    SoCbessMin = SoCLim[1]; # Min State of Charge [p.u.]
    SoCbessMax = SoCLim[2]; # Max State of Charge [p.u.]
    SoCbess0 = SoC0; # Initial SoC [p.u.]

    # @unpack_Generic GenInfo
    @unpack PowerLim, P0, SoCLim, SoC0, initQ, SoHQ, SoHR0, Np, Ns, η, OCVParam, vLim, ηC  = GenInfo
    @unpack ocvLine = OCVParam;
    Npbess = Np; Nsbess = Ns;
    crate = PbessMax * 1e3 / (Nsbess * Npbess * Qbess0 * vLim[2]); # C-rate

    # Add variables
    @variables(model, begin
        1e-2 ≤ Qbess ≤ 50, (start = 1e-2) # BESS capacity Q0 [kWh]
        Pbess[i ∈ 1:num_samples], (start = 0.), Infinite(t) # output power [kW]
        SoCbess[i ∈ 1:num_samples], Infinite(t) # State of Charge [kWh]
    end);

    @constraints(model, begin
        # Bidirectional power flow, ensuring only export or import
        [i ∈ 1:num_samples], - Qbess*crate ≤ Pbess[i]
        [i ∈ 1:num_samples], Pbess[i] ≤ Qbess*crate
        # State of Charge limits [kWh]
        [i ∈ 1:num_samples], SoCbess[i] ≥ SoCbessMin * Qbess
        [i ∈ 1:num_samples], SoCbess[i] ≤ SoCbessMax * Qbess
        # Initial conditions
        [i ∈ 1:num_samples], SoCbess[i](t0) ==  SoCbess0 .* Qbess
        # Transition function
        [i ∈ 1:num_samples], ∂.(SoCbess[i], t) .== -Pbess[i]

    end);
    return model;
end

function evRFO!(model::InfiniteModel, sets::modelSettingsRFO, data::Dict) # electric vehicle
    t=model[:t];
    nEV=sets.nEV; num_samples = sets.num_samples;
    t0=supports(t)[1];
    Dt = sets.dTime;

    # Extract data
    # General Info
    # Battery pack data
    PevMax = zeros(nEV)
    PevMin = zeros(nEV)
    Pev0 = zeros(nEV)
    SoCevMax = zeros(nEV)
    SoCevMin = zeros(nEV)
    SoCev0 = zeros(nEV)
    Npev = zeros(nEV)
    Nsev = zeros(nEV)
    ηev = zeros(nEV)
    Qev0 = zeros(nEV)
    SoCev0 = zeros(nEV)
    aOCV = zeros(nEV)
    bOCV = zeros(nEV)
    vMax = zeros(nEV)
    performanceParams = Vector{PerfParams}(undef, nEV)
    SoCdep = zeros(nEV)
    γ = Matrix{Vector{Float64}}(undef, num_samples, nEV)
    Pdrive = Matrix{Vector{Float64}}(undef, num_samples, nEV)


    for n in 1:nEV
        @unpack GenInfo, PerfParameters, AgingParameters = data["EV"][n].carBatteryPack
        # @unpack_Generic GenInfo
        @unpack PowerLim, P0, SoCLim, SoC0, initQ, SoHQ, SoHR0, Np, Ns, η, OCVParam, vLim, ηC  = GenInfo
        @unpack ocvLine = OCVParam

        performanceParams[n] = PerfParameters

        PevMax[n] = PowerLim[2] # Max power [kW]
        PevMin[n] = PowerLim[1] # Min power [kW]
        Pev0[n] = P0 # Initial power [kW]
        SoCevMin[n] = SoCLim[1] # Min State of Charge [p.u.]
        SoCevMax[n] = SoCLim[2] # Max State of Charge [p.u.]
        SoCev0[n] = SoC0 # Initial SoC [p.u.]
        Npev[n] = Np; Nsev[n] = Ns;
        ηev[n] = η; Qev0[n] = initQ * SoHQ;
        aOCV[n] = ocvLine[1]; bOCV[n] = ocvLine[2];
        vMax[n] = vLim[2];
        SoCdep[n] = data["EV"][n].driveInfo.SoCdep
        γ[:, n] = data["EV"][n].driveInfo.γ # is missing some info with the it0:itend, check later
        Pdrive[:,n] = data["EV"][n].driveInfo.Pdrive
    end

    # Add variables
    # Pev > 0 -> out power and Pev < 0 -> in power
    Eev0 = Qev0.*Npev.*Nsev.*vMax / 1000; # [kWh]
    @variables(model, begin
        PevMin[n] .≤ Pev[i ∈ 1:num_samples, n ∈ 1:nEV] .≤ PevMax[n], Infinite(t)  # EV charger power
        PevMin[n] .≤ PevTot[i ∈ 1:num_samples, n ∈ 1:nEV] .≤ PevMax[n], Infinite(t)  # total power of each EV, driving+V2G
        SoCevMin[n] .* Eev0[n] .≤ SoCev[i ∈ 1:num_samples, n ∈ 1:nEV] .≤ SoCevMax[n] .* Eev0[n], (start = SoCev0[n] .* Eev0[n]), Infinite(t) # State of Charge [kWh]
    end);

    # Now we need to project it into the cont t-domain.
    # create the samples for the availability in ℝ^(nₛ × nₑᵥ)
    # create the interpolation functions in a vector of samples nₛ
    γ_interps = [linear_interpolation((Dt, 1:nEV), hcat(γ[i,:])) for i in 1:num_samples]
    # make InfiniteOpt compatible on the t-cont domain
    @parameter_function(model, γf[i ∈ 1:num_samples, n ∈ 1:nEV] == (t) -> γ_interps[i](t, n))

    # User requirement at departure time.
    # The penalty is only for the first departure time.
    # get departure time index
    depIdx = [findfirst(diff(γ[i,n]) .== -1) for i ∈ 1:num_samples, n ∈ 1:nEV];
    tDep = [Dt[depIdx[i,n]] for i ∈ 1:num_samples, n ∈ 1:nEV] # departure time
    model[:ϵSoC] = [SoCev[i,n](tDep[i,n])/Eev0[n] - SoCdep[n] for i ∈ 1:num_samples, n ∈ 1:nEV];
    # for the constraints we need all departure and arrival times
    depIdx = [findall(diff(γ[i,n]) .== -1) for i ∈ 1:num_samples, n ∈ 1:nEV];
    arrIdx = [findall(diff(γ[i,n]) .== 1) for i ∈ 1:num_samples, n ∈ 1:nEV];
    # the complimentary γ is
    neg_γP = [1. .- γ[i,n] for i ∈ 1:num_samples, n ∈ 1:nEV]
    # now we form the trajectories of the Pdriveₙₑᵥ,ᵢ(t) with diff values for each driving period.
    for i ∈ 1:num_samples # loop over the samples
        for n ∈ 1:nEV # loop over the EVs
            for j ∈ 1:length(depIdx[i,n]) # loop over the connection periods
                dI = depIdx[i,n][j];
                # warning: the arrival might be in the last timestep thus length(arrIdx) = N-1 (smaller)
                aI = length(arrIdx[i,n]) .== length(depIdx[i,n]) ? arrIdx[i,n][j] : length(γ[i,n]);
                neg_γP[i,n][dI:aI] .= Pdrive[i,n][j]
            end
        end
    end
    # and we convert it into a parameter function
    neg_γP_interps = [linear_interpolation((Dt, 1:nEV), hcat(neg_γP[i,:])) for i in 1:num_samples]
    @parameter_function(model, Pdrivef[i ∈ 1:num_samples, n ∈ 1:nEV] == (t) -> neg_γP_interps[i](t, n))
    @constraints(model, begin
        [i ∈ 1:num_samples, n ∈ 1:nEV], SoCev[i,n](t0) == SoCev0[n] .* Eev0[n] # Initial conditions
        [i ∈ 1:num_samples, n ∈ 1:nEV], γf[i,n].*Pev[i,n] + Pdrivef[i,n] - PevTot[i,n] .== 0 # power balance
        [i ∈ 1:num_samples, n ∈ 1:nEV], ∂.(SoCev[i,n], t) .== -PevTot[i,n] # Transition function
    end);
    return model;
end

# Thermal devices
"""
    st(model, data)

The `st` function calculates the solar thermal power based on the given model and data.
# Arguments
- `model::InfiniteModel`: The model object representing the system.
- `data::Dict`: A dictionary containing the necessary data for the calculation.
# Returns
- `model::InfiniteModel`: The updated model object.
Note: The curtailment part of the code is currently commented out and left for future implementation.
"""
function stRFO!(model::InfiniteModel, data::Dict) # solar thermal
    # The thermal pv/heatpipes are only the converted measurement of the irradiance.
    t = model[:t];
    Dt = supports(t);
    t0 = supports(t)[1]; Δt = supports(t)[2]-supports(t)[1];
    # make new time window
    it0 = round(Int,(t0/Δt));
    itend = it0+length(supports(t))-1;
    # Extract params
    ηST = data["ST"].η; # conversion factor from Electric PV to thermal

    # Extract data
    MPPTmeas = data["SPV"].MPPTData[it0:itend];
    @parameter_function(model, Pst == (t) -> ηST*MPPTmeas[zero_order_time_index(Dt, t)])
    return model;
end;

function heatpumpRFO!(model::InfiniteModel, sets::modelSettingsRFO, data::Dict) # heat pump
    # The heat pump has a variable (electrical) and a subordinate finite_param (thermal)
    # Extract params
    t = model[:t];
    # Pdrive = model[:Pdrive];
    num_samples = sets.num_samples; # number of samples
    PhpRated = data["HP"].RatedPower; # rated power of the heat pump
    @variable(model, 0 ≤ Phpe[i ∈ 1:num_samples] ≤ PhpRated, Infinite(t));  # Electric power
    return model;
end;

"""
    tess(model::InfiniteModel, data::Dict)

The `tess` function models a Thermal Energy Storage System (TESS).
It adds variables and constraints to the model to represent the TESS's state of charge,
thermal power, and binary variable for TESS power. The function also includes initial
conditions and a bucket model constraint.

# Arguments
- `model::InfiniteModel`: The InfiniteModel to which the TESS variables and constraints will be added.
- `data::Dict`: A dictionary containing the TESS data, including capacity, power limits, initial state of charge, and thermal efficiency.

# Returns
- `model::InfiniteModel`: The updated InfiniteModel with the TESS variables and constraints added.
"""
function tessRFO!(model::InfiniteModel, sets::modelSettingsRFO, data::Dict) # stationary battery pack
    # Thermal Energy Storage System
    t = model[:t];
    # Pdrive = model[:Pdrive];
    num_samples = sets.num_samples;
    t0 = supports(t)[1];
    # Bucket model
    # Extract data
    Qtess = data["TESS"].Q; # Capacity [kWh]
    PtessMin = data["TESS"].PowerLim[1]; # Min power [kW]
    PtessMax = data["TESS"].PowerLim[2]; # Max power [kW]
    SoCtessMin = data["TESS"].SoCLim[1]; # Min State of Charge [p.u.]
    SoCtessMax = data["TESS"].SoCLim[2]; # Max State of Charge [p.u.]
    SoCtess0 = data["TESS"].SoC0; # Initial State of Charge [p.u.]
    ηtess = data["TESS"].η; # thermal efficiency

    # Add variables
    @variables(model, begin
        SoCtessMin .≤ SoCtess[i ∈ 1:num_samples] .≤ SoCtessMax, Infinite(t) # State of Charge
        PtessMin ≤ Ptess[i ∈ 1:num_samples] ≤ PtessMax, Infinite(t) # Thermal power
    end);

    # Initial conditions
    @constraints(model, begin
        [i ∈ 1:num_samples], SoCtess[i](t0) .== SoCtess0
        [i ∈ 1:num_samples], ∂.(SoCtess[i], t) .== -ηtess*Ptess[i]/Qtess # Bucket model
    end);

    return model;
end;

# other
# Grids
function gridConnRFO!(model::InfiniteModel, sets::modelSettingsRFO, data::Dict) # power electronic interface
    # infinite parameters
    t = model[:t];
    # Pdrive = model[:Pdrive];
    num_samples = sets.num_samples;
    # Grid limits
    PgMin = data["grid"].PowerLim[1]; # Min power going out
    PgMax = data["grid"].PowerLim[2]; # Max power coming in

    # Add variables
    @variables(model, begin
        PgMin ≤ Pg[i ∈ 1:num_samples] ≤ PgMax, Infinite(t)  # grid power
    end)
    return model;
end;

"""
    gridThermalRFO!(model::InfiniteModel, sets::modelSettings, data::Dict)

This function adds the thermal balance to the EMS model.

# Arguments
- `model::InfiniteModel`: The InfiniteModel object representing the grid.
- `sets::modelSettings`: The model settings.
- `data::Dict`: A dictionary containing the necessary data for the calculation.

# Returns
- `model::InfiniteModel`: The updated InfiniteModel object with the thermal power balance constraint.

"""
function gridThermalRFO!(model::InfiniteModel, sets::modelSettingsRFO, data::Dict) # thermal grid
    # Load data
    Dt = sets.dTime; num_samples = sets.num_samples;
    t = model[:t];
    t0=supports(t)[1]; Δt=supports(t)[2]-supports(t)[1];
    # make new time window
    it0 = round(Int,(t0/Δt));
    itend = it0+length(Dt)-1;
    # make InfiniteOpt compatible Plt(t) work for arbitrary times
    loadTh = data["grid"].loadTh[it0:itend];
    @parameter_function(model, Plt == (t) -> loadTh[zero_order_time_index(Dt, t)])

    # Extract params
    ηHP=data["HP"].η; # conversion factor from Electric to thermal heat pump
    # Thermal Power balance
    model[:thBalance]=@constraint(model, [i ∈ 1:num_samples], model[:Pst] + model[:Phpe][i] .* ηHP + model[:Ptess][i] .==  Plt);
    return model;
end;

"""
    peiRFO!(model::InfiniteModel, sets::modelSettings, data::Dict)

This function adds the power balance to the EMS model.

# Arguments
- `model::InfiniteModel`: The InfiniteModel object representing the grid.
- `sets::modelSettings`: The model settings.
- `data::Dict`: A dictionary containing the necessary data for the calculation.

# Returns
- `model::InfiniteModel`: The updated InfiniteModel object with the power balance constraint.

"""
function peiRFO!(model::InfiniteModel, sets::modelSettingsRFO, data::Dict) # power electronic interface
    t = model[:t];
    # Pdrive = model[:Pdrive];
    # Extract data
    Dt = sets.dTime; num_samples = sets.num_samples;
    nEV=sets.nEV;
    t0=supports(t)[1]; Δt=supports(t)[2]-supports(t)[1];
    # make new time window
    it0 = round(Int,(t0/Δt));
    itend = it0+length(Dt)-1;
    # Load data
    loadElec = data["grid"].loadE[it0:itend];
    @parameter_function(model, Ple == (t) -> loadElec[zero_order_time_index(Dt, t)])

    # Power balance DC busbar
    # If we have EVs
    Pev_sum = [sum(model[:γf][i,n].*model[:Pev][i,n] for n in 1:nEV) for i ∈ 1:num_samples];
    # If we have HP
    Phpe = any(name.(all_variables(model)) .== "Phpe[1]") ? model[:Phpe] : zeros(num_samples);
    model[:powerBalance]=@constraint(model, [i ∈ 1:num_samples], model[:PpvMPPT] .+ model[:Pbess][i] .+ Pev_sum[i] .+ model[:Pg][i] .== Ple .+ Phpe[i])
    return model;
end;

"""
    costFunctionRFO!(model, sets::modelSettings, data::Dict)

Adds Objective function to the model. The model may have three components:
``C_{\textrm{grid}}``, ``C_{\textrm{loss}}`` and a penalty for not charging the EVs.
The function includes weights and picks how to build the objective depending on the settings.

# Arguments
- `model`: The model object representing the FLEXINet simulation.
- `sets::modelSettings`: The settings object containing the model settings.
- `data`: The data object containing the simulation data.

# Returns
- `model`: The updated model object.

"""
function costFunctionRFO!(model, sets::modelSettingsRFO, data::Dict) # Objective function
    nEV = sets.nEV; num_samples = sets.num_samples;
    W = sets.costWeights;
    Dt = sets.dTime;
    t = model[:t];
    t0=supports(t)[1]; Δt=supports(t)[2]-supports(t)[1];
    # make new time window
    it0 = round(Int,t0/Δt);
    itend = it0+length(Dt)-1;

    # Note: We need to use hcat() to form the axis cause otherwise λ[:,c] broadcasts into a vector.
    priceBuy=hcat(data["grid"].λ[it0:itend, 1].*1e-3) # convert from €/MWh to €/kWh
    @parameter_function(model, λ == (t) -> priceBuy[zero_order_time_index(Dt, t)])

    # CAPEX - Battery
    CAPEX = data["BESS"].GenInfo.initValue;

    # STAGE COST - Operation
    # Grid costs
    Wgrid = W[1]; # regularization factor for grid cost. max(λ)*max(P)
    cgrid = Wgrid .* model[:Pg] * λ;

    # Define penalty for not charging
    WSoCDep = W[2]
    pDep = (isempty(model[:ϵSoC]) ? 0 : WSoCDep .* [sum(model[:ϵSoC][i,n] .^ 2 for n ∈ 1:nEV) for i ∈ 1:num_samples])

    # Stage cost
    𝒞 = ∫.(cgrid,t) .+ pDep;
    # Define objective function
    Qbess = model[:Qbess];
    @objective(model, Min, 1/num_samples * sum(𝒞[i] for i ∈ 1:num_samples) + CAPEX .* Qbess)
    return model;
end;
