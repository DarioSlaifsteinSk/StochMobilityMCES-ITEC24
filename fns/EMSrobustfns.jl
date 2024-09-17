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
using Parameters, EMSmodule, Serialization
import EMSmodule.availabilityEV
################ TYPE DEFINITION ################
@with_kw mutable struct modelSettingsRFO # structure of sets
    nEV::Int64 # number of EVs
    t0::Float64 # initial time [hr]
    Tw::Float64 # time window [hr]
    Δt::Float64 # time length of each step [hr]
    tend::Float64 = t0+Tw # final time [hr]
    # dTime::Array = collect(t0:Δt:tend)*3600 # discrete time array [s]
    dTime::Array = collect(t0:Δt:tend) # discrete time array [hr]
    steps::Int64 # number of times to move the window
    num_samples::Int64 # number of samples for RandomFieldOpt
    costWeights::Array=[1 1000 1000 0.0]; # [Wgrid WSoCDep Wtess Wπ]; array with the corresponding weigths for each cost
    # inputs for build_data
    season::String = "summer" # "summer" or "winter"
    profType::String # "daily" or "weekly" or "biweekly"
    loadType::String # "GV" or "mffbas" or "base_models"
    year::Int64 = 2023
    cellID::String = "SYNSANYO" # cell ID for the battery packs
end

@with_kw mutable struct driveDataRFO  # <:DriveDataType structure of sets
    # Driving info
    # Consumed power
    Pdrive::Vector{Vector{Float64}} # matrix of power consumption ℝ^{num_samples}
    # arrival and departure times
    SoCdep::Float64 # desired SoC for departure [p.u.]
    γ::Vector{Vector{Float64}} # matrix of availability of the EV ℝ^{num_samples x t}. Each element is a timeseries.
    tDep::Vector{Vector{Float64}} # matrix of departure times ℝ^{num_samples}
    tArr::Vector{Vector{Float64}} # matrix of arrival times ℝ^{num_samples}
end

@with_kw mutable struct EVDataRFO <:StorageAssetData # structure of sets
    carBatteryPack::BESSData # battery pack
    driveInfo::driveDataRFO # driving information
end

function rejCriteriaEnergy(μDrive, σDrive, tDep, tArr, Tconn, Ns, Np, vmax, Q0, chgPmax)
    Pdrive = rand(truncated(Normal(μDrive, σDrive); lower = 0.01), length(tDep));
    Ereq = Pdrive .* (tArr .- tDep) # energy required for each session [kWh]
    Emax = chgPmax .* Tconn # max energy that can be charged [kWh]
    Eev = Ns .* Np .* vmax .* Q0 .* 1e-3 * 0.8 # 80% energy of the EV battery [kWh]
    # Check if the energy required is less than the max energy that can be charged
    # if not adjust the demanded Pdrive
    for i ∈ 1:length(Ereq)
        if Ereq[i] > Emax[i] || Ereq[i] > Eev
            # pick the lowest bound between charger and EV battery
            println("Energy required for session $i is too high")
            if Emax[i] < Eev
                # adjust the power to the max that can be charged
                println("Adjusting power to the max that can be charged")
                println("From $(Ereq[i]) kWh to $(Emax[i]) kWh")
                Pdrive[i] = Emax[i] / (tArr[i] - tDep[i])

            else
                println("Adjusting power to the max that can be charged")
                println("From $(Ereq[i]) kWh to $(Eev) kWh")
                Pdrive[i] = Eev / (tArr[i] - tDep[i])
            end
        end
    end
    return Pdrive, Ereq, Emax
end

# to generate samples
function availabilityEV(ns, # number of samples
    fs::Int64 = 4, # samples per hour
    μDrive::Float64 = 3.5, # mean of Pdrive [kW]
    σDrive::Float64 = 1.5, # standard dev. of Pdrive [kW]
    Ns::Int64 = 100, # number of series cells
    Np::Int64 = 25, # number of parallel Branches
    vmax::Float64 = 4.2, # max voltage [V]
    Q0::Float64 = 5.2, # initial capacity [Ah]
    chgPmax::Float64 = 17.5, # EV charger limits [kW]
    type::String = "rfo", # type of availability
    )
    @assert type ∈ ["rfo", "mean"] "Invalid type of availability"

    ndays=Int(floor(ns/fs/24)); # number of days

    # First, we create availability vectors for each EV in the disc t-domain.
    γ = ones(ns)

    # using the data from Elaadusing Serialization
    # Deserialize the mixture model
    open("../data/gmmElaadFit.dat", "r") do f
        global gmm = deserialize(f) # Gaussian Mixture Model
    end
    # load the lookup table of the connection times
    μtCon_tarr_df = CSV.read("../data/input/Elaad Data/Data downloaded/mean-session-length-per.csv",
                        DataFrame,silencewarnings=true);
    # sort following the arrival times
    sort!(μtCon_tarr_df, "Arrival Time")
    # add a column with the arrival time in hs
    μtCon_tarr_df.tArr = collect(0:0.5:23.5)
    gmmt = truncated(gmm, 0, 23.5) # truncate the GMM to the limits
    # get the median of the GMM
    if type == "mean"
        tArr  = [mean(gmm) for i ∈ 1:(ndays+1)]
    else
        tArr = rand(gmmt, ndays+1)
    end
    # Interpolate mean session length for arrival times
    tCon_interp = linear_interpolation(μtCon_tarr_df.tArr, μtCon_tarr_df.home)
    Tconn = tCon_interp.(tArr) # session length
    tDep = tArr .+ Tconn

    # Adjust departure times if outside the limits
    tDep = [td > 23.5 ? td - 23.5 : td for td ∈ tDep]

    # Adjust first arrival and last departure times
    tDep = tDep[1:end-1]
    tArr = tArr[2:end]

    # Ensure departure time is before arrival time and not 0
    tDep = [t == 0 ? 0.5 : t for t in tDep]
    tArr = [tDep[i] > t ? tDep[i] + 1.0 : t for (i, t) in enumerate(tArr)]

    # Create the availability signal
    for day in 1:ndays
        # Determine the time indices corresponding to arrival and departure for this day
        t = collect(0:1/fs:(24-1/fs))
        # get the index of the departure and arrival times
        depIdx = findmin(abs.(tDep[day] .- t))[2]
        arrIdx = findmin(abs.(tArr[day] .- t))[2]
        # modify the index to be in the range of the time series, to avoid modifying supports
        tDep[day] = t[depIdx]
        tArr[day] = t[arrIdx]
        # Correct for the day
        depIdx = depIdx + (day-1)*24*fs
        arrIdx = arrIdx + (day-1)*24*fs
        # Mark the time series as parked during the parked interval for this day
        if depIdx <= arrIdx
            γ[depIdx:arrIdx] .= 0
        end
    end

    # Since we removed the the first element of tArr,
    # the session length is the same as the first tDep append
    # the first element of tDep in the first position of Tconn
    # Tconn = [tDep[1]; Tconn]; Tconn = Tconn[1:end-1];
    Tconn = [24. + tDep[i] - tArr[i-1] for i ∈ 2:ndays]
    Tconn = [tDep[1]; Tconn]

    # Create the driving signal
    Pdrive, Ereq, Emax = rejCriteriaEnergy(μDrive, σDrive, tDep, tArr, Tconn, Ns, Np, vmax, Q0, chgPmax)

    return γ, tDep, tArr, Pdrive, Tconn, Ereq, Emax;
end

function getResultsRFO(model)
    x = all_variables(model)
    names = [name(var) for var in x]

    # remove all the variables with "" name
    id = findall(isequal(""), names)
    ic = [i for i ∈ 1:length(x) if i ∉ id]
    # select x not in id
    x = x[ic]
    values = [value(var) for var in x]
    names = [name(var) for var in x]
    # create a dictionary with the values and the names
    results = Dict(zip(names, values))
    return results
end

################ MODEL CREATION ################
function build_data(; nEV::Int64=2, # number of EVs in the system
    season::String = "winter", # "summer" or "winter"
    profType::String, # "weekly" or "daily"
    loadType::String, # "GV" or "mffbas" or "base_models"
    year::Int64,
    fs::Int64 = 4, # samples per hour
    cellID::String = "SYNSANYO", # cell ID for the battery packs
    stoch_num_samples::Int64 = 1 # number of samples for the stochastic optimization
    )
    # This function builds the data for the optimization problem.
    # The inputs are:
    # - season: "summer" or "winter"
    # - profType: "weekly" or "daily"
    # - loadType: "GV" or "mffbas" or "base_models"
    # - year: the year of the simulation
    # The output is a dictionary that contains the models for all the different devices in the Multi-carrier Energy System.
    # Check inputs
    # @assert typeof(nEV) == UnitRange{Int64} "nEV must be a range of integers"
    @assert typeof(nEV) == Int64 "max nEV must be a integer"
    @assert season ∈ ["summer", "winter"] "Invalid season"
    @assert profType ∈ ["biweekly", "weekly", "daily", "yearly"] "Invalid profile type"
    @assert loadType ∈ ["GV", "mffbas", "base_models"] "Invalid load type"

    ## PV model
    if loadType == "base_models" # From Joel's Base models
        MPPT = CSV.read("../data/input/Base models/PV_15min.csv", DataFrame, header=false) # measurement of the max. power point tracking
    else
        # From Ibrahim's PV model for the NL
        pvData = matread("../data/input/Arnhem_results_Saffirio_Diab.mat")
        MPPT=fromSecTo15min(pvData["P_DC_sec"]);
    end
    MPPT=Array(MPPT);
    Npv=30; # number of pv panels
    if profType ≠ "yearly"
        # get the daily seasonal profile for the PV
        MPPT=getSeasonalProfiles(MPPT, type = profType)[season];
        # for biweekly profiles we need to repeat the weekly profile twice and append the first day to the end
        if profType == "biweekly" MPPT=repeat(MPPT[1:(end-fs*24)], outer=2); append!(MPPT, MPPT[1:fs*24]) end
    end

    spvModel = SPVData(MPPTData = MPPT/1e3*Npv);

    ## BESS model
    # A battery pack model is composed of three parts:
    # - its general information, contained in the GenInfo type.
    # - its performance submodel, contained in the PerfParameters type. This describes the SoC and terminal voltage of a cell.
    # - its ageing submodel, contained in the AgingParameters type. This describes the evolution of the performance sub-model parameters.

    # Read data from the E2 of the ESCtoolbox from Plett.
        # OCV non linear data model
        bessOCV=JSON3.read(open("../data/input/cell_models/$cellID-modelocv.json", "r"),
                    Dict{String, Vector{Float64}});
        # Dynamic ECM model, with np=1
        bessDYN=JSON3.read(open("../data/input/cell_models/$cellID-modeldyn-no-hys.json", "r"),
                    Dict{String, Union{Vector{Float64}, Vector}});
        # bessOCVmat = matread("../data/input/cell_models/E2model-ocv.mat")
        # # Dynamic ECM model
        # bessDYNmat = matread("../data/input/cell_models/E2model.mat")
        # # Unpack
        # bessOCVmat=bessOCVmat["model"];
        # bessDYNmat=bessDYNmat["model"];

        # Bucket model info
        Q0 = bessOCV["OCVQ"]; Q0 = mean(Q0); # [Ah/cell]
        η = bessOCV["OCVeta"]; η = mean(η); # [p.u.]

        # ECM info
        T=bessDYN["temps"]; T=T[1,:]; # [C]
        R0Param=bessDYN["R0Param"]; # [Ohm]
        RParam=bessDYN["RParam"]; # [Ohm]
        RCParam=bessDYN["RCParam"]; # τ [s]
        Tind25=findall(x->x==25, T)

    # Heliox 43kWh battery pack seems a little bit too much lets use a 20kWh pack.
    # PowerLim = [-17, 17]; P0 = 0; # Power limits and initial condition [kW].
    PowerLim = [-25, 25]; P0 = 0; # Power limits and initial condition [kW].
    SoCLim = [0.2, 0.95]; SoC0 = 0.5; # SoC limits and initial condition [p.u.].
    ηC = 0.95; # Charger efficiency [p.u.].
    # State of Health
    SoHQ = 1; SoHR0 = R0Param[Tind25][1];
    if cellID == "SYNSANYO"
        Np = 10; Ns = 100; # Branches in parallel and series cells per branch.
        ocv_params=OCVlinearPerfParams()
        vLim = [2.8, 4.2]
        # Ageing submodel
        # aging_params=empAgingParams();
        aging_params=JinAgingParams();
    elseif cellID == "A123"
        Np = 25; Ns = 110; # Branches in parallel and series cells per branch.
        # first option,
        # ocv_params.ocvLine = [2.5, 1.1]
        # second option from 20% to 95% SoC
        ocv_params=OCVlinearPerfParams(ocvLine = [3.2, 0.2105])
        # Ageing submodel
        # aging_params=empAgingParams();
        Cell = Construct("A123");
        aging_params=JinAgingParams(Rs = Cell.Neg.Rs,
                            An = Cell.Const.CC_A,
                            Ln = Cell.Neg.L,
                            z100p = Cell.Neg.θ_100,
                            z0p = Cell.Neg.θ_0,
                            # εₑ0 = Cell.Neg.ϵ_e,
                            t⁺₀ = Cell.Const.t_plus,
                            # DeRef= Cell.Neg.De,
                            ce_avg = Cell.Const.ce0,
                            # ce_max=Cell.Const.ce0,
                            σn = Cell.Neg.σ,
                            εₛ = Cell.Neg.ϵ_s, # check
                            );
    end
    initVal=6e-2; # Cost info [USD/kWh/day].
    # General info definition
    gen_params=Generic(PowerLim, P0, SoCLim, SoC0, Q0, SoHQ, SoHR0, Np, Ns, η, ocv_params, vLim, ηC, initVal);
    # Performance submodel
    perf_params=bucketPerfParams();
    # perf_params=ECMPerfParams(R0Param=R0Param[Tind25], RParam=RParam[Tind25], RCParam=RCParam[Tind25]);
    # wrap everything in a BESSData type
    # battModel = BESSData(gen_params, perf_params, aging_params, initVal);
    battModel = BESSData(gen_params, perf_params, aging_params, cellID);

    ## EV MODEL
    # An electric vehicle model is composed of two parts:
    # - its battery pack, contained in a BESSData type.
    # - its driving submodel, contained in the driveData type. This describes the availability,
    # times of departure and arrival, and the reference SoC of the EV.
    # Battery pack definition
    PowerLim = [-12.5, 12.5] #check
    SoCLim = [0.1, 1.]; # SoC limits [p.u.]
    # The battery pack has to be around 400Vdc and 50kWh
    if cellID == "SYNSANYO"
        Ns = 100; Np = 25;
    elseif cellID == "A123"
        Ns = 110; Np = 61;
    end
    P0=[0, 0]; # initial
    Q0 = Q0.*ones(nEV); # [Ah/cell]
    SoC0 = [0.6, 0.8]; # initial
    vt0=[gen_params.OCVParam.ocvLine[1]+gen_params.OCVParam.ocvLine[2].*SoC0[n] for n ∈ 1:nEV]
    gen_params = [Generic(PowerLim, P0[n], SoCLim, SoC0[n], Q0[n], SoHQ, SoHR0, Np, Ns, η, ocv_params, vLim, ηC, initVal) for n ∈ 1:nEV]
    # perf_params=[ECMPerfParams(R0Param=R0Param[Tind25], RParam=RParam[Tind25],
    #             RCParam=RCParam[Tind25], vt0=vt0[n]) for n ∈ 1:nEV]
    perf_params = [bucketPerfParams() for n ∈ 1:nEV]
    batteryPack = [BESSData(gen_params[n], perf_params[n], aging_params, cellID) for n ∈ 1:nEV]

    # Driving information definition
    μD = 3.5; σD=1.5; depParam=[6, 12]; arrParam=[1, 8]; # Parameters for the Gaussian distributions
    # availability
    # avObj = [availabilityEV(length(MPPT), 4) for i ∈ 1:stoch_num_samples, n ∈ 1:nEV];
    # for the deterministic case
    # avObj = [availabilityEV(length(MPPT), 4,  μD, σD, Ns, Np, vLim[2], Q0[n], PowerLim[2] * ηC, "mean") for i ∈ 1:stoch_num_samples, n ∈ 1:nEV];
    # for the RFO case
    avObj = [availabilityEV(length(MPPT), 4,  μD, σD, Ns, Np, vLim[2], Q0[n], PowerLim[2] * ηC) for i ∈ 1:stoch_num_samples, n ∈ 1:nEV];
    γ = [avObj[:,n][i][1] for i ∈ 1:stoch_num_samples, n ∈ 1:nEV];
    tDep = [avObj[:,n][i][2] for i ∈ 1:stoch_num_samples, n ∈ 1:nEV];
    tArr = [avObj[:,n][i][3] for i ∈ 1:stoch_num_samples, n ∈ 1:nEV];
    Pdrive = [avObj[:,n][i][4] for i ∈ 1:stoch_num_samples, n ∈ 1:nEV];
    refSoC=[0.85, 0.85]; # user requirement
    # drive_info = [driveDataRFO(μD, σD, refSoC[n], γ[:,n], tDep[:,n], tArr[:,n]) for n in 1:nEV]
    drive_info = [driveDataRFO(Pdrive[:,n], refSoC[n], γ[:,n], tDep[:,n], tArr[:,n]) for n in 1:nEV]
    # wrap everything in a EVData type
    evModel = [EVDataRFO(batteryPack[n], drive_info[n]) for n in 1:nEV]

    # Solar thermal model
    Pn=0.5; capex= 1500; ηST = 0.6;
    stModel = ElectroThermData(Pn, capex, ηST);

    # TESS model
    tessModel = TESSData(); tessModel.SoC0=0.4;
    # Power Electronic Interface model
    peiModel = peiData();

    ## Grid Model
    # The grid is represented by:
    # - λ energy prices. [buy; sell]
    # - loadE electrical load measurement
    # - loadTh thermal load measurement
    # for anual profiles the length is (365*24*fs)+1=35041.

    ## pick file paths, read and convert to array
    # for the prices
    # Raw EPEX FTP server data
    pricePath = "../data/input/EPEX/auction_spot_prices_netherlands_$year.csv"
    spotPriceDF = CSV.read(pricePath, DataFrame, delim=',', header=2);

    #= for IECON 2023 =#
        # summary data
        # pricePath = "../data/input/EPEX/summaryPrice_Winter.csv";
        # spotPriceDF = CSV.read(pricePath, DataFrame, delim=',', header=1);
        # pricePath = "../data/input/EPEX/summaryPrice_Summer.csv";
        # spotPriceDF = CSV.read(pricePath, DataFrame, delim=',', header=1);

    # for the electrical load
    if loadType == "GV"
        # processed synthezided load profile 1 year
        loadEPath = "../data/input/GV/Load_1.csv";
        loadE = CSV.read(loadEPath, DataFrame, header = false) # electric load
        # loadE = CSV.read("../data/input/GV/Load_1.csv", DataFrame, header=false) # electric load
        loadE= Vector(loadE[!,1]);
    elseif loadType == "mffbas"
        # From Market Facilitation Forum (MFF) and the Beheerder Afspraken Stelsel (BAS) i.e. mffbas
        loadEPath = "../data/input/mffbas/summaryE1_$year.csv"
        #= for IECON 2023 =#
        # loadEPath = "../data/input/mffbas/summaryE1_mean_Winter.csv"
        # loadEPath = "../data/input/mffbas/summaryE1_mean_Summer.csv"
        loadE = CSV.read(loadEPath, DataFrame) # electric load
        loadE= Vector(loadE[!,2]);

    elseif loadType == "base_models" # Joel's base models
        loadEPath = "../data/input/Base models/Load_Profile_15min.csv";
        loadE = CSV.read(loadEPath, DataFrame, transpose=true, header=false)
        loadE= Vector(loadE[!,1]);
    end
    # normalize loadE
    loadE=loadE./maximum(loadE);
    # peak of 5kW
    loadE=loadE*5.;
    # for the thermal load
    if loadType == "base_models" || loadType == "mffbas"
        loadThPath= "../data/input/Base models/Thermal_load_15min.csv";
        loadTh = CSV.read(loadThPath, DataFrame, header = false) # thermal load
        loadTh= Vector(loadTh[!,1]); loadTh=loadTh*1e-3;
    elseif profType == "daily" # Nikos' models
        loadThPath_folder = "../data/input/model_Thermal_Nikos/";
        loadThPath = loadThPath_folder * "Residential_$(season)_heating_consumption_dailyAvg.csv";
        loadTh = CSV.read(loadThPath, DataFrame)
        loadTh= Vector(loadTh.P_con);# thermal load in kW and min resolution
        # downsample to 15 min with average every 4 samples
        loadTh = [mean(loadTh[i:i+14]) for i ∈ 1:15:length(loadTh)];
        loadTh = repeat(loadTh, outer=365); # repeat for the whole year
    end

    # check length of loadE, loadTh
    length(loadE) == (365*24*fs+1) ? nothing : loadE = [loadE[1]; loadE];
    length(loadTh) == (365*24*fs+1) ? nothing : loadTh = [loadTh[1]; loadTh];

    # Data processing (upsampling, seasonal patterns, etc.)
    if profType ≠ "yearly"
        priceData = processPrices(spotPriceDF; type="raw", profType = profType, season = season);
        # get the profile for the loads
        loadE=getSeasonalProfiles(loadE, type = profType)[season];
        loadTh=getSeasonalProfiles(loadTh, type = profType)[season];
        # for biweekly profiles we need to repeat the weekly profile twice and append the first day to the end
        if profType == "biweekly"
            loadE=repeat(loadE[1:(end-fs*24)], outer=2); append!(loadE, loadE[1:fs*24])
            loadTh=repeat(loadTh[1:(end-fs*24)], outer=2); append!(loadTh, loadTh[1:fs*24])
        end
    else
        priceData = processPrices(spotPriceDF; type="raw", profType = profType);
    end
    # Grid connection limits
    gridModel = gridData([-17, 17], 0.9, priceData, loadE, loadTh);

    # Heat pump model
    # only a uniderectional (heating) HP for now
    Pn=4; capex= 500; ηHP = 3;
    # The initial condition follow the power balance
    # Phpe0=(loadTh[1]-MPPT[1]*ηST)/ηHP;
    hpModel = ElectroThermData(Pn, capex, ηHP);

    data=Dict("SPV"=>spvModel, "BESS"=>battModel, "EV"=>evModel,
        "ST"=>stModel, "HP"=>hpModel, "TESS"=>tessModel,
        "grid"=>gridModel, "PEI"=>peiModel);
    return data
end

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
    # # simulated forecast
    # MPPTnoisy = zeros(size(MPPTmeas));
    # MPPTnoisy[MPPTmeas .> 0] = [rand(Uniform(0.5*MPPTmeas[tt],1.1*MPPTmeas[tt])) for tt ∈ findall(MPPTmeas .> 0)]
    # # Add forecast
    # @parameter_function(model, PpvMPPT == (t) -> MPPTnoisy[zero_order_time_index(Dt, t)])
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
    PbessMin = PowerLim[1]; # Min power [kW]
    # Pbess0=P0; # Initial power [kW]
    SoCbessMin = SoCLim[1]; # Min State of Charge [p.u.]
    SoCbessMax = SoCLim[2]; # Max State of Charge [p.u.]
    SoCbess0=SoC0; # Initial SoC [p.u.]
    vbessMin = vLim[1]; vbessMax = vLim[2]; # Voltage limits [V]

    # @unpack_Generic GenInfo
    @unpack PowerLim, P0, SoCLim, SoC0, initQ, SoHQ, SoHR0, Np, Ns, η, OCVParam, vLim, ηC  = GenInfo
    @unpack ocvLine = OCVParam;
    Npbess = Np; Nsbess = Ns;
    Qbess0 = initQ*SoHQ; ηbess = η;
    # aOCV=ocvLine[1]; bOCV=ocvLine[2];
    crate = PbessMax * 1e3 / (Nsbess * Npbess * Qbess0 * vLim[2]); # C-rate

    # Add variables
    # OCVbess0 = aOCV+bOCV*SoCbess0;
    @variables(model, begin
        1e-2 ≤ Qbess ≤ 50, (start = 1e-2) # BESS capacity Q0 [kWh]
        Pbess[i ∈ 1:num_samples], (start = 0.), Infinite(t) # output power [kW]
        SoCbess[i ∈ 1:num_samples], Infinite(t) # State of Charge [kWh]
    end);

    t1 = t0; # time of the day
    @constraints(model, begin
        # Bidirectional power flow, ensuring only export or import
        [i ∈ 1:num_samples], - Qbess*crate ≤ Pbess[i]
        [i ∈ 1:num_samples], Pbess[i] ≤ Qbess*crate
        # State of Charge limits [kWh]
        [i ∈ 1:num_samples], SoCbess[i] ≥ SoCbessMin * Qbess
        [i ∈ 1:num_samples], SoCbess[i] ≤ SoCbessMax * Qbess
        # Initial conditions
        [i ∈ 1:num_samples], SoCbess[i](t0) ==  SoCbess0 .* Qbess
        # terminal condition
        # [i ∈ 1:num_samples], SoCbess[i](t1) ==  SoCbess[i](t1+24)
        # Transition function
        [i ∈ 1:num_samples], ∂.(SoCbess[i], t) .== -Pbess[i]

    end);
    #= BINARY VARIABLE STUFF
        @variables(model, begin
            1e-2 ≤ Qbess ≤ 50, (start = 1e-2) # BESS capacity Q0 [kWh]
            Pbess[i ∈ 1:num_samples], (start = 0.), Infinite(t) # output power [kW]
            bPbess[i ∈ 1:num_samples], Infinite(t), Bin # Binary variable for output power
            SoCbess[i ∈ 1:num_samples], Infinite(t) # State of Charge [kWh]
            # Dummy variables for bidirectional flow
            0 ≤ PbessPos[i ∈ 1:num_samples], Infinite(t) # Pbess^+ out power
            PbessNeg[i ∈ 1:num_samples] ≤ 0, Infinite(t) # Pbess^- in power
            # PerfModel variables
            # Nsbess * vbessMin ≤ OCVbess[i ∈ 1:num_samples] ≤ Nsbess * vbessMax, Infinite(t),(start=OCVbess0) # open circuit voltage of the cell
            # aOCV ≤ vtbess ≤ aOCV+bOCV, Infinite(t),(start=OCVbess0)  # terminal voltage of the cell
            # ibess[i ∈ 1:num_samples], Infinite(t), (start=0.0) # total current per branch
            # iR1bess, Infinite(t), (start=0.0) # pole current
        end);

        # fix(Qbess, Nsbess * Npbess * Qbess0 * vLim[2] *1e-3, force = true) # uncomment to simulate the deterministic reponse
        # fix(Qbess, 1e-2, force = true) # uncomment to simulate the deterministic reponse
        t1 = t0; # time of the day
        @constraints(model, begin
            # Bidirectional power flow, ensuring only export or import
            [i ∈ 1:num_samples], PbessNeg[i] + PbessPos[i] == Pbess[i]
            [i ∈ 1:num_samples], - bPbess[i]*Qbess*crate ≤ PbessNeg[i]
            [i ∈ 1:num_samples], PbessPos[i] ≤ (1-bPbess[i])*Qbess*crate
            # State of Charge limits [kWh]
            [i ∈ 1:num_samples], SoCbess[i] ≥ SoCbessMin * Qbess
            [i ∈ 1:num_samples], SoCbess[i] ≤ SoCbessMax * Qbess
            # Initial conditions
            [i ∈ 1:num_samples], SoCbess[i](t0) ==  SoCbess0 .* Qbess
            # terminal condition
            # [i ∈ 1:num_samples], SoCbess[i](t1) ==  SoCbess[i](t1+24)
            # ibess == 1e3*Pbess/Npbess/Nsbess/vtbess # current per branch. 1e3 to convert kW->W
            # [i ∈ 1:num_samples], ibess[i] == 1e3*Pbess[i]/Npbess/Nsbess/OCVbess[i] # current per branch. 1e3 to convert kW->W
            # Transition function
            [i ∈ 1:num_samples], ∂.(SoCbess[i], t) .== -(ηbess * bPbess[i] + (1-bPbess[i]))*Pbess[i]
            # Output equations
            # [i ∈ 1:num_samples], OCVbess[i] .== aOCV+bOCV*SoCbess[i] # Linear OCV model
        end);
    =#
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
    agingParams = Vector{AgingParams}(undef, nEV)
    # μDrive = zeros(nEV)
    # σDrive = zeros(nEV)
    SoCdep = zeros(nEV)
    γ = Matrix{Vector{Float64}}(undef, num_samples, nEV)
    Pdrive = Matrix{Vector{Float64}}(undef, num_samples, nEV)
    # tDep = []

    for n in 1:nEV
        @unpack GenInfo, PerfParameters, AgingParameters = data["EV"][n].carBatteryPack
        # @unpack_Generic GenInfo
        @unpack PowerLim, P0, SoCLim, SoC0, initQ, SoHQ, SoHR0, Np, Ns, η, OCVParam, vLim, ηC  = GenInfo
        @unpack ocvLine = OCVParam

        performanceParams[n] = PerfParameters
        agingParams[n] = AgingParameters

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
        # μDrive[n] = data["EV"][n].driveInfo.μDrive
        # σDrive[n] = data["EV"][n].driveInfo.σDrive
        SoCdep[n] = data["EV"][n].driveInfo.SoCdep
        # γ[n, :] = data["EV"][n].driveInfo.γ[it0:itend]
        γ[:, n] = data["EV"][n].driveInfo.γ # is missing some info with the it0:itend, check later
        # γ[:, n] = [data["EV"][n].driveInfo.γ] # is missing some info with the it0:itend, check later
        Pdrive[:,n] = data["EV"][n].driveInfo.Pdrive
    end

    # Add variables
    # Pev > 0 -> out power and Pev < 0 -> in power
    OCVev0 = aOCV .+ bOCV .* SoCev0;
    Eev0 = Qev0.*Npev.*Nsev.*vMax / 1000; # [kWh]
    @variables(model, begin
        PevMin[n] .≤ Pev[i ∈ 1:num_samples, n ∈ 1:nEV] .≤ PevMax[n], Infinite(t)  # EV charger power
        PevMin[n] .≤ PevTot[i ∈ 1:num_samples, n ∈ 1:nEV] .≤ PevMax[n], Infinite(t)  # total power of each EV, driving+V2G
        SoCevMin[n] .* Eev0[n] .≤ SoCev[i ∈ 1:num_samples, n ∈ 1:nEV] .≤ SoCevMax[n] .* Eev0[n], (start = SoCev0[n] .* Eev0[n]), Infinite(t) # State of Charge [kWh]
    end);

    #= BINARY VARIABLE STUFF
        @variables(model, begin
            Pev[i ∈ 1:num_samples, n ∈ 1:nEV], Infinite(t)  # EV charger power
            bPev[i ∈ 1:num_samples, n ∈ 1:nEV], Infinite(t), Bin  # EV charger power binary
            PevTot[i ∈ 1:num_samples, n ∈ 1:nEV], Infinite(t)  # total power of each EV, driving+V2G
            SoCevMin[n] .* Eev0[n] .≤ SoCev[i ∈ 1:num_samples, n ∈ 1:nEV] .≤ SoCevMax[n] .* Eev0[n], (start = SoCev0[n] .* Eev0[n]), Infinite(t) # State of Charge [kWh]
            # Nsev[n].*aOCV[n] .≤ OCVev[i ∈ 1:num_samples, n ∈ 1:nEV] .≤ Nsev[n] .* (aOCV[n]+bOCV[n]), Infinite(t) # open circuit voltage of the cell
            # iev[i ∈ 1:num_samples, n ∈ 1:nEV], Infinite(t)  # total current per branch
            # Dummy variables for bidirectional flow
            0 ≤ PevPos[i ∈ 1:num_samples, n ∈ 1:nEV], Infinite(t) # Pev^+ out power
            PevNeg[i ∈ 1:num_samples, n ∈ 1:nEV] ≤ 0, Infinite(t) # Pev^- in power
        end);
        @constraints(model, begin
            [i ∈ 1:num_samples, n ∈ 1:nEV], PevNeg[i,n] + PevPos[i,n] == Pev[i,n]
            [i ∈ 1:num_samples, n ∈ 1:nEV], bPev[i,n]*PevMin[n] ≤ PevNeg[i,n]
            [i ∈ 1:num_samples, n ∈ 1:nEV], PevPos[i,n] ≤ (1-bPev[i,n])*PevMax[n]
            [i ∈ 1:num_samples, n ∈ 1:nEV], SoCev[i,n](t0) == SoCev0[n] .* Eev0[n] # Initial conditions
        end);
        @constraints(model, begin
            [i ∈ 1:num_samples, n ∈ 1:nEV], γf[i,n].*Pev[i,n] + Pdrivef[i,n] - PevTot[i,n] .== 0 # power balance
            [i ∈ 1:num_samples, n ∈ 1:nEV], ∂.(SoCev[i,n], t) .== -(ηev[n] * bPev[i,n] + (1-bPev[i,n]))*PevTot[i,n]
            # [i ∈ 1:num_samples, n ∈ 1:nEV], OCVev[i,n] .== aOCV[n]+bOCV[n]*SoCev[i,n] # linear voltage model
            # [i ∈ 1:num_samples, n ∈ 1:nEV], iev[i,n] .== 1e3*PevTot[i,n]/Npev[n]/Nsev[n]/OCVev[i,n] # current per branch
        end);
    =#

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
    # PstRated = data["ST"].RatedPower; # rated power of the panels
    ηST = data["ST"].η; # conversion factor from Electric PV to thermal

    # Extract data
    MPPTmeas = data["SPV"].MPPTData[it0:itend];
    @parameter_function(model, Pst == (t) -> ηST*MPPTmeas[zero_order_time_index(Dt, t)])
    # # simulated forecast
    # MPPTnoisy = zeros(size(MPPTmeas));
    # MPPTnoisy[MPPTmeas .> 0] = [rand(Uniform(0.5*MPPTmeas[tt],1.1*MPPTmeas[tt])) for tt ∈ findall(MPPTmeas .> 0)]
    # @parameter_function(model, Pst == (t) -> ηST*MPPTnoisy[zero_order_time_index(Dt, t)])
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
    # Ptess0 = data["TESS"].P0; # Initial Power [p.u.]
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

    #= BINARY VARIABLE STUFF
        @variables(model, begin
            SoCtessMin .≤ SoCtess[i ∈ 1:num_samples] .≤ SoCtessMax, Infinite(t) # State of Charge
            Ptess[i ∈ 1:num_samples], Infinite(t) # Thermal power
            bPtess[i ∈ 1:num_samples], Infinite(t), Bin # Binary variable for TESS power
            0 ≤ PtessPos[i ∈ 1:num_samples], Infinite(t) # Ptess^+ out power
            PtessNeg[i ∈ 1:num_samples] ≤ 0, Infinite(t) # Ptess^- in power
        end);

        # Dummy variables for bidirectional power flow, ensuring only export or import
        @constraints(model, begin
            [i ∈ 1:num_samples], bPtess[i]*PtessMin ≤ PtessNeg[i]
            [i ∈ 1:num_samples], PtessNeg[i] + PtessPos[i] .== Ptess[i]
            [i ∈ 1:num_samples], PtessPos[i] ≤ (1-bPtess[i])*PtessMax
        end);

        # Bucket model
        # @constraint(model, [i ∈ 1:num_samples], ∂.(SoCtess[i], t) .== -ηtess*Ptess[i]/Qtess/3600);
        @constraint(model, [i ∈ 1:num_samples], ∂.(SoCtess[i], t) .== -ηtess*Ptess[i]/Qtess);
    =#

    return model;
end;

function tess!(model::InfiniteModel, sets::modelSettingsRFO, data::Dict) # stationary battery pack
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
    # Ptess0 = data["TESS"].P0; # Initial Power [p.u.]
    SoCtessMin = data["TESS"].SoCLim[1]; # Min State of Charge [p.u.]
    SoCtessMax = data["TESS"].SoCLim[2]; # Max State of Charge [p.u.]
    SoCtess0 = data["TESS"].SoC0; # Initial State of Charge [p.u.]
    ηtess = data["TESS"].η; # thermal efficiency

    # Add variables
    @variables(model, begin
        SoCtessMin ≤ SoCtess ≤ SoCtessMax, Infinite(t) # State of Charge
        PtessMin ≤ Ptess ≤ PtessMax, Infinite(t) # Thermal power
    end);

    # Initial conditions
    @constraints(model, begin
        SoCtess(t0) .== SoCtess0
        ∂.(SoCtess, t) .== -ηtess*Ptess/Qtess # Bucket model
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
    ηg = data["grid"].η; # converter efficiency

    # Add variables
    @variables(model, begin
        PgMin ≤ Pg[i ∈ 1:num_samples] ≤ PgMax, Infinite(t)  # grid power
    end)

    # Dummy variables for bidirectional power flow, ensuring only export or import
    #= BINARY VARIABLE STUFF
        @variables(model, begin
            Pg[i ∈ 1:num_samples], Infinite(t)  # grid power
            bPg[i ∈ 1:num_samples], Infinite(t), Bin # Binary variable for Grid power
            0 ≤ PgPos[i ∈ 1:num_samples], Infinite(t) # Pg^+ out/buy power
            PgNeg[i ∈ 1:num_samples] ≤ 0, Infinite(t), (start=0) # Pg^- in/sell power
        end)

        # Dummy variables for bidirectional power flow, ensuring only export or import
        @constraints(model, begin
            [i ∈ 1:num_samples], PgPos[i] .≤ (1-bPg[i])*PgMax
            [i ∈ 1:num_samples], bPg[i]*PgMin .≤ PgNeg[i]
            # PgNeg + PgPos .== Pg
            [i ∈ 1:num_samples], PgNeg[i] * (1/ηg) + PgPos[i] * ηg .== Pg[i]
        end)
    =#
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
    # # white noise to simulate a forecast
    # εₗᵗʰ= randn(Int(length(loadTh)*Δt/3600)) .* 0.05 # 50W
    # εₗᵗʰ = repeat(εₗᵗʰ, inner = Int(3600/Δt))
    # loadTh = loadTh .+ εₗᵗʰ; # add noise
    @parameter_function(model, Plt == (t) -> loadTh[zero_order_time_index(Dt, t)])

    # When there´s no FCR then there´s no decision variable Ppve hence Pst can just be a @parameter_function.
    # Extract params
    ηHP=data["HP"].η; # conversion factor from Electric to thermal heat pump
    # Thermal Power balance
    model[:thBalance]=@constraint(model, [i ∈ 1:num_samples], model[:Pst] + model[:Phpe][i] .* ηHP + model[:Ptess][i] .==  Plt);
    return model;
end;

function gridThermal!(model::InfiniteModel, sets::modelSettingsRFO, data::Dict) # thermal grid
    # Load data
    Dt = sets.dTime; num_samples = sets.num_samples;
    t = model[:t];
    t0=supports(t)[1]; Δt=supports(t)[2]-supports(t)[1];
    # make new time window
    it0 = round(Int,(t0/Δt));
    itend = it0+length(Dt)-1;
    # make InfiniteOpt compatible Plt(t) work for arbitrary times
    loadTh = data["grid"].loadTh[it0:itend];
    # # white noise to simulate a forecast
    # εₗᵗʰ= randn(Int(length(loadTh)*Δt/3600)) .* 0.05 # 50W
    # εₗᵗʰ = repeat(εₗᵗʰ, inner = Int(3600/Δt))
    # loadTh = loadTh .+ εₗᵗʰ; # add noise
    @parameter_function(model, Plt == (t) -> loadTh[zero_order_time_index(Dt, t)])

    # When there´s no FCR then there´s no decision variable Ppve hence Pst can just be a @parameter_function.
    # Extract params
    ηHP=data["HP"].η; # conversion factor from Electric to thermal heat pump
    # Thermal Power balance
    model[:thBalance]=@constraint(model, model[:Pst] + model[:Phpe] .* ηHP + model[:Ptess] .==  Plt);
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
    # white noise to simulate a forecast
    # εₗᵉ= randn(Int(length(loadElec)*Δt/3600)) .* 0.2 # 200W
    # εₗᵉ = repeat(εₗᵉ, inner = Int(3600/Δt))
    # loadElec = loadElec .+ εₗᵉ; # add noise
    # loadElec can't be negative
    # loadElec[loadElec .< 0] .= 0
    @parameter_function(model, Ple == (t) -> loadElec[zero_order_time_index(Dt, t)])

    # Power balance DC busbar
    # If we have EVs
    # if any(name.(all_variables(model)) .== "Pev[1,1]")
    #     Pev_sum = sum(model[:γf][i,n].*model[:Pev][i,n] for i ∈ 1:num_samples, n in 1:nEV);
    # else
    #     Pev_sum = 0
    # end
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
    # In order to stay consistent with the other DenseAxisArray we need matrices, so here hcat does that for us.
    # priceBuy=hcat(data["grid"].λ[it0:itend, 1].*1e-3/3600) # convert from €/MWh to €/kWs
    # priceSell=hcat(data["grid"].λ[it0:itend, 2].*1e-3/3600) # convert from €/MWh to €/kWs
    priceBuy=hcat(data["grid"].λ[it0:itend, 1].*1e-3) # convert from €/MWh to €/kWh
    priceSell=hcat(data["grid"].λ[it0:itend, 2].*1e-3) # convert from €/MWh to €/kWh
    # # simulated forecast
    # ελ = randn(Int(length(priceBuy)*Δt/3600)) .* 20*1e-3/3600 # 20 €/MWh noise
    # ελ = repeat(ελ, inner = Int(3600/Δt))
    # priceBuy = priceBuy .+ ελ; # add noise
    # priceSell = priceSell .+ ελ; # add noise
    # @parameter_function(model, λbuy == (t) -> priceBuy[zero_order_time_index(Dt, t)])
    # @parameter_function(model, λsell == (t) -> priceSell[zero_order_time_index(Dt, t)])
    @parameter_function(model, λ == (t) -> priceBuy[zero_order_time_index(Dt, t)])

    # CAPEX - Battery
    CAPEX = data["BESS"].GenInfo.initValue;

    # STAGE COST - Operation
    # Grid costs
    Wgrid = W[1]; # regularization factor for grid cost. max(λ)*max(P)
    # cgrid = Wgrid .* (model[:PgPos]*λbuy + model[:PgNeg]*λsell);
    cgrid = Wgrid .* model[:Pg] * λ;

    # Define penalty for not charging
    WSoCDep = W[2]
    pDep = (isempty(model[:ϵSoC]) ? 0 : WSoCDep .* [sum(model[:ϵSoC][i,n] .^ 2 for n ∈ 1:nEV) for i ∈ 1:num_samples])
    # pDep = (isempty(model[:ϵSoC]) ? 0 : WSoCDep .* [sum(-model[:ϵSoC][i,n] for n ∈ 1:nEV) for i ∈ 1:num_samples])

    # # Penalty/soft constraint for TESS overcharging
    # Wtess = W[3]; # penalty for TESS overcharging
    # SoCtess = model[:SoCtess]; SoCtessMax = data["TESS"].SoCLim[2];
    # @variable(model, auxTess[i ∈ 1:num_samples] ≥ 0, Infinite(t));
    # @constraint(model, [i ∈ 1:num_samples], auxTess[i] ≥ SoCtess[i] .- SoCtessMax);

    # # Penalty for excessive actions
    # xπ = Vector{GeneralVariableRef}();
    # for (~,var) ∈ enumerate([:Pbess, :Pev, :Phpe])
    #     append!(xπ, model[var]);
    # end
    # Wπ = W[4]; # penalty for excessive actions
    # cπ = Wπ*∫(sum(xπ[i].^2 for i ∈ eachindex(xπ)),t);

    # Stage cost
    # 𝒞 = (∫.(cgrid,t) .+ Wtess*∫.(auxTess,t) .+ cπ)./sum(Dt) .+ pDep;
    # 𝒞 = ∫.(cgrid,t) .+ Wtess*∫.(auxTess,t) .+ cπ .+ pDep;
    # 𝒞 = (∫.(cgrid,t))./sum(Dt) .+ pDep;
    𝒞 = ∫.(cgrid,t) .+ pDep;
    # Define objective function
    Qbess = model[:Qbess];
    @objective(model, Min, 1/num_samples * sum(𝒞[i] for i ∈ 1:num_samples) + CAPEX .* Qbess)
    # @objective(model, Min, 1/num_samples * sum(𝒞[i] for i ∈ 1:num_samples))
    return model;
end;
