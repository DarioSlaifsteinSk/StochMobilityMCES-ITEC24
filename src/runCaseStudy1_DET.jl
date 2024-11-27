cd(@__DIR__)

# Import all the necessary packages
using Random,
    InfiniteOpt,
    JuMP,
    Ipopt,
    JSON3,
    MAT,
    StatsBase,
    CSV,
    DataFrames,
    Revise,
    Distributions,
    Interpolations,
    GLMakie,
    CairoMakie,
    Makie
includet("../fns/EMSrfofns.jl")
includet("../fns/build_data.jl")
Random.seed!(1234);
## Define all the necessary functions
function solvePolicies(optimizer, # model optimizer
    sets::modelSettingsRFO, # number of EVs, discrete time supports, etc.
    data::Dict, # information for the model (mainly parameters), these are the devices (EV, BESS, PV, etc.), the costs (interests, capex, etc) and the revenues
    )
    # Sets
    tend=sets.dTime[end]
    t0=sets.dTime[1]; # initial time, can´t divide by 0
    model = InfiniteModel(optimizer) # create model

    # define cont t
    @infinite_parameter(model, t ∈ [t0, tend], supports = collect(sets.dTime))
    # Add devices
    # Electrical
    spvRFO!(model, data);
    bessRFO!(model, sets, data); # ok
    evRFO!(model, sets, data);
    # Thermal
    stRFO!(model, data);
    heatpumpRFO!(model, sets, data);
    tessRFO!(model, sets, data); 
    gridThermalRFO!(model, sets, data); # thermal balance
    # grids and balances
    gridConnRFO!(model, sets, data); # grid connection
    peiRFO!(model, sets, data); # electrical balance
    # Add objective function
    costFunctionRFO!(model, sets, data);
    # Solve model
    optimize!(model)
    return model;
end;

# Define the weights
# opt weights
Wgrid = 1; WSoCDep = 1000; Wtess = 1000; Wπ = 0.0;
W=[Wgrid WSoCDep Wtess Wπ];
# Initialize EMS
s=modelSettingsRFO(nEV=2, t0=1/4,Tw=48-1/4, Δt=1/4, steps=1, num_samples = 1, costWeights=W, 
                season="summer", profType="daily", loadType="GV", year=2023, cellID="SYNSANYO");
EMSData=build_data_DET(;nEV = s.nEV,
                season = s.season,
                profType = s.profType,
                loadType = s.loadType,
                year = s.year,
                cellID = s.cellID,
                stoch_num_samples = s.num_samples,
                );
#### Case study 1 - Picking optimal Qbess battery capacity
model = solvePolicies(Ipopt.Optimizer, s, EMSData);

## Save results
function optValues(model)
    dict=Dict(:Qbess => value(model[:Qbess]),
            # Retrieve the results,
            :Pg => value.(model[:Pg]),
            :Ppv => value.(model[:PpvMPPT]),
            :PloadE => value.(model[:Ple]),
            :Pbess => value.(model[:Pbess]),
            :Pev => value.(model[:Pev]),
            :PevTot => value.(model[:PevTot]),
            :SoCb => value.(model[:SoCbess]),
            :SoCev => value.(model[:SoCev]),
            :γf => value.(model[:γf]),
            :Pdrivef => value.(model[:Pdrivef]),
            :PloadTh => value.(model[:Plt]),
            :Pst => value.(model[:Pst]),
            :Phpe => value.(model[:Phpe]),
            :Ptess => value.(model[:Ptess]),
            :SoCtess => value.(model[:SoCtess]),
            :cost => objective_value(model),
            :t => value.(model[:t]),
            )
    return dict
end
res = optValues(model)

folder = "../data/output/"
open(folder * "cs1_$(s.season)_RFO_s$(s.num_samples)_ipopt.json", "w") do f
    JSON3.pretty(f, res)
end

# Now lets solve all samples individually to show the different results
# lets solve all the deterministic cases
model_det = Vector{InfiniteModel}(undef, s.num_samples)
s_det = modelSettingsRFO(nEV=2, t0=1/4,Tw=48-1/4, Δt=1/4, steps=50, num_samples = 1, costWeights=W, season="winter",
                profType="daily", loadType="GV", year=2023, cellID="SYNSANYO");
for i ∈ 1:s.num_samples
    EMSData_i = copy(EMSData)
    evModel = Vector{EVDataRFO}(undef, s.nEV)
    for n ∈ 1:s.nEV
        batteryPack = EMSData["EV"][n].carBatteryPack
        drive_info = driveDataRFO([EMSData["EV"][n].driveInfo.Pdrive[i]],
                                 EMSData["EV"][n].driveInfo.SoCdep,
                                 [EMSData["EV"][n].driveInfo.γ[i]],
                                 [EMSData["EV"][n].driveInfo.tDep[i]],
                                 [EMSData["EV"][n].driveInfo.tArr[i]])
        evModel[n] = EVDataRFO(batteryPack, drive_info)
    end
    EMSData_i["EV"] = evModel;
    model_det[i] = solvePolicies(Ipopt.Optimizer, s_det, EMSData_i, Dict())
end
results_det = [getResultsRFO(model) for model ∈ model_det]
results_det = [optValues(model) for model ∈ model_det]
folder = "../data/output/RFO/"
open(folder * "cs1_$(s.season)_DET_s$(s.num_samples)_ipopt.json", "w") do f
    JSON3.pretty(f, results_det)
end