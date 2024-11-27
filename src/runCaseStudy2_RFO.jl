cd(@__DIR__)

# Import all the necessary packages
using Random,
    InfiniteOpt,
    JuMP,
    MathOptInterface,
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
s=modelSettingsRFO(nEV=2, t0=1/4,Tw=48-1/4, Δt=1/4, steps=50, num_samples = 500, costWeights=W, 
                season="summer", profType="daily", loadType="GV", year=2023, cellID="SYNSANYO");
EMSData=build_data_RFO(;nEV = s.nEV,
                season = s.season,
                profType = s.profType,
                loadType = s.loadType,
                year = s.year,
                cellID = s.cellID,
                stoch_num_samples = s.num_samples,
                );
makeInputsplot(EMSData["grid"], EMSData["SPV"])

### Case study 2
# Define the sensitivity analysis
sens_res = DataFrame("CAPEX"=>[], "Qbess"=>[], "cost"=>[])
for CAPEX ∈ 1e-2:1e-2:.3
    EMSData["BESS"].GenInfo.initValue = CAPEX;
    # Run the rolling horizon simulation
    model = solvePolicies(Ipopt.Optimizer, s, EMSData);
    println("The chosen battery capacity is ", value(model[:Qbess]), "kWh")
    println("With and objective value of ", objective_value(model))
    # store the results
    # add the results to the DataFrame
    sens_res = vcat(sens_res, DataFrame("CAPEX"=>CAPEX,
                                        "Qbess"=>value(model[:Qbess]),
                                        "cost"=>objective_value(model)))
end

# Save the results in a CSV file
folder = "../data/output/"
open(folder * "sensitivity_analysis_$(s.season)_RFO_s$(s.num_samples).csv", "w") do f
    CSV.write(f, sens_res)
end