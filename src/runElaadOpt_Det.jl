cd(@__DIR__)

# Import all the necessary packages
using EMSmodule,
    Random,
    InfiniteOpt,
    JuMP,
    MathOptInterface,
    KNITRO,
    JSON3,
    MAT,
    StatsBase,
    CSV,
    DataFrames,
    Revise,
    Distributions,
    GaussianMixtures,
    Interpolations,
    GLMakie,
    CairoMakie,
    Makie
includet("../fns/EMSrobustfns.jl")
Random.seed!(1234);
## Define all the necessary functions
function solvePolicies(optimizer, # model optimizer
    sets::modelSettingsRFO, # number of EVs, discrete time supports, etc.
    data::Dict, # information for the model (mainly parameters), these are the devices (EV, BESS, PV, etc.), the costs (interests, capex, etc) and the revenues
    preRes::Dict, # previous results to warm start the model
    )
    # Sets
    tend=sets.dTime[end]
    t0=sets.dTime[1]; # initial time, can´t divide by 0
    model = InfiniteModel(optimizer) # create model

    # Optimizer attributes
    # KNITRO attributes
    set_optimizer_attributes(model, 
                            # "tuner"=>1,
                            "scale"=>1,
                            "outlev"=>1,
                            "opttol"=>1e-3,
                            "feastol"=>1e-4,
                            "mip_opt_gap_rel"=>1e-3,
                            "mip_multistart"=>1,
                            "mip_method" => 1,
                            # # "tuner_file"=>tunerpath,
                            # "mip_maxnodes" => 6000,
                            # "maxtime_real"=>1000,
                            # # options
                            # "hessopt" => 1,
                            # "hessian_no_f" => 1,
                            # "mip_branchrule" => 2,
                            # "mip_heuristic_strategy" => 2,
                            # "mip_heuristic_feaspump" => 1,
                            # "mip_heuristic_mpec" => 1,
                            # "mip_heuristic_strategy" => 1,
                            # "mip_knapsack" => 1,
                            # "mip_lpalg" => 3,
                            # "mip_mir" => 2,
                            # "mip_pseudoinit" => 1,
                            # "mip_rootalg" => 1,
                            # "mip_rounding" => 3,
                            # "mip_selectrule" => 2,
                            # "mip_zerohalf" => 0,
                            # "mip_liftproject" => 0,
                            # "mip_heuristic_lns" => 0,
                            # "mip_heuristic_misqp" => 0,
                            # "mip_heuristic_diving" => 0,
                            # "mip_clique" => 0,
                            # "mip_gomory" => 0,
                            )

    # define cont t
    @infinite_parameter(model, t ∈ [t0, tend], supports = collect(sets.dTime),
                        )
                        # derivative_method = FiniteDifference(Forward(),true))
    # Add devices
    # Electrical
    spvRFO!(model, data);
    bessRFO!(model, sets, data); # ok
    evRFO!(model, sets, data);
    # # Thermal
    stRFO!(model, data);
    heatpumpRFO!(model, sets, data);
    tessRFO!(model, sets, data); 
    gridThermalRFO!(model, sets, data); # thermal balance
    # grids and balances
    gridConnRFO!(model, sets, data); # grid connection
    peiRFO!(model, sets, data); # electrical balance
    # Add objective function
    costFunctionRFO!(model, sets, data)
    # Warm starting.
    # check if the previous solution is not empty
    !isempty(preRes) ? set_warm_start(model, preRes) : nothing;
    # Solve model
    optimize!(model)
    # results = getResults(model)
    # MOI.empty!(InfiniteOpt.backend(model))
    return model;
end;

# Define the weights
# opt weights
Wgrid = 1; WSoCDep = 1000; Wtess = 1000; Wπ = 0.0;
W=[Wgrid WSoCDep Wtess Wπ];
# Initialize EMS
s=modelSettingsRFO(nEV=2, t0=1/4,Tw=48-1/4, Δt=1/4, steps=50, num_samples = 1, costWeights=W, 
                season="winter", profType="daily", loadType="GV", year=2023, cellID="SYNSANYO");
EMSData=build_data(;nEV = s.nEV,
                season = s.season,
                profType = s.profType,
                loadType = s.loadType,
                year = s.year,
                cellID = s.cellID,
                stoch_num_samples = s.num_samples,
                );
makeInputsplot(EMSData["grid"], EMSData["SPV"])
model = solvePolicies(KNITRO.Optimizer, s, EMSData, Dict());
latex_formulation(model)
[EMSData["EV"][n].carBatteryPack.GenInfo.PowerLim = [-25., 25.] for n ∈ 1:s.nEV];
EMSData["grid"].PowerLim = [-50., 50.];

## Save results
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

folder = "../data/output/det/"
open(folder * "cs1_R2_$(s.season)_DET_s$(s.num_samples).json", "w") do f
    JSON3.pretty(f, results)
end

latex_formulation(model)
# Extract the results
opt_Qbess = value(model[:Qbess])
# Retrieve the results[:]
opt_Pg = value.(model[:Pg])[1]
opt_Ppv = value.(model[:PpvMPPT])
opt_PloadE = value.(model[:Ple])
opt_Pbess = value.(model[:Pbess])[1]
opt_Pev = value.(model[:Pev])[1]
opt_PevTot = value.(model[:PevTot])[1]
opt_SoCb = value.(model[:SoCbess])[1]
opt_SoCev = value.(model[:SoCev])[1]
γf = value.(model[:γf])[1]
Pdrivef = value.(model[:Pdrivef])[1]

opt_PloadTh = value.(model[:Plt])
opt_Pst = value.(model[:Pst])
opt_Phpe = value.(model[:Phpe])[1]
opt_Ptess = value.(model[:Ptess])[1]
opt_SoCtess = value.(model[:SoCtess])[1]

opt_cost = objective_value(model)
t = value.(model[:t])
results = getResultsRFO(model)

function plot_Pdet()
    GLMakie.activate!()
    set_theme!(theme_latexfonts(), fontsize = 18)
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], xlabel = "Time [h]", ylabel = "Power [kW]")
    colors = Makie.wong_colors()
    stairs!(ax, t, opt_Pg, label = "Grid")
    stairs!(ax, t, opt_Ppv, label = "PV")
    stairs!(ax, t, opt_PloadE, label = "Load")
    stairs!(ax, t, opt_Phpe, label = "HP")
    stairs!(ax, t, opt_Pbess, label = "BESS")
    stairs!(ax, t, opt_Pev, label = "EV")
    axislegend(ax, position = :lb)
    return fig
end
function plot_PThdet()
    GLMakie.activate!()
    set_theme!(theme_latexfonts(), fontsize = 18)
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], xlabel = "Time [h]", ylabel = "Power [kW]")
    colors = Makie.wong_colors()
    stairs!(ax, t, opt_Ppv .* EMSData["ST"].η, label = "ST")
    stairs!(ax, t, opt_Pst, label = "ST")
    stairs!(ax, t, opt_PloadTh, label = "Load")
    stairs!(ax, t, opt_Phpe .* EMSData["HP"].η, label = "HP")
    stairs!(ax, t, opt_Ptess, label = "TESS")
    axislegend(ax, position = :lb)
    return fig
end
function plot_SoCdet()
    GLMakie.activate!()
    set_theme!(theme_latexfonts(), fontsize = 18)
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], xlabel = "Time [h]", ylabel = "Power [kW]")
    colors = Makie.wong_colors()
    stairs!(ax, t, opt_SoCb, label = "BESS")
    stairs!(ax, t, opt_SoCev, label = "EV")
    stairs!(ax, t, opt_SoCtess, label = "TESS")
    axislegend(ax, position = :lb)
    return fig
end
plot_SoCdet()
plot_PThdet()
plot_Pdet()

γ59 = [EMSData["EV"][n].driveInfo.γ[59] for n ∈ 1:s.nEV]
tArr59 = [EMSData["EV"][n].driveInfo.tArr[59] for n ∈ 1:s.nEV]

sens_res = Dict("Qtess"=>[], "Qbess"=>[], "cost"=>[], "results"=>[])
EMSData["TESS"].Q = 100;
for SoC ∈ 0.2:0.2:1.
    EMSData["TESS"].SoC0 = SoC;
    # Run the rolling horizon simulation
    model = solvePolicies(KNITRO.Optimizer, s, EMSData, Dict())
    results = getResultsRFO(model)
    println("The chosen BESS capacity is ", value(model[:Qbess]), "kWh")
    println("With and objective value of ", objective_value(model))
    # store the results
    # add the results to the DataFrame
    sens_res = vcat(sens_res, Dict("SoCtess"=>SoCtess,
                                    "Qbess"=>value(model[:Qbess]),
                                    "cost"=>objective_value(model)),
                                    "results"=>getResultsRFO(model))
end
