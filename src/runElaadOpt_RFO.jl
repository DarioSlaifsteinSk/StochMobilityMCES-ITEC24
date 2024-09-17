cd(@__DIR__)

# Import all the necessary packages
using EMSmodule,
    Random,
    InfiniteOpt,
    JuMP,
    MathOptInterface,
    KNITRO,
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
    # set_optimizer_attributes(model, 
    #                         # "tuner"=>1,
    #                         "scale"=>1,
    #                         "outlev"=>1,
    #                         "opttol"=>1e-3,
    #                         "feastol"=>1e-4,
    #                         "mip_opt_gap_rel"=>1e-3,
    #                         "mip_multistart"=>1,
    #                         "mip_method" => 1,
    #                         # # "tuner_file"=>tunerpath,
    #                         # "mip_maxnodes" => 6000,
    #                         # "maxtime_real"=>1000,
    #                         # # options
    #                         # "hessopt" => 1,
    #                         # "hessian_no_f" => 1,
    #                         # "mip_branchrule" => 2,
    #                         # "mip_heuristic_strategy" => 2,
    #                         # "mip_heuristic_feaspump" => 1,
    #                         # "mip_heuristic_mpec" => 1,
    #                         # "mip_heuristic_strategy" => 1,
    #                         # "mip_knapsack" => 1,
    #                         # "mip_lpalg" => 3,
    #                         # "mip_mir" => 2,
    #                         # "mip_pseudoinit" => 1,
    #                         # "mip_rootalg" => 1,
    #                         # "mip_rounding" => 3,
    #                         # "mip_selectrule" => 2,
    #                         # "mip_zerohalf" => 0,
    #                         # "mip_liftproject" => 0,
    #                         # "mip_heuristic_lns" => 0,
    #                         # "mip_heuristic_misqp" => 0,
    #                         # "mip_heuristic_diving" => 0,
    #                         # "mip_clique" => 0,
    #                         # "mip_gomory" => 0,
    # )

    # define cont t
    @infinite_parameter(model, t ∈ [t0, tend], supports = collect(sets.dTime),
                        )
                        # derivative_method = FiniteDifference(Forward(),true))
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

function solvePolicies_b(optimizer, # model optimizer
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
    # set_optimizer_attributes(model, 
    #                         # "tuner"=>1,
    #                         "scale"=>1,
    #                         "outlev"=>1,
    #                         "opttol"=>1e-3,
    #                         "feastol"=>1e-4,
    #                         "mip_opt_gap_rel"=>1e-3,
    #                         "mip_multistart"=>1,
    #                         "mip_method" => 1,
    #                         # # "tuner_file"=>tunerpath,
    #                         # "mip_maxnodes" => 6000,
    #                         # "maxtime_real"=>1000,
    #                         # # options
    #                         # "hessopt" => 1,
    #                         # "hessian_no_f" => 1,
    #                         # "mip_branchrule" => 2,
    #                         # "mip_heuristic_strategy" => 2,
    #                         # "mip_heuristic_feaspump" => 1,
    #                         # "mip_heuristic_mpec" => 1,
    #                         # "mip_heuristic_strategy" => 1,
    #                         # "mip_knapsack" => 1,
    #                         # "mip_lpalg" => 3,
    #                         # "mip_mir" => 2,
    #                         # "mip_pseudoinit" => 1,
    #                         # "mip_rootalg" => 1,
    #                         # "mip_rounding" => 3,
    #                         # "mip_selectrule" => 2,
    #                         # "mip_zerohalf" => 0,
    #                         # "mip_liftproject" => 0,
    #                         # "mip_heuristic_lns" => 0,
    #                         # "mip_heuristic_misqp" => 0,
    #                         # "mip_heuristic_diving" => 0,
    #                         # "mip_clique" => 0,
    #                         # "mip_gomory" => 0,
    #                     )

    # # define cont t
    @infinite_parameter(model, t ∈ [t0, tend], supports = collect(sets.dTime),
                        )
                        # derivative_method = FiniteDifference(Forward(),true))
    # Add devices
    # Electrical
    spvRFO!(model, data);
    bessRFO!(model, sets, data); # ok
    evRFO!(model, sets, data);
    # Thermal
    stRFO!(model, data);
    heatpump!(model, data);
    tess!(model, sets, data);
    gridThermal!(model, sets, data); # thermal balance
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
s=modelSettingsRFO(nEV=2, t0=1/4,Tw=48-1/4, Δt=1/4, steps=50, num_samples = 500, costWeights=W, 
                season="summer", profType="daily", loadType="GV", year=2023, cellID="SYNSANYO");
EMSData=build_data(;nEV = s.nEV,
                season = s.season,
                profType = s.profType,
                loadType = s.loadType,
                year = s.year,
                cellID = s.cellID,
                stoch_num_samples = s.num_samples,
                );
makeInputsplot(EMSData["grid"], EMSData["SPV"])

#### Case study 1 - Picking optimal Qbess battery capacity
model_ipopt = solvePolicies(Ipopt.Optimizer, s, EMSData, Dict());
model_knitro = solvePolicies(KNITRO.Optimizer, s, EMSData, Dict());
modelb = solvePolicies_b(Ipopt.Optimizer, s, EMSData, Dict());

model = solvePolicies(KNITRO.Optimizer, s, EMSData, Dict());
modelb = solvePolicies_b(KNITRO.Optimizer, s, EMSData, Dict());
latex_formulation(model)
[EMSData["EV"][n].carBatteryPack.GenInfo.PowerLim = [-25., 25.] for n ∈ 1:s.nEV];

## Save results
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
results = getResultsRFO(model)
res_ipopt = optValues(model_ipopt)
res_knitro = optValues(model_knitro)

folder = "../data/output/RFO/"
open(folder * "cs1_R7_$(s.season)_RFO_s$(s.num_samples)_ipopt.json", "w") do f
    JSON3.pretty(f, res_ipopt)
end

# Now lets solve all 200 samples individually to show the different results
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
open(folder * "cs1_R5_$(s.season)_DET_s$(s.num_samples)_ipopt.json", "w") do f
    JSON3.pretty(f, results_det)
end

### Case study 2
# Define the sensitivity analysis
sens_res = DataFrame("CAPEX"=>[], "Qbess"=>[], "cost"=>[])
for CAPEX ∈ 1e-2:1e-2:1.
    EMSData["BESS"].GenInfo.initValue = CAPEX;
    # Run the rolling horizon simulation
    model = solvePolicies(Ipopt.Optimizer, s, EMSData, Dict())
    println("The chosen battery capacity is ", value(model[:Qbess]), "kWh")
    println("With and objective value of ", objective_value(model))
    # store the results
    # add the results to the DataFrame
    sens_res = vcat(sens_res, DataFrame("CAPEX"=>CAPEX,
                                        "Qbess"=>value(model[:Qbess]),
                                        "cost"=>objective_value(model)))
end

# Save the results in a CSV file
folder = "../data/output/RFO/"
open(folder * "sensitivity_analysis_R7_$(s.season)_RFO_s$(s.num_samples).csv", "w") do f
    CSV.write(f, sens_res)
end

### Case study 0.1 - Testing different Qtess
# Define the sensitivity analysis
sens_res = Dict("Qtess"=>[], "Qbess"=>[], "cost"=>[], "results"=>[])
for Qtess ∈ 25:25:200
    EMSData["TESS"].Q = Qtess;
    # Run the rolling horizon simulation
    model = solvePolicies(KNITRO.Optimizer, s, EMSData, Dict())
    results = getResultsRFO(model)
    println("The chosen BESS capacity is ", value(model[:Qbess]), "kWh")
    println("With and objective value of ", objective_value(model))
    # store the results
    # add the results to the DataFrame
    sens_res = vcat(sens_res, Dict("Qtess"=>Qtess,
                                    "Qbess"=>value(model[:Qbess]),
                                    "cost"=>objective_value(model)),
                                    "results"=>getResultsRFO(model))
end

# Save the results in a CSV file
folder = "../data/output/RFO/"
open(folder * "sensitivity_analysis_R3_$(s.season)_RFO_s$(s.num_samples).json", "w") do f
    JSON3.write(f, sens_res)
end

latex_formulation(model)

γ59 = [EMSData["EV"][n].driveInfo.γ[59] for n ∈ 1:s.nEV]
tArr59 = [EMSData["EV"][n].driveInfo.tArr[59] for n ∈ 1:s.nEV]
# plot the γf
CairoMakie.activate!(type = "svg")
fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1], xlabel = "Time [h]", ylabel = "γf")
ax2 = Axis(fig[1, 2], xlabel = "Time [h]")
ax = [ax1, ax2]
for n ∈ 1:s.nEV
    heatmap!(ax[n], t, 1:size(γf,1), hcat(γf[:,n]), colormap = :viridis)
end
fig

function plot_ElecAnalysis(dict)
    GLMakie.activate!()
    set_theme!(theme_latexfonts(), fontsize = 18)
    fig = Figure(size = (800, 600))
    ax1 = Axis(fig[1, 1], xlabel = "Time [h]", ylabel = "Power [kW]")
    ax2 = Axis(fig[2, 1], xlabel = "Time [h]", ylabel = "SoC [%]")
    colors = Makie.wong_colors()
    t = dict[:t]
    stairs!(ax1, t, dict[:Ppv], color = colors[1], label = L"P_{\textrm{PV}}")
    stairs!(ax1, t, dict[:PloadE], color = colors[2], label = L"P_{\textrm{load}}")
    [stairs!(ax1, t, dict[:Phpe][i], color = (colors[3], 0.1), label = L"P_{\textrm{HP}}") for i ∈ 1:s.num_samples]
    [stairs!(ax1, t, dict[:Pg][i], color = (colors[4], 0.1), label = L"P_{\textrm{g}}") for i ∈ 1:s.num_samples]
    [stairs!(ax1, t, dict[:Pbess][i], color = (colors[5], 0.1), label = L"P_{\textrm{BESS}}") for i ∈ 1:s.num_samples]
    [stairs!(ax1, t, dict[:Pev][i,n], color = (colors[n+5], 0.1), label = L"P_{\textrm{EV}}") for i ∈ 1:s.num_samples, n ∈ 1:s.nEV]
    [stairs!(ax2, t, dict[:SoCb][i], color = (colors[4], 0.1), label = "TESS") for i ∈ 1:s.num_samples]
    [stairs!(ax2, t, dict[:SoCev][i,n], color = (colors[n+5], 0.1), label = "EV") for i ∈ 1:s.num_samples, n ∈ 1:s.nEV]

    axislegend(ax1, position = :lb, merge = true)
    return display(GLMakie.Screen(), fig)
end

function plot_ThermAnalysis(dict)
    CairoMakie.activate!(type = "svg")
    set_theme!(theme_latexfonts(), fontsize = 18)
    fig = Figure(size = (800, 600))
    ax1 = Axis(fig[1, 1], xlabel = "Time [h]", ylabel = "Power [kW]")
    ax2 = Axis(fig[2, 1], xlabel = "Time [h]", ylabel = "SoC [%]")	
    colors = Makie.wong_colors()
    t = dict[:t]
    stairs!(ax1, t, dict[:Pst], color = colors[1], label = L"P_{\textrm{ST}}")
    stairs!(ax1, t, dict[:PloadTh], color = colors[2], label = L"P_{\textrm{load}}")
    [stairs!(ax1, t, dict[:Phpe][i] .* EMSData["HP"].η, color = (colors[3], 0.1), label = L"P_{\textrm{HP}}") for i ∈ 1:s.num_samples]
    [stairs!(ax1, t, dict[:Ptess][i], color = (colors[4], 0.1), label = L"P_{\textrm{TESS}}") for i ∈ 1:s.num_samples]
    [stairs!(ax2, t, dict[:SoCtess][i] .* 100, color = (colors[5], 0.1), label = "TESS") for i ∈ 1:s.num_samples]
    axislegend(ax1, position = :lb, merge = true)
    return save("../images/stochastic/plot_ThermAnalysis_r5.pdf", fig)
    # return display(GLMakie.Screen(), fig)
end

# ALL TOGETHER
function plot_Analysis(dict)
    GLMakie.activate!()
    set_theme!(theme_latexfonts(), fontsize = 18)
    fig = Figure(size = (400, 700))
    ax1 = Axis(fig[1, 1], xlabel = "Time [h]", ylabel = "Power [kW]")
    ax2 = Axis(fig[2, 1], xlabel = "Time [h]", ylabel = "Power [kWth]")
    ax3 = Axis(fig[3, 1], xlabel = "Time [h]", ylabel = "SoC [%]")	
    colors = Makie.wong_colors()
    t = dict[:t]
    num_samples = s.num_samples
    # num_samples = 10;
    # i=rand(1:1:s.num_samples)
    stairs!(ax1, t, dict[:Ppv], color = colors[1], label = L"P_{\textrm{PV}}")
    stairs!(ax1, t, dict[:PloadE], color = colors[2], label = L"P_{\textrm{load}}")
    [stairs!(ax1, t, dict[:Phpe][i], color = (colors[3], 0.1), label = L"P_{\textrm{HP}}") for i ∈ 1:num_samples]
    [stairs!(ax1, t, dict[:Pg][i], color = (colors[4], 0.3), label = L"P_{\textrm{g}}") for i ∈ 1:num_samples]
    [stairs!(ax1, t, dict[:Pbess][i], color = (colors[5], 0.1), label = L"P_{\textrm{BESS}}") for i ∈ 1:num_samples]
    [stairs!(ax1, t, dict[:Pev][i,n], color = (colors[n+5], 0.1), label = L"P_{\textrm{EV}}") for i ∈ 1:num_samples, n ∈ 1:s.nEV]

    # stairs!(ax1, t, dict[:Phpe][i], color = (colors[3], 0.6), label = L"P_{\textrm{HP}}")
    # stairs!(ax1, t, dict[:Pg][i], color = (colors[4], 0.6), label = L"P_{\textrm{g}}")
    # stairs!(ax1, t, dict[:Pbess][i], color = (colors[5], 0.6), label = L"P_{\textrm{BESS}}")
    # [stairs!(ax1, t, dict[:Pev][i,n], color = (colors[n+5], 0.6), label = L"P_{\textrm{EV}}") for n ∈ 1:s.nEV]

    stairs!(ax2, t, dict[:Pst], color = colors[1], label = L"P_{\textrm{ST}}")
    stairs!(ax2, t, dict[:PloadTh], color = colors[2], label = L"P_{\textrm{load}}")
    [stairs!(ax2, t, dict[:Phpe][i] .* EMSData["HP"].η, color = (colors[3], 0.1), label = L"P_{\textrm{HP}}") for i ∈ 1:num_samples]
    [stairs!(ax2, t, dict[:Ptess][i], color = (colors[4], 0.1), label = L"P_{\textrm{TESS}}") for i ∈ 1:num_samples]
    # stairs!(ax2, t, dict[:Phpe][i] .* EMSData["HP"].η, color = (colors[3], 0.6), label = L"P_{\textrm{HP}}")
    # stairs!(ax2, t, dict[:Ptess][i], color = (colors[4], 0.6), label = L"P_{\textrm{TESS}}")
    
    
    [stairs!(ax3, t, dict[:SoCev][i,n], color = (colors[n+5], 0.8), label = "EV") for i ∈ 1:num_samples, n ∈ 1:s.nEV]
    [stairs!(ax3, t, dict[:SoCb][i], color = (colors[4], 0.6), label = "BESS") for i ∈ 1:num_samples]
    [stairs!(ax3, t, dict[:SoCtess][i] .* 100, color = (colors[5], 0.8), label = "TESS") for i ∈ 1:num_samples]
    axislegend(ax1, position = :lb, merge = true)
    axislegend(ax2, position = :lb, merge = true)
    axislegend(ax3, position = :lb, merge = true)
    return display(GLMakie.Screen(), fig)
end
plot_ElecAnalysis(res_knitro)
plot_ElecAnalysis(res_ipopt)
plot_ThermAnalysis(res_knitro)
plot_ThermAnalysis(res_ipopt)
plot_Analysis(res_knitro)
plot_Analysis(res_ipopt)