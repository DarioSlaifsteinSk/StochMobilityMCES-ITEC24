cd(@__DIR__)

# Import all the necessary packages
# using JuMP, InfiniteOpt, KNITRO
# using LinearAlgebra, Distributions, Statistics, Parameters, Random, Revise, Test
# using DataFrames, LaTeXStrings, Printf, JSON3, Makie, CairoMakie, GLMakie
using EMSmodule, Random, InfiniteOpt, KNITRO, JSON3, MAT, StatsBase, CSV, DataFrames
#  Statistics, Printf, Dates, LinearAlgebra, Distributions, Parameters, Revise, Test
includet("../fns/EMSrobustfns.jl")
Random.seed!(1234);

## Define all the necessary functions
function solvePolicies(optimizer, # model optimizer
    sets::modelSettings, # number of EVs, discrete time supports, etc.
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
                            "outlev"=>0,
                            # "tuner_file"=>tunerpath,
                            "mip_maxnodes" => 6000,
                            "maxtime_real"=>1000,
                            "opttol"=>1e-4,
                            "feastol"=>1e-4,
                            # options
                            "hessopt" => 1,
                            "hessian_no_f" => 1,
                            "mip_branchrule" => 2,
                            "mip_heuristic_strategy" => 2,
                            "mip_heuristic_feaspump" => 1,
                            "mip_heuristic_mpec" => 1,
                            "mip_heuristic_strategy" => 1,
                            "mip_knapsack" => 1,
                            "mip_lpalg" => 3,
                            "mip_method" => 1,
                            "mip_mir" => 2,
                            "mip_pseudoinit" => 1,
                            "mip_rootalg" => 1,
                            "mip_rounding" => 3,
                            "mip_selectrule" => 2,
                            "mip_zerohalf" => 0,
                            "mip_liftproject" => 0,
                            "mip_heuristic_lns" => 0,
                            "mip_heuristic_misqp" => 0,
                            "mip_heuristic_diving" => 0,
                            "mip_clique" => 0,
                            "mip_gomory" => 0,
                            )

    # define cont t
    @infinite_parameter(model, t ∈ [t0, tend], supports = collect(sets.dTime),
                        derivative_method = FiniteDifference(Backward()))

    @infinite_parameter(model, tarr ~ Uniform(tarr_min, tarr_max), num_supports = sets.num_samples)
    # @infinite_parameter(model, tdep ~ Uniform(tdep_min, tdep_max), num_supports = sets.num_samples)
    # Add devices
    # Electrical
    spvRob!(model, data); # ok
    bess!(model, sets, data); # ok
    ev!(model, sets, data);
    # Thermal
    stRob!(model, data);
    heatpump!(model, data);
    tess!(model, data);
    gridThermalRob!(model, sets, data); # thermal balance
    # grids and balances
    gridConn!(model, data); # grid connection
    peiRob!(model, sets, data); # electrical balance
    # Add objective function
    costFunctionRob!(model, sets, data)
    # Warm starting.
    # check if the previous solution is not empty
    !isempty(preRes) ? set_warm_start(model, preRes) : nothing;
    # Solve model
    optimize!(model)
    results = getResults(model)
    MOI.empty!(InfiniteOpt.backend(model))
    return results;
end;

function handleInfeasible!(results, ctrlRes, ts, s; typeOpt::String="MPC")
    # this function is used to handle infeasible solutions
    # it is called in the solvePolicies() function
    # If an MPC solution is infeasible implement the previous solution 
    @assert typeOpt ∈ ["MPC", "day-ahead"];
    typeOpt == "MPC" ? shift = 1 : nothing;
    # take the length from results because its the one handled by handleInfeasible()
    typeOpt == "day-ahead" ? shift = Int(ceil(length(results[ts]["t"])/2))-1 : nothing;

    println("\r Checking the status of the last set of results")
    if results[ts][:"status"] == LOCALLY_INFEASIBLE
    # discard the last set of results and implement the second step of the previous solution at ts-1
        println("\r Infeasible solution at step $ts, implementing the second step of the previous solution at step $(ts-1)")
        results[ts]=copy(ctrlRes[ts-1]);
        if typeOpt == "MPC"
            # the first setpoints from results[ts-1] were already implemented in the previous step, 
            # so we need to shift the setpoints one step, notice that the end of the time window won't move! 
            for k in keys(results[ts])
                # Check if the key requires special handling and skip the "status" key
                if k == :"status"
                    continue
                elseif k == :"γ_cont"
                    if s.nEV !== 1
                        results[ts][k] = [results[ts][k][n][2:end] for n ∈ 1:s.nEV];
                    else
                        results[ts][k] = results[ts][k][2:end];
                    end
                else
                    results[ts][k] = results[ts][k][2:end];
                end
            end
        else # day-ahead
            for k in keys(results[ts])
                # Check if the key requires special handling and skip the "status" key
                if k == :"status"
                    continue
                elseif k == :"γ_cont"
                    if s.nEV !== 1
                        results[ts][k] = [results[ts][k][n][(shift+1):end] for n ∈ 1:s.nEV];
                    else
                        results[ts][k] = results[ts][k][(shift+1):end];
                    end
                else
                    results[ts][k] = results[ts][k][(shift+1):end];
                end
            end
        end
        # update the controller setpoints so that for next iteration this stored somewhere
        ctrlRes[ts]=copy(results[ts]);
    else
        println("\r Locally feasible solution at step $ts")
    end
    return results, ctrlRes;
end

function rollingHorizon(s;
    typeOpt::String="MPC") # MPC or day-ahead
    ## Rolling Horizon Simulation.
    # This function simulates the EMS for a given number of steps. 
    # The EMS is initialized at time t0 and then it is solved for a time window of Tw hours.
    # Then, the EMS is solved again for the next Tw hours, but this time the initial conditions are the ones obtained from the previous solution.
    # This process is repeated until the number of steps is reached.
    # The output of this function are:
    # - results::Vector{Dict}, where each dictionary contains the results of the EMS for each time window.
    # - rhDict::Dict, which is a dictionary with the concatenated results of the EMS for each time window.
    @assert typeOpt ∈ ["MPC", "day-ahead"] "typeOpt must be either MPC or DA"
    if typeOpt == "day-ahead"
        @assert s.Tw == 48-1/4 "tend must be 48h for DA"
    end

    steps = s.steps; Tw = s.Tw; Δt = s.Δt;
    Dt = s.dTime[1]:(s.dTime[2]-s.dTime[1]):s.dTime[end];
    # allocate memory
    results=Vector{Dict}(undef, steps);
    controllerRes=Vector{Dict}(undef, steps);
    # Initialize an array to store the times
    times = Vector{Float64}(undef, steps)

    # Build the data dictionary
    EMSData=build_data(;nEV = s.nEV,
                        # season = s.season,
                        profType = s.profType,
                        loadType = s.loadType,
                        year = s.year,
                        cellID = s.cellID,
                        );
    # Modify the TESS init condition
    EMSData[:"TESS"].SoC0=0.4;
    # # Modify the BESS perf. model
    EMSData[:"BESS"].GenInfo.SoC0=0.5;
    # EMSData[:"BESS"].PerfParameters = bucketPerfParams();
    # # Modify the EV model
    # [EMSData[:"EV"][n].carBatteryPack.PerfParameters = bucketPerfParams() for n ∈ 1:s.nEV];
    # Modify the BESS ageing model
    EMSData[:"BESS"].AgingParameters = empAgingParams();
    # Modify the EV model
    [EMSData[:"EV"][n].carBatteryPack.AgingParameters = empAgingParams() for n ∈ 1:s.nEV];

    for ts in 1:steps
        times[ts] = @elapsed begin
            # build+solve model
            results[ts]=solvePolicies(KNITRO.Optimizer, s, EMSData);
            controllerRes[ts] = copy(results[ts]);
            # Check the status of the last set of results
            handleInfeasible!(results, controllerRes, ts, s; typeOpt=typeOpt);
            # EMSData=update_measurements(results[ts], s, EMSData; typeOpt=typeOpt);
            simTransitionFun!(results[ts], EMSData, s; typeOpt = typeOpt)
            # move time window
            typeOpt == "MPC" ? Dt = Dt .+ Δt*3600.0 : Dt = Dt.+ (Tw+Δt)/2*3600.0;
            # update_forecasts(EMSData, Dt)
            s.dTime=collect(Dt);
        end
        Base.GC.gc(); # collect garbage
        println("\r Step $ts out of $steps done!")
    end
    [results[ts]["compTime"]=times[ts] for ts in 1:steps]
    return results, EMSData,s, controllerRes;
end

function simMonteCarlo(s;
    typeOpt::String="MPC") # MPC or day-ahead
    ## Rolling Horizon Simulation.
    # This function simulates the EMS for a given number of steps. 
    # The EMS is initialized at time t0 and then it is solved for a time window of Tw hours.
    # Then, the EMS is solved again for the next Tw hours, but this time the initial conditions are the ones obtained from the previous solution.
    # This process is repeated until the number of steps is reached.
    # The output of this function are:
    # - results::Vector{Dict}, where each dictionary contains the results of the EMS for each time window.
    # - rhDict::Dict, which is a dictionary with the concatenated results of the EMS for each time window.
    @assert typeOpt ∈ ["MPC", "day-ahead"] "typeOpt must be either MPC or DA"
    if typeOpt == "day-ahead"
        @assert s.Tw == 48-1/4 "tend must be 48h for DA"
    end

    steps = s.steps; Tw = s.Tw; Δt = s.Δt;
    Dt = s.dTime[1]:(s.dTime[2]-s.dTime[1]):s.dTime[end];
    # allocate memory
    results=Vector{Dict}(undef, steps);
    controllerRes=Vector{Dict}(undef, steps);
    # Initialize an array to store the times
    times = Vector{Float64}(undef, steps)

    # Initial solution
    times[1] = @elapsed begin
        # Build the data dictionary
        EMSData=build_data(;nEV = s.nEV,
                # season = s.season,
                profType = s.profType,
                loadType = s.loadType,
                year = s.year,
                cellID = s.cellID,
                );
        results[1]=solvePolicies(KNITRO.Optimizer, s, EMSData, Dict());
        controllerRes[1] = copy(results[1]);
        # Check the status of the last set of results
        handleInfeasible!(results, controllerRes, 1, s; typeOpt=typeOpt);
        simTransitionFun!(results[1], EMSData, s; typeOpt = typeOpt)
    end

    for ts in 2:steps
        try
            times[ts] = @elapsed begin
                # rebuild the input data
                EMSData=build_data(;nEV = s.nEV,
                                    # season = s.season,
                                    profType = s.profType,
                                    loadType = s.loadType,
                                    year = s.year,
                                    cellID = s.cellID,
                                    );
                # build+solve model
                results[ts]=solvePolicies(KNITRO.Optimizer, s, EMSData, controllerRes[ts-1]);
                controllerRes[ts] = copy(results[ts]);
                # Check the status of the last set of results
                handleInfeasible!(results, controllerRes, ts, s; typeOpt=typeOpt);
                # EMSData=update_measurements(results[ts], s, EMSData; typeOpt=typeOpt);
                simTransitionFun!(results[ts], EMSData, s; typeOpt = typeOpt)
            end
            # Base.GC.gc(); # collect garbage
            println("\r Step $ts out of $steps done!")
        catch e
            println("Error at step $ts")
            showerror(stdout, e, catch_backtrace())
            return results, EMSData,s, controllerRes;
        end
    end
    [results[ts]["compTime"]=times[ts] for ts in 1:steps]
    return results, EMSData,s, controllerRes;
end

# Define the weights
# opt weights
Wgrid = 1; WSoCDep = 1000; Wtess = 1000; Wπ = 0.0;
W=[Wgrid WSoCDep Wtess Wπ];
# Initialize EMS
s=modelSettings(nEV=2, t0=1/4,Tw=48-1/4, Δt=1/4, steps=50, costWeights=W, season="summer",
                profType="daily", loadType="GV", year=2023, cellID="SYNSANYO");
EMSData=build_data(;nEV = s.nEV,
                # season = s.season,
                profType = s.profType,
                loadType = s.loadType,
                year = s.year,
                cellID = s.cellID,
                );
# Run the rolling horizon simulation
results,EMSData,s, controllerRes = simMonteCarlo(s; typeOpt="day-ahead");
# Plotting functions
folder = "../images/AppEnergy/cs1/"
plotname ="DACEmpDeg_W11e4_$(s.season)_EMSplots.pdf"
makeEMSplots(rhDict, EMSData; backend="GLMakie")
makeEMSplots(rhDict, EMSData; backend="CairoMakie", filename= folder * plotname)
makeEBplot(rhDict, EMSData)
makeEBplot(concatResultsRH(controllerRes[end-1:end]; typeOpt="day-ahead"), EMSData)
makeEBplot(results[1], EMSData)
compareEB(results, EMSData)

plotTestSoCeESS(rhDict, EMSData, ["BESS", "EV", "TESS"]);
optStatusPlots(results)
testPBalance(rhDict, EMSData)

## Save results using JSON3
folder = "../data/output/nEV2/"
open(folder * "cs1_R4_$(s.season)_MC_s$(s.steps)_nEV$(s.nEV).json", "w") do f
    JSON3.pretty(f, results)
end