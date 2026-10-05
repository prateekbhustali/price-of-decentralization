#= 
Validate the heuristic policies on the k-out-of-n problem
=#
using Distributed
addprocs(8)

@everywhere begin
    
    # Load the make_pomdp function
    include("k_out_of_n_infinite.jl")

    import YAML

    ############################## Create the POMDP ########################
    if length(ARGS) == 1
        setting = ARGS[1]
    else
        setting = "hard-4-of-4_infinite"
    end

    env_config = YAML.load_file("./env_configs/" * setting * ".yaml")
    pomdp = make_POMDP(env_config)

    ############################### Evaluation #############################
    struct DoNothing <: Policy end
    POMDPs.action(::DoNothing, ::Any) = [1 for i in 1:4]
    POMDPs.updater(::DoNothing) = NothingUpdater()

    struct MyPolicy <: Policy end
    POMDPs.action(::MyPolicy, ::Any) = [1, 2, 3, 1]
    POMDPs.updater(::MyPolicy) = NothingUpdater()

end

# policy = DoNothing()
policy = MyPolicy()

q = []
for i in 1:10000
    push!(q, Sim(pomdp, policy, max_steps=20))
end

data = run_parallel(q)

using DataFrames
using Statistics

# Compute the average reward and standard deviation
avg_reward = mean(data.reward)
std_reward = std(data.reward)

@show avg_reward
@show std_reward