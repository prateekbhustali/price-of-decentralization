# This is the main script to run SARSOP on a POMDP defined in k_out_of_n_infinite.jl
# It uses the SARSOP package to solve the POMDP and evaluate the policy.

import YAML

# Solvers
using SARSOP

# Load the make_pomdp function
include("k_out_of_n_infinite.jl")

############################## Create the POMDP ########################
if length(ARGS) == 1
    setting = ARGS[1]
else
    setting = "hard-1-of-4_infinite"
end

env_config_path = normpath(
    joinpath(
        @__DIR__,
        "..",
        "imprl",
        "imprl",
        "envs",
        "structural_envs",
        "env_configs",
        setting * ".yaml",
    ),
)
env_config = YAML.load_file(env_config_path)
pomdp = make_POMDP(env_config)

############################### Logging ################################
using Random
using Dates
using Sockets

_location = joinpath(@__DIR__, "experiments") # location to save the experiment results
_timestamp = Dates.now()
_random_string = randstring(8)
_experiment_name = string(_timestamp) * "-" * _random_string
_experiment_dir = joinpath(_location, _experiment_name)

println("Experiment name: ", _experiment_name)

# create a directory for the experiment
mkpath(_experiment_dir)

# save the environment configuration
YAML.write_file(joinpath(_experiment_dir, "env_config.yaml"), env_config)

# write std out to a file
stdout_file = joinpath(_experiment_dir, "stdout.log")
println("All subsequent output is being redirected to: ", stdout_file)

# write std out to a file as well
open(stdout_file, "w") do io
    redirect_stdout(io)
end

################################ Solver ################################

solver = SARSOPSolver(
    fast=true, # set to true to use fast SARSOP
    randomization=false, # randomization for the sampling algorithm
    timeout=180, #  [sec] If running time exceeds the specified value, pomdpsol writes out a policy and terminates
    memory=nothing, # [MB] If memory usage exceeds the specified value, pomdpsol writes out a policy and terminates
    precision=0.01, # convergence criteria
    trial_improvement_factor=0.01, # terminates when the gap between bounds reaches this value
    policy_interval=nothing, # the time interval between two consecutive write-out of policy files
    pomdp_filename=joinpath(_experiment_dir, "pomdp.pomdp"),
    policy_filename=joinpath(_experiment_dir, "policy.policy"),
)

################################ Solve #################################

policy = solve(solver, pomdp)

############################### Evaluation #############################

sim = SARSOPSimulator(
    fast=true,
    sim_len=20,
    sim_num=100,
    pomdp_filename=joinpath(_experiment_dir, "pomdp.pomdp"),
    policy_filename=joinpath(_experiment_dir, "policy.policy"),
)

POMDPs.simulate(sim)
