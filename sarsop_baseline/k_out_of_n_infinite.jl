#= 
Inspection and Maintenance Planning for Multi-Component Systems 

Modelling the problem as a POMDP:
    - Infinite horizon
    - 3 damage states
    - Initial damage state distribution
    - Imperfect failure observations
    - Mobilisation costs for actions

=#

using IterTools
using LinearAlgebra
using Printf
using Statistics

# POMDP models
using POMDPs, QuickPOMDPs, POMDPModels, POMDPTools, POMDPSimulators, POMDPFiles


function make_POMDP(env_config)
    ######################### POMDP specification ##########################

    # extract parameters from the configuration
    num_components = env_config["n_components"]
    num_damage_states = env_config["n_damage_states"]
    num_component_actions = env_config["n_comp_actions"]
    num_damage_obs = num_damage_states

    # state space
    state_space = reduce(vcat, [(ds) for ds in product(fill(1:num_damage_states, num_components)...)])
    # push!(state_space, tuple(zeros(Int, num_components)...))

    # action space
    action_space = [a for a in product(fill(1:num_component_actions, num_components)...)]

    # observation space
    observation_space = reduce(vcat, [(ds) for ds in product(fill(1:num_damage_states, num_components)...)])
    # push!(observation_space, (tuple(zeros(Int, num_components)...)))

    num_states = length(state_space)
    num_actions = length(action_space)
    num_obs = length(observation_space)

    # initial state
    initial_probs = env_config["initial_belief"]
    _x = zeros(num_states)

    for i in 1:num_states
        _p = 1.0
        ds = state_space[i]

        for j in 1:num_components
            _p *= initial_probs[ds[j]]
        end
        _x[i] = _p
    end
    initial_state = SparseCat(state_space, _x)

    # open(stdout_file, "w") do io
    #     redirect_stdout(io)  # Add this line to redirect the output to the file

    #     println("Number of states: ", num_states, " (", num_damage_states, "^", num_components, ")")
    #     println("Number of actions: ", num_actions, " (", num_component_actions, "^", num_components, ")")
    #     println("Number of observations: ", num_obs, " (", num_damage_states, "^", num_components, ")")
    # end

    ####################### terminal state function ########################
    function is_terminal(s)
        return false
    end

    ######################## transition function ########################
    replacement_table = zeros(num_components, num_damage_states, num_damage_states)
    for c in 1:num_components
        r = env_config["replacement_accuracies"][c]
        replacement_table[c, :, :] = [1 0 0;
            r 1-r 0;
            r 0 1-r]
    end

    transition_table = zeros(num_components, num_component_actions, num_damage_states, num_damage_states)

    for c in 1:num_components

        deterioration = transpose(cat(env_config["transition_model"][c]..., dims=2))

        # do nothing: deterioration
        transition_table[c, 1, :, :] = deterioration

        # replacement: replace instantly + deterioration
        transition_table[c, 2, :, :] = replacement_table[c, :, :] * deterioration

        # inspect: deterioration
        transition_table[c, 3, :, :] = deterioration
    end

    transition_model = function (s, a)

        "
        Input
        -----
        s: state (ds)
            ds: tuple of integers
                damage state of each component

        a: action (a1, a2, ..., an)
            ai: integer
                action for component i

        Output
        ------
        SparseCat
            next state distribution
        "

        next_probs = zeros(num_states)

        ds = s

        for idx_prime in 1:num_states  # next states

            ds_prime = state_space[idx_prime]

            _prod = 1.0
            for c in 1:num_components  # component
                _t = transition_table[c, a[c], :, :]
                _prod *= _t[ds[c], ds_prime[c]]
            end
            next_probs[idx_prime] = _prod
        end

        return SparseCat(state_space, next_probs)
    end

    ############################ reward function ###########################
    rewards_table = zeros(num_components, num_damage_states, num_component_actions)

    rewards_table[:, :, 2] = transpose(repeat(cat(env_config["replacement_rewards"]..., dims=2), outer=[num_damage_states, 1]))

    rewards_table[:, :, 3] = transpose(repeat(cat(env_config["inspection_rewards"]..., dims=2), outer=[num_damage_states, 1]))

    system_replacement_reward = sum(env_config["replacement_rewards"])

    function reward_model(s, a)

        "
        Input
        -----
        s: state (ds)
            ds: tuple of integers
                damage state of each component

        a: action (a1, a2, ..., an)
            ai: integer
                action for component i

        Output
        ------
        reward: float
            reward for taking action a in state s

        "

        ds = s
        reward = 0.0

        # action costs
        for c in 1:num_components
            reward += rewards_table[c, ds[c], a[c]]
        end

        # mobilisation costs
        mobilised = sum(a) > num_components
        reward += mobilised * env_config["mobilisation_reward"]

        # check if system is functional
        _temp = ds .÷ num_damage_states
        n_working = num_components - sum(_temp)
        functional = n_working >= env_config["k"]

        penalty = 0.0
        if !functional
            penalty += system_replacement_reward * env_config["failure_penalty_factor"]
        end

        reward += penalty

        return reward
    end

    ########################## observation function ########################
    inspection_model = zeros(num_components, num_damage_states, num_damage_states)

    for c in 1:num_components
        p = env_config["obs_accuracies"][c]
        f_p = env_config["failure_obs_accuracies"][c]
        inspection_model[c, :, :] = [
            p 1-p 0.0;
            (1-p)/2 p (1-p)/2;
            0.0 1-f_p f_p]
    end

    failure_obs_model = [
        1/3 1/3 1/3;
        1/3 1/3 1/3;
        1/3 1/3 1/3]

    observation_table = zeros(num_components, num_component_actions, num_damage_states, num_damage_obs)

    for c in 1:num_components
        observation_table[c, 1, :, :] = failure_obs_model
        observation_table[c, 2, :, :] = failure_obs_model
        observation_table[c, 3, :, :] = inspection_model[c, :, :]
    end

    observation_model = function (a, sp)

        "
        Input
        -----
        a: action (a1, a2, ..., an)
            ai: integer
                action for component i

        sp: state (ds)
            ds: tuple of integers
                damage state of each component

        Output
        ------
        SparseCat
            observation distribution
        "

        _probs = zeros(num_obs)


        for idx_obs in 1:num_obs

            ds_prime = sp
            obs = observation_space[idx_obs] # observation of components

            _prob = 1.0

            for c in 1:num_components  # component
                _prob *= observation_table[c, a[c], ds_prime[c], obs[c]]
            end
            _probs[idx_obs] = _prob
        end

        return SparseCat(observation_space, _probs)
    end

    ########################### Setup the POMDP ############################
    pomdp = QuickPOMDP(
        states=state_space,
        actions=action_space,
        observations=observation_space,
        transition=transition_model,
        reward=reward_model,
        observation=observation_model,
        discount=env_config["discount_factor"],
        initialstate=initial_state,
        terminalstates=is_terminal,
    )

    return pomdp

end

###### Write to .pomdp file ######
# POMDPFiles.write(_location * _experiment_name * "/pomdp.pomdp", pomdp)