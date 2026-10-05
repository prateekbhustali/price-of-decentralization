# SARSOP Baseline (Julia)

Julia scripts to solve and evaluate the structural `k_out_of_n_infinite` POMDP with SARSOP.

## Install Julia
Install Julia `1.10.x` (tested with `1.10.4`) using `juliaup`:

```bash
curl -fsSL https://install.julialang.org | sh
juliaup add 1.10.4
juliaup default 1.10.4
julia --version
```

## Setup
From repo root:

```bash
cd sarsop_baseline
julia
```

In Julia package mode (`]`):

```julia
activate ./point-based-solvers
instantiate
exit() # back to shell
```

## Run SARSOP
From `sarsop_baseline/`:

```bash
julia --project="./point-based-solvers" main.jl hard-1-of-4_infinite
```

The run prints the experiment name and log path, then redirects subsequent output to `experiments/<timestamp-random_id>/stdout.log`.

`main.jl` will:
- Load environment config from `../imprl/imprl/envs/structural_envs/env_configs/<setting>.yaml`
- Build the POMDP via `k_out_of_n_infinite.jl`
- Solve with SARSOP
- Write outputs under `./experiments/<timestamp-random_id>/`
- Save `env_config.yaml`, `pomdp.pomdp`, `policy.policy`, and logs

Use any available setting file name from:
- `imprl/imprl/envs/structural_envs/env_configs`

## Baseline Assets
- `SARSOP_models_and_policies/` stores generated models/policies by setting and run id.
- Archive file is typically `SARSOP_models_and_policies.tar` (for transfer/download workflows).

## File Map
- `main.jl`: Entry point for experiments
- `k_out_of_n_infinite.jl`: Defines an infinite-horizon k-out-of-n problem
- `benchmark_heuristic_policies.jl`: Runs simple rule-based policies on the k-out-of-n problem
- `point-based-solvers/`: Julia project (`Project.toml`, `Manifest.toml`)

## References
- [POMDPs.jl Documentation](https://juliapomdp.github.io/POMDPs.jl/stable/)
- [POMDPModels.jl Documentation](http://juliapomdp.github.io/POMDPs.jl/latest/)
