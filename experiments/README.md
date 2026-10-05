# Experiments

This directory contains Hydra configs used in the paper. The training entry points are in `imprl/`.

## Contents

```text
experiments/
├── configs/
│   ├── k_out_of_n_infinite/
│   │   └── *.yaml
│   └── matrix_game/
│       └── *.yaml
└── README.md
```

- `configs/`
    - `./k_out_of_n_infinite/`: Hydra configs for `k_out_of_n_infinite` environments.
    - `./matrix_game/`: Hydra configs for `matrix_game` environments.
- `imprl/train_and_log.py`: training entry point for DCMAC, DDQN, IACC-PS, IAC-PS, JAC, QMIX-PS, and VDN-PS.
- `imprl/imprl/agents/IPPO_PS.py`: training entry point for IPPO_PS.
- `imprl/imprl/agents/MAPPO_PS.py`: training entry point for MAPPO_PS.

## How To Run

Run commands from the repository root:

```bash
cd <repo-root>
```

The selected config already specifies the environment name, environment setting, and the main training hyperparameters, so in most cases you only need to choose `--config-path` and `--config-name`.

Hydra resolves `--config-path` relative to the script location, not the repository root.
- For `imprl/train_and_log.py`, use `../experiments/...`
- For `imprl/imprl/agents/IPPO_PS.py` and `imprl/imprl/agents/MAPPO_PS.py`, use `../../../experiments/...`

Although there are also default configs under `imprl/imprl/agents/configs`, use the configs in `experiments/configs` if you want to reproduce the results reported in the paper.

Each run stores checkpoints in the config-defined `CHECKPOINT_DIR` and also saves a copy of the resolved config, so the same setup can be reconstructed later during inference or policy visualisation.

### 1) Train non-PPO algorithms (`imprl/train_and_log.py`)

Default run:

```bash
python imprl/train_and_log.py
```

Explicit config examples:

```bash
python imprl/train_and_log.py --config-path ../experiments/configs/k_out_of_n_infinite --config-name DCMAC
python imprl/train_and_log.py --config-path ../experiments/configs/k_out_of_n_infinite --config-name DDQN
python imprl/train_and_log.py --config-path ../experiments/configs/matrix_game --config-name JAC
python imprl/train_and_log.py --config-path ../experiments/configs/k_out_of_n_infinite --config-name DCMAC WANDB.mode=disabled
```

### 2) Train IPPO_PS

```bash
python imprl/imprl/agents/IPPO_PS.py --config-path ../../../experiments/configs/k_out_of_n_infinite --config-name IPPO_PS
python imprl/imprl/agents/IPPO_PS.py --config-path ../../../experiments/configs/matrix_game --config-name IPPO_PS
python imprl/imprl/agents/IPPO_PS.py --config-path ../../../experiments/configs/matrix_game --config-name IPPO_PS WANDB.mode=disabled
```

### 3) Train MAPPO_PS

```bash
python imprl/imprl/agents/MAPPO_PS.py --config-path ../../../experiments/configs/k_out_of_n_infinite --config-name MAPPO_PS
python imprl/imprl/agents/MAPPO_PS.py --config-path ../../../experiments/configs/matrix_game --config-name MAPPO_PS
python imprl/imprl/agents/MAPPO_PS.py --config-path ../../../experiments/configs/matrix_game --config-name MAPPO_PS WANDB.mode=disabled
```

## Notes

- WandB behavior is controlled in each YAML under `WANDB`.
- For additional command-line overrides and composition patterns, refer to the [Hydra documentation](https://hydra.cc/docs/intro/).
