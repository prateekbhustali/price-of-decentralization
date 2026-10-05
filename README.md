# The price of decentralization in managing engineering systems through multi-agent reinforcement learning

[![Zenodo Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23103891.svg)](https://doi.org/10.5281/zenodo.23103891)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ress.2026.113536-blue.svg)](https://doi.org/10.1016/j.ress.2026.113536)

This repository accompanies the paper and includes the [`imprl`](https://github.com/prateekbhustali/imprl) library used to study how multi-agent reinforcement learning decentralization affects performance when planning inspection and maintenance for engineering systems.

## Repository Structure
```text
.
├── experiments/               # published experiment configs and run instructions
│   ├── configs/
│   │   ├── k_out_of_n_infinite/
│   │   │   └── *.yaml
│   │   └── matrix_game/
│   │       └── *.yaml
│   └── ..
├── imprl/                     # published as a submodule containing the Python package
│   └── ..
├── sarsop_baseline/           # published Julia SARSOP baseline code
│   ├── main.jl
│   ├── ..
│   └── point-based-solvers/
│       └── *.toml
└── results/                   # published result tables, notebooks, and figures
    ├── data/ 
    │   └── *.csv
    ├── plot_results_main.ipynb
    ├── plot_policy_visualizations_kn_systems.ipynb
    ├── ..
    └── figures/
        └── *.png/.pdf
```

## Installation
Clone the repository with the `imprl` submodule and install the Python package from `imprl/`:

```bash
git clone --recurse-submodules https://github.com/prateekbhustali/price-of-decentralization.git
cd price-of-decentralization

# if you already cloned without submodules
git submodule update --init --recursive

# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# create and activate a virtual environment
uv venv --python 3.11

# activate the virtual environment
source .venv/bin/activate

# installs dependencies using uv.lock (including imprl)
# sync using the imprl project + lockfile, but install into the active root .venv
uv sync --project imprl --active
```

See [imprl/README.md](./imprl/README.md) for package-level installation details, optional dependency groups, and usage examples.

## Results Data

The files below are already included in this repository under `results/data/`.

- `results/data/climb_game_evals.csv`: evaluation results for climb-game experiments.
- `results/data/kn_infinite_results_epymarl.csv`: k-out-of-n infinite-horizon EPyMARL comparison results.
- `results/data/results_evals.csv`: Evaluation metrics across experimental runs for each checkpoint.
- `results/data/results_summary.csv`: summary of `results_evals.csv` with best performing checkpoint per run.


## Download Artifacts

You can download the following files from the Zenodo record for this project:

- `model_checkpoints.tar.gz`: all trained model checkpoints for all experiments.
- `SARSOP_models_and_policies.tar.gz` : SARSOP baseline models and policies for all environments.
- `rollout_data.tar.gz`: rollout data for `k_out_of_n_infinite` environments for all algorithms.

Use the interactive helper script from the repository root:

```bash
./download_artifacts.sh
```

The script downloads the Zenodo artifacts, and extracts them into the expected repository locations.

Once you have the model checkpoints and SARSOP baseline models/policies downloaded, you can run the notebooks in `results/` to reproduce the main results and figures in the paper.

## Citation

[The price of decentralization in managing engineering systems through multi-agent reinforcement learning](https://doi.org/10.1016/j.ress.2026.113536)

```bibtex
@article{bhustali_priceofdecentralization_2026,
title = {The price of decentralization in managing engineering systems through multi-agent reinforcement learning},
journal = {Reliability Engineering & System Safety},
pages = {113536},
year = {2026},
issn = {0951-8320},
doi = {10.1016/j.ress.2026.113536},
url = {https://www.sciencedirect.com/science/article/pii/S0951832026013396},
author = {Prateek Bhustali and Pablo G. Morato and Konstantinos G. Papakonstantinou and Charalampos P. Andriotis},
}
```
