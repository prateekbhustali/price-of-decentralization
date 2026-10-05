"""
Usage:
  python results/get_rollout_data.py               # best run per (setting, algorithm)
  python results/get_rollout_data.py --all-seeds   # every run_id in results_summary.csv

Notes:
 - Best mode picks the best run_id + checkpoint per (setting, algorithm) from
   results/data/results_summary.csv and saves to:
     results/data/rollout_data/k_out_of_n_infinite/<setting>/<ALG>/rollout_data.pkl
 - All-seeds mode rolls out every run_id (best_checkpoint per run) and saves to:
     results/data/rollout_data_all_seeds/k_out_of_n_infinite/<setting>/<ALG>/<run_id>.pkl
   Existing per-run files are skipped unless --overwrite is given. SARSOP and
   InspectRepair have no seeds and are skipped in this mode.
 - Recreates the eval environment using the run's config.yaml
   (model_checkpoints/k_out_of_n_infinite/<setting>/<ALG>/<run_id>/config.yaml)
 - Loads weights from model_weights/ and rolls out --episodes episodes
   (default: NUM_INFERENCE_EPISODES; for all-seeds runs you likely want --episodes 100)
"""

import argparse
import os
import pickle
from pathlib import Path
import itertools
import multiprocessing as mp

import pandas as pd
import torch

import imprl.agents
import imprl.envs
import yaml
import logging
import time
from imprl.post_process.policy_visualizer import PolicyVisualizer
from imprl.agents.SARSOP import SARSOPAgent
from imprl.baselines.inspection_repair import InspectRepairHeuristicAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV_NAME = "k_out_of_n_infinite"
SETTINGS = [f"hard-{k}-of-4_infinite" for k in range(1, 5)]

ALGORITHMS = [
    "DDQN",
    "JAC",
    "DCMAC",
    "IACC_PS",
    "QMIX_PS",
    "VDN_PS",
    "IAC_PS",
    "MAPPO_PS",
    "IPPO_PS",
    "InspectRepair",
    "SARSOP",
]
TIME_LIMIT = 20

# Episodes per rollout unless overridden with --episodes.
NUM_INFERENCE_EPISODES = 10_000

# Base directories using Path
RESULTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = RESULTS_DIR.parent

# Output roots per mode.
BEST_OUT_ROOT = RESULTS_DIR / "data" / "rollout_data" / ENV_NAME
ALL_SEEDS_OUT_ROOT = RESULTS_DIR / "data" / "rollout_data_all_seeds" / ENV_NAME
SARSOP_BASELINE_DIR = REPO_ROOT / "sarsop_baseline" / "SARSOP_models_and_policies"
SARSOP_SUMMARY_PATH = SARSOP_BASELINE_DIR / "summary.csv"

# Load consolidated summary used across plots
# Columns: run_id, setting, algorithm, best_cost, best_checkpoint, ...
results_summary_df = pd.read_csv(RESULTS_DIR / "data" / "results_summary.csv")
sarsop_summary_df = (
    pd.read_csv(SARSOP_SUMMARY_PATH) if SARSOP_SUMMARY_PATH.exists() else pd.DataFrame()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out trained policies.")
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Roll out every run_id from results_summary.csv instead of only the "
        "best one per (setting, algorithm).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=NUM_INFERENCE_EPISODES,
        help=f"Episodes per rollout (default: {NUM_INFERENCE_EPISODES}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="All-seeds mode: regenerate existing <run_id>.pkl files instead of skipping.",
    )
    return parser.parse_args()


def _load_run_config(run_dir: Path) -> dict:
    """Load the per-run config.yaml stored alongside checkpoints.

    This is a copy of experiments/configs/<ALG>.yaml with env_setting
    already set for that specific run. No fallbacks are used.
    """
    cfg = run_dir / "config.yaml"
    if not cfg.exists():
        raise FileNotFoundError(f"Missing per-run config: {cfg}")
    return yaml.safe_load(cfg.read_text())


def _select_runs(setting_dir: str, algorithm: str, all_seeds: bool):
    """Return [(run_id, checkpoint), ...] — every run, or only the best one."""
    data = results_summary_df[
        (results_summary_df["setting"] == setting_dir)
        & (results_summary_df["algorithm"] == algorithm)
    ]
    if data.empty:
        raise ValueError(
            f"No results found for setting={setting_dir}, algorithm={algorithm} in data/results_summary.csv"
        )
    if all_seeds:
        return [
            (str(row["run_id"]), int(row["best_checkpoint"]))
            for _, row in data.iterrows()
        ]
    best = data.loc[data["best_cost"].idxmin()]
    return [(str(best["run_id"]), int(best["best_checkpoint"]))]


def _load_agent_for_run(env, algorithm: str, run_dir: Path, checkpoint):
    alg_config = _load_run_config(run_dir)
    agent = imprl.agents.make(algorithm, env, alg_config, device)
    path = run_dir / "model_weights"
    if not path.is_dir():
        raise FileNotFoundError(
            f"Expected model_weights directory not found: {path}\n"
            f"Ensure models are normalized (tools/normalize_trained_models.py)."
        )
    agent.load_weights(str(path), checkpoint)
    return agent


def _make_env_and_agent(env_setting: str, alg: str, run_id: str, checkpoint):
    """Build the eval env from the run's config.yaml and load the trained agent."""
    run_dir = REPO_ROOT / "model_checkpoints" / ENV_NAME / env_setting / alg / run_id
    run_cfg = _load_run_config(run_dir)
    env_cfg = run_cfg.get("ENV_CONFIG", {}) if isinstance(run_cfg, dict) else {}
    # Use per-run config to set env kwargs (prefer inference_env_kwargs)
    inference_kwargs = dict(
        env_cfg.get("inference_env_kwargs") or env_cfg.get("kwargs") or {}
    )
    # Use the same rollout horizon for all algorithms in this script.
    inference_kwargs["time_limit"] = TIME_LIMIT

    single_agent = alg in ["DDQN", "JAC"]
    env = imprl.envs.make(
        ENV_NAME,
        env_setting,
        single_agent=single_agent,
        **inference_kwargs,
    )
    agent = _load_agent_for_run(env, alg, run_dir, checkpoint)
    return env, agent


def _select_sarsop_run_by_alpha_vectors(env_setting: str, rank: int = 1):
    """Pick SARSOP run by alpha-vector rank (1 = smallest)."""
    data = sarsop_summary_df[sarsop_summary_df["env_setting"] == env_setting]
    if data.empty:
        raise ValueError(f"No SARSOP runs found for env_setting={env_setting}")
    sort_cols = ["num_alpha_vectors", "run_id"]
    valid = []
    for run_id, nvec in data.sort_values(sort_cols)[
        ["run_id", "num_alpha_vectors"]
    ].itertuples(index=False, name=None):
        run_dir = SARSOP_BASELINE_DIR / env_setting / str(run_id)
        policy_path, pomdp_path = run_dir / "policy.policy", run_dir / "pomdp.pomdp"
        if policy_path.exists() and pomdp_path.exists():
            valid.append((str(run_id), int(nvec), policy_path, pomdp_path))
    if rank <= len(valid):
        return valid[rank - 1]
    raise ValueError(
        f"Requested rank={rank}, but only {len(valid)} valid SARSOP runs found for env_setting={env_setting}"
    )


def get_rollout_data(ap):
    return ap.get_sample_rollout()


def _rollout_and_save(env, agent, out_file: Path, episodes: int, nproc: int) -> None:
    """Common rollout for all agent types via PolicyVisualizer."""
    ap = PolicyVisualizer(env, agent)
    iterable = itertools.repeat(ap, episodes)
    with mp.Pool(nproc) as pool:
        all_data = pool.map(get_rollout_data, iterable)

    os.makedirs(out_file.parent, exist_ok=True)
    all_data = {f"rollout_data_{i}": data for i, data in enumerate(all_data)}
    with open(out_file, "wb") as f:
        pickle.dump(all_data, f)


if __name__ == "__main__":
    args = parse_args()
    episodes = args.episodes

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    nproc = max(1, mp.cpu_count())
    start_all = time.perf_counter()
    mode = "all-seeds" if args.all_seeds else "best-run"
    logging.info(
        f"Starting rollouts | mode={mode} | procs={nproc} | episodes={episodes} | time_limit={TIME_LIMIT}"
    )

    out_root = ALL_SEEDS_OUT_ROOT if args.all_seeds else BEST_OUT_ROOT

    for env_setting in SETTINGS:
        logging.info(f"Setting: {env_setting}")
        for alg in ALGORITHMS:

            if alg in ("SARSOP", "InspectRepair"):
                if args.all_seeds:
                    logging.info(f"Algorithm: {alg} | skipped in all-seeds mode (no seeds)")
                    continue
                t0 = time.perf_counter()
                try:
                    if alg == "SARSOP":
                        # Baseline rollouts via alpha-vector rank (default rank=1).
                        env = imprl.envs.make(
                            ENV_NAME, env_setting, single_agent=False, time_limit=TIME_LIMIT
                        )
                        run_id, num_alpha_vectors, policy_path, pomdp_path = (
                            _select_sarsop_run_by_alpha_vectors(env_setting, rank=1)
                        )
                        logging.info(
                            f"Algorithm: {alg} | run_id={run_id} | num_alpha_vectors={num_alpha_vectors}"
                        )
                        agent = SARSOPAgent(env, str(policy_path), str(pomdp_path))
                    else:
                        env = imprl.envs.make(
                            ENV_NAME,
                            env_setting,
                            single_agent=False,
                            percept_type="obs",
                            time_limit=TIME_LIMIT,
                        )
                        agent = InspectRepairHeuristicAgent(env)
                        logging.info(
                            f"Algorithm: InspectRepair | policy=InspectRepairHeuristic | percept=obs | time_limit={TIME_LIMIT}"
                        )

                    _rollout_and_save(
                        env, agent, out_root / env_setting / alg / "rollout_data.pkl",
                        episodes, nproc,
                    )
                    logging.info(
                        f"Saved {alg} rollouts for {env_setting} in {time.perf_counter()-t0:.1f}s"
                    )
                except Exception as e:
                    logging.exception(f"Failed for setting={env_setting}, alg={alg}: {e}")
                continue

            # Learned algorithms: best run only, or every run_id in all-seeds mode.
            try:
                runs = _select_runs(env_setting, alg, args.all_seeds)
            except ValueError as e:
                logging.exception(f"Failed for setting={env_setting}, alg={alg}: {e}")
                continue

            for run_id, checkpoint in runs:
                if args.all_seeds:
                    out_file = out_root / env_setting / alg / f"{run_id}.pkl"
                    if out_file.exists() and not args.overwrite:
                        logging.info(f"Algorithm: {alg} | run_id={run_id} | exists, skipping")
                        continue
                else:
                    out_file = out_root / env_setting / alg / "rollout_data.pkl"

                t0 = time.perf_counter()
                try:
                    logging.info(
                        f"Algorithm: {alg} | run_id={run_id} | checkpoint={checkpoint}"
                    )
                    env, agent = _make_env_and_agent(env_setting, alg, run_id, checkpoint)
                    _rollout_and_save(env, agent, out_file, episodes, nproc)
                    logging.info(
                        f"Saved {alg} rollouts for {env_setting} ({run_id}) in {time.perf_counter()-t0:.1f}s"
                    )
                except Exception as e:
                    logging.exception(
                        f"Failed for setting={env_setting}, alg={alg}, run_id={run_id}: {e}"
                    )

        logging.info("")

    # Final summary once
    logging.info(
        f"Saved rollouts under {out_root} | total time {time.perf_counter()-start_all:.1f}s"
    )
