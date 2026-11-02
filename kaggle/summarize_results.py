"""Summarize completed runs; debug/smoke results never enter research tables."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired_bootstrap(values, seed=42, resamples=2000):
    """User-level CI, conditional on the trained seeds, not a seed-level test."""
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, resamples, 16):
        indices = rng.integers(len(values), size=(min(16, resamples - start), len(values)))
        estimates.extend(values[indices].mean(axis=1).tolist())
    return np.quantile(estimates, [0.025, 0.975])


def summarize(root, baseline="mba", candidate="adaptive", resamples=2000):
    if resamples < 1:
        raise ValueError("resamples must be positive.")
    root = Path(root)
    rows, runs, skipped, signatures = [], {}, 0, {}
    for path in sorted(root.rglob("result.json")):
        result = json.loads(path.read_text(encoding="utf8"))
        if result.get("debug", False):
            skipped += 1
            continue
        method = result.get("method", result["variant"])
        identity = (result["dataset"], result["backbone"], method, result["seed"])
        if identity in runs:
            raise ValueError(f"Duplicate run identity {identity}; summarize distinct experiments separately.")
        config = json.loads((path.parent / "config.json").read_text(encoding="utf8"))
        # Mean/std across seeds must describe the same experiment, not mixed hyperparameters.
        ignored = {"seed", "output", "teacher_cache", "data_root", "device", "cpu_threads", "item_chunk", "eval_batch_size"}
        signature = dict(settings={k: v for k, v in config["config"].items() if k not in ignored},
                         sources={k: v["sha256"] for k, v in config["sources"].items()})
        group = identity[:3]
        if group in signatures and signatures[group] != signature:
            raise ValueError(f"Different data/settings across seeds for {group}; use separate method labels.")
        signatures[group] = signature
        runs[identity] = path
        rows.append(dict(dataset=result["dataset"], backbone=result["backbone"], method=method,
                         seed=result["seed"], **result["test"]))
    if not rows:
        print(f"No completed full runs. Ignored {skipped} debug runs.")
        return
    frame = pd.DataFrame(rows)
    metrics = [c for c in frame if "@" in c]
    summary = frame.groupby(["dataset", "backbone", "method"])[metrics].agg(["mean", "std", "count"])
    frame.to_csv(root / "all_runs.csv", index=False)
    summary.to_csv(root / "mean_std.csv")
    print(summary.to_string())
    differences = []
    for dataset, backbone in sorted({k[:2] for k in runs}):
        seed_a = {k[3] for k in runs if k[:3] == (dataset, backbone, baseline)}
        seed_b = {k[3] for k in runs if k[:3] == (dataset, backbone, candidate)}
        seeds = sorted(seed_a & seed_b)
        if not seeds:
            continue
        for metric in metrics:
            per_seed = []
            baseline_means = []
            reference_users = None
            for seed in seeds:
                folder_a = runs[(dataset, backbone, baseline, seed)].parent
                folder_b = runs[(dataset, backbone, candidate, seed)].parent
                config_a = json.loads((folder_a / "config.json").read_text(encoding="utf8"))
                config_b = json.loads((folder_b / "config.json").read_text(encoding="utf8"))
                hashes_a = {k: v["sha256"] for k, v in config_a["sources"].items()}
                hashes_b = {k: v["sha256"] for k, v in config_b["sources"].items()}
                if hashes_a != hashes_b or any(config_a["config"][k] != config_b["config"][k]
                                               for k in ("split_seed", "val_fraction", "select_metric", "top_k")):
                    raise ValueError("Cannot compare runs with different data or validation splits.")
                with np.load(folder_a / "test_per_user.npz") as a, np.load(folder_b / "test_per_user.npz") as b:
                    if not np.array_equal(a["users"], b["users"]):
                        raise ValueError("Paired evaluation requires identical users.")
                    if reference_users is not None and not np.array_equal(reference_users, a["users"]):
                        raise ValueError("Users differ across seeds.")
                    reference_users = a["users"].copy()
                    per_seed.append(b[metric] - a[metric])
                    baseline_means.append(float(a[metric].mean()))
            diff = np.stack(per_seed).mean(axis=0)
            low, high = paired_bootstrap(diff, resamples=resamples)
            denominator = np.mean(baseline_means)
            differences.append(dict(dataset=dataset, backbone=backbone, metric=metric, baseline=baseline,
                candidate=candidate, paired_seeds=len(seeds), users=len(diff), absolute_gain=diff.mean(),
                relative_gain_percent=100 * diff.mean() / denominator if denominator else None,
                user_bootstrap_ci_low=low, user_bootstrap_ci_high=high))
    if differences:
        pd.DataFrame(differences).to_csv(root / "paired_differences.csv", index=False)
        print("\nUser-bootstrap CIs are conditional on trained seeds; inspect seed standard deviations separately.")
        print(pd.DataFrame(differences).to_string(index=False))
    print(f"\nIgnored {skipped} debug runs. CSV files saved to {root.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/kaggle/working/mba_runs")
    p.add_argument("--baseline", default="mba")
    p.add_argument("--candidate", default="adaptive")
    p.add_argument("--resamples", type=int, default=2000)
    args = p.parse_args()
    summarize(args.root, args.baseline, args.candidate, args.resamples)
