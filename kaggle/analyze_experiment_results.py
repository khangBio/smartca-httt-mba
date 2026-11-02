"""Audit existing MBA outputs and compare with the supplied paper (no retraining).

Paper values below were transcribed and visually verified from Table 3, p.8,
of docs/2305.05585v1.pdf. Comparisons to the paper are descriptive because
validation and training-data protocols differ. Requires numpy and pandas.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"]
PAPER = {
    "beibei": {"mf": [0.0901, 0.1438, 0.0668, 0.0855],
               "mba_mf": [0.1127, 0.1742, 0.0834, 0.1046],
               "mba_lightgcn": [0.1078, 0.1698, 0.0786, 0.1002]},
    "taobao": {"mf": [0.0303, 0.0420, 0.0190, 0.0224],
               "mba_mf": [0.0579, 0.0812, 0.0369, 0.0435],
               "mba_lightgcn": [0.0538, 0.0780, 0.0327, 0.0396]},
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bootstrap_ci(difference, rng, samples):
    estimates = []
    for start in range(0, samples, 16):
        ids = rng.integers(len(difference), size=(min(16, samples - start), len(difference)))
        estimates.extend(difference[ids].mean(axis=1).tolist())
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def compare(root, output, data_root, bootstrap_samples):
    root, output, data_root = Path(root), Path(output), Path(data_root)
    output.mkdir(parents=True, exist_ok=True)
    runs, scores, comparisons, curves, groups, times, paired = {}, [], [], [], [], [], []
    audit = dict(source_root=str(root.resolve()), paper="docs/2305.05585v1.pdf, Table 3, page 8",
                 bootstrap_samples=bootstrap_samples, runs={}, pairs={}, files={})
    for result_path in sorted(root.glob("*/*/result.json")):
        folder = result_path.parent
        result = json.loads(result_path.read_text(encoding="utf8"))
        metadata = json.loads((folder / "config.json").read_text(encoding="utf8"))
        config = metadata["config"]
        ds, method, seed = result["dataset"], result["variant"], result["seed"]
        if result["debug"] or config["max_steps"] or config["eval_user_limit"]:
            raise ValueError(f"Debug run cannot enter a research comparison: {folder}")
        if result["backbone"] != "mf":
            raise ValueError("This analysis compares the MF experiments only.")
        identity = (ds, method, seed)
        if identity in runs:
            raise ValueError(f"Duplicate run: {identity}")
        with np.load(folder / "test_per_user.npz", allow_pickle=False) as archive:
            per_user = {k: archive[k].copy() for k in archive.files}
        assert len(per_user["users"]) == result["users_evaluated"]
        assert all(np.isfinite(per_user[m]).all() for m in METRICS)
        metric_csv = pd.read_csv(folder / "metrics.csv").iloc[0]
        for metric in METRICS:
            np.testing.assert_allclose(per_user[metric].mean(), result["test"][metric], atol=1e-14, rtol=0)
            np.testing.assert_allclose(metric_csv[metric], result["test"][metric], atol=1e-14, rtol=0)
        # Verify the uploaded files match the local CSVs and the recorded hashes.
        for source_name, source in metadata["sources"].items():
            local = data_root / ds / f"{ds}_{source_name}.csv"
            if local.is_file():
                assert digest(local) == source["sha256"], local
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
        target_history = [x for x in history if x["stage"] == method]
        selected = next(x for x in target_history if x["epoch"] == result["best_epoch"])
        np.testing.assert_allclose(selected["validation"][result["selection_metric"]], result["best_validation"], atol=1e-14)
        stage_stats = {}
        for stage in dict.fromkeys(x["stage"] for x in history):
            entries = [x for x in history if x["stage"] == stage]
            valid = [x for x in entries if "validation" in x]
            stage_stats[stage] = dict(epochs=len([x for x in entries if "epoch" in x]),
                last_epoch=entries[-1].get("epoch"), train_seconds=sum(x.get("seconds", 0) for x in entries),
                reused_teacher=any("reused_cache" in x for x in entries),
                best_epoch=max(valid, key=lambda x: x["validation"]["Recall@20"])["epoch"] if valid else None)
            for entry in entries:
                if "validation" in entry:
                    curves.append(dict(dataset=ds, method=method, seed=seed, stage=stage, epoch=entry["epoch"],
                                       **entry["validation"]))
        target_stats = stage_stats[method]
        epoch_seconds = target_stats["train_seconds"] / target_stats["epochs"]
        count_buy = metadata["data"]["buy_fit"]
        count_click = target_history[0]["click_pairs"]
        steps_per_epoch = math.ceil(count_buy / config["batch_size"]) + math.ceil(count_click / config["batch_size"])
        times.append(dict(dataset=ds, method=method, seed=seed, target_epochs=target_stats["epochs"],
                          best_epoch=result["best_epoch"], target_train_seconds=target_stats["train_seconds"],
                          average_train_seconds_per_epoch=epoch_seconds, click_pairs_per_epoch=count_click,
                          click_retained_fraction=count_click / metadata["data"]["clicked_only"],
                          steps_per_epoch=steps_per_epoch, total_steps=steps_per_epoch * target_stats["epochs"]))
        run_audit = dict(result=result, config=config, data=metadata["data"], environment=metadata["environment"],
                         stages=stage_stats, selected_epoch_log=selected, last_epoch_log=target_history[-1],
                         metrics_equal_per_user_means=True, target_capped_at_max_epoch=target_stats["last_epoch"] == config["epochs"])
        beta_path = folder / "beta_by_user.npy"
        if beta_path.is_file():
            beta = np.load(beta_path, allow_pickle=False)
            run_audit["beta_by_user"] = dict(min=float(beta.min()), mean=float(beta.mean()), max=float(beta.max()))
        audit["runs"]["/".join(map(str, identity))] = run_audit
        scores.append(dict(dataset=ds, method=method, seed=seed, **result["test"]))
        for name in ["result.json", "config.json", "history.jsonl", "metrics.csv", "test_per_user.npz", "splits.npz"]:
            path = folder / name
            audit["files"][str(path.relative_to(root))] = digest(path)
        runs[identity] = dict(folder=folder, result=result, metadata=metadata, per_user=per_user)

    for ds, seed in sorted({(ds, seed) for ds, method, seed in runs}):
        base, adaptive = runs[(ds, "mba", seed)], runs[(ds, "adaptive", seed)]
        a, b = base["per_user"], adaptive["per_user"]
        np.testing.assert_array_equal(a["users"], b["users"])
        np.testing.assert_array_equal(a["buy_train_degree"], b["buy_train_degree"])
        for k in ["split_seed", "val_fraction", "select_metric", "top_k"]:
            assert base["metadata"]["config"][k] == adaptive["metadata"]["config"][k]
        assert base["metadata"]["sources"] == adaptive["metadata"]["sources"]
        pair_audit = dict(users_equal=True, users=len(a["users"]), degree_equal=True, hashes_equal={})
        for name in ["teacher_buy.pt", "teacher_pv.pt", "splits.npz"]:
            same = digest(base["folder"] / name) == digest(adaptive["folder"] / name)
            pair_audit["hashes_equal"][name] = same
            assert same, f"Different {name} between methods for {ds}"
        with np.load(base["folder"] / "splits.npz", allow_pickle=False) as split_a, np.load(adaptive["folder"] / "splits.npz", allow_pickle=False) as split_b:
            assert split_a.files == split_b.files
            assert all(np.array_equal(split_a[k], split_b[k]) for k in split_a.files)
            pair_audit["split_arrays_equal"] = True
            pair_audit["buy_validation_fraction_actual"] = len(split_a["buy_val"]) / len(split_a["buy_train"])
            for low, high in [(0, 3), (4, 6), (7, 999999)]:
                mask = (a["buy_train_degree"] >= low) & (a["buy_train_degree"] <= high)
                if not mask.any():
                    continue
                # IDs in NPZ per-user outputs are original IDs, whereas reliability uses mapped IDs.
                indices = np.searchsorted(split_a["user_ids"], a["users"][mask])
                reliability = split_a["reliability"][indices]
                for metric in METRICS:
                    x, y = float(a[metric][mask].mean()), float(b[metric][mask].mean())
                    groups.append(dict(dataset=ds, seed=seed, group=f"{low}-{high}" if high < 999999 else "7+",
                        users=int(mask.sum()), metric=metric, mba=x, adaptive=y, absolute_delta=y-x,
                        relative_delta_percent=100*(y/x-1) if x else None,
                        mean_reliability=float(reliability.mean()),
                        click_kl_weight_mean_upper_bound=float((0.2 + 0.8 * reliability).mean())))
        audit["pairs"][f"{ds}/{seed}"] = pair_audit
        teacher = base["result"]["purchase_teacher_test"]
        scores.append(dict(dataset=ds, method="purchase_teacher", seed=seed, **teacher))
        for paper_method, values in PAPER[ds].items():
            scores.append(dict(dataset=ds, method=f"paper_{paper_method}", seed="paper_mean_3_runs", **dict(zip(METRICS, values))))
        for metric, paper_value in zip(METRICS, PAPER[ds]["mba_mf"]):
            mba_value, new_value = base["result"]["test"][metric], adaptive["result"]["test"][metric]
            for candidate, value, reference, ref_value in [
                ("mba", mba_value, "paper_mba_mf", paper_value),
                ("adaptive", new_value, "paper_mba_mf", paper_value),
                ("adaptive", new_value, "mba", mba_value),
                ("mba", mba_value, "purchase_teacher", teacher[metric]),
                ("adaptive", new_value, "purchase_teacher", teacher[metric]),
            ]:
                comparisons.append(dict(dataset=ds, seed=seed, metric=metric, candidate=candidate, reference=reference,
                    candidate_value=value, reference_value=ref_value, absolute_delta=value-ref_value,
                    relative_delta_percent=100*(value/ref_value-1)))
            difference = b[metric] - a[metric]
            low, high = bootstrap_ci(difference, np.random.default_rng(42), bootstrap_samples)
            paired.append(dict(dataset=ds, seed=seed, metric=metric, users=len(difference),
                absolute_delta=float(difference.mean()), relative_delta_percent=100*float(difference.mean())/mba_value,
                ci_low=low, ci_high=high, users_improved=int((difference > 1e-14).sum()),
                users_worse=int((difference < -1e-14).sum()), users_tied=int((np.abs(difference) <= 1e-14).sum())))

    # Verify existing exported summary numbers without modifying source outputs.
    existing = pd.read_csv(root / "all_runs.csv")
    for row in existing.to_dict("records"):
        current = runs[(row["dataset"], row["method"], int(row["seed"]))]["result"]["test"]
        for m in METRICS:
            np.testing.assert_allclose(row[m], current[m], atol=1e-14, rtol=0)
    old_paired = pd.read_csv(root / "paired_differences.csv")
    for row in old_paired.to_dict("records"):
        new = next(x for x in paired if x["dataset"] == row["dataset"] and x["metric"] == row["metric"])
        np.testing.assert_allclose(new["absolute_delta"], row["absolute_gain"], atol=1e-14, rtol=0)
        if bootstrap_samples == 2000:
            np.testing.assert_allclose([new["ci_low"], new["ci_high"]],
                                      [row["user_bootstrap_ci_low"], row["user_bootstrap_ci_high"]], atol=1e-10, rtol=0)
    audit["source_summary_verified"] = True
    audit["seeds"] = sorted({seed for ds, method, seed in runs})
    audit["backbones"] = sorted({run["result"]["backbone"] for run in runs.values()})
    tables = dict(scores=scores, comparisons=comparisons, learning_curves=curves,
                  user_groups=groups, training_times=times, paired_recomputed=paired)
    for name, table in tables.items():
        pd.DataFrame(table).to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    (output / "audit_summary.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf8")
    print(pd.DataFrame(scores).to_string(index=False))
    print("\nRELATIVE DELTAS (%)")
    print(pd.DataFrame(comparisons).pivot_table(index=["dataset", "candidate", "reference"], columns="metric", values="relative_delta_percent").to_string())
    print("\nTARGET TRAINING TIMES (exclude validation/checkpoint writing)")
    print(pd.DataFrame(times).to_string(index=False))
    print("\nUSER GROUPS, RECALL@20")
    print(pd.DataFrame(groups).query("metric == 'Recall@20'").to_string(index=False))
    print("\nChecks passed: per-user means, source CSV hashes, teacher/split equality, recorded summaries and bootstrap CIs.")
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="kaggle/output")
    parser.add_argument("--output", default="output/analysis_mba_2026_11_02")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    compare(args.root, args.output, args.data_root, args.bootstrap_samples)
