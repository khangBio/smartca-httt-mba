"""Train the improved MBA_A (B3) from docs/huong_dan_cai_tien_MBA_A.pdf.

Standalone Kaggle entry point: upload this single .py file and the dataset CSVs.
Default: B3 = per-user support weights + linear KL warmup, with an MF backbone.
Use --variants B0 B3 to compare baseline MBA and MBA_A on the same validation split.

Example:
  python train_mba_a_kaggle.py --data-root /kaggle/input/my-mba-data \
      --dataset beibei --seeds 2020 --output /kaggle/working/mba_a_results

Dependencies: torch, numpy, pandas (no imports from the original repository).
The implementation follows the existing train_mba_kaggle.py engine, with MBA_A
selected by default. It does not reproduce the paper's test-selection protocol.
Validation selects checkpoints; held-out purchase test is evaluated afterwards.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = {"B0": (False, False), "B1": (True, False),
            "B2": (False, True), "B3": (True, True)}
IMPLEMENTATION_VERSION = "mba-kaggle-mf-v1"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pairs(path):
    frame = pd.read_csv(path, sep="\t", usecols=[0, 1])
    values = frame.to_numpy()
    if len(values) == 0 or not np.isfinite(values.astype(float)).all():
        raise ValueError(f"Empty or invalid interaction file: {path}")
    pairs = values.astype(np.int64)
    if not np.equal(values, pairs).all() or (pairs < 0).any():
        raise ValueError(f"User/item IDs must be nonnegative integers: {path}")
    return np.unique(pairs, axis=0)


def locate_files(root, dataset):
    """Accept a dataset folder, repo root, or an entire Kaggle input mount."""
    root = Path(root)
    name = f"{dataset}_buy_train.csv"
    direct = [root / name, root / dataset / name, root / "data" / dataset / name]
    candidates = [p for p in direct if p.is_file()]
    if not candidates:
        candidates = sorted(root.rglob(name))
    candidates = sorted(set(p.resolve() for p in candidates))
    if len(candidates) != 1:
        raise ValueError(f"Expected one {name} under {root}, found {candidates}. "
                         "Set --data-root to the intended dataset directory.")
    directory = candidates[0].parent
    paths = {key: directory / f"{dataset}_{suffix}.csv" for key, suffix in
             [("buy", "buy_train"), ("click", "pv_train"), ("test", "buy_test")]}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def split_by_user(pairs, ratio, seed):
    """Hold out per user, always retaining at least one training interaction."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    boundaries = np.r_[0, np.flatnonzero(np.diff(pairs[:, 0])) + 1, len(pairs)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group = pairs[start:end]
        count = min(len(group) - 1, max(1, int(round(len(group) * ratio))))
        if len(group) < 2:
            count = 0
        order = rng.permutation(len(group))
        train.extend(group[order[count:]])
        val.extend(group[order[:count]])
    return (np.asarray(train, dtype=np.int64).reshape(-1, 2),
            np.asarray(val, dtype=np.int64).reshape(-1, 2))


def pair_keys(pairs, items):
    return pairs[:, 0] * np.int64(items) + pairs[:, 1]


def contains(sorted_keys, keys):
    positions = np.searchsorted(sorted_keys, keys)
    valid = positions < len(sorted_keys)
    result = np.zeros(len(keys), dtype=bool)
    result[valid] = sorted_keys[positions[valid]] == keys[valid]
    return result


def as_user_sets(pairs):
    result = defaultdict(set)
    for user, item in pairs:
        result[int(user)].add(int(item))
    return dict(result)


@dataclass
class Data:
    buy: np.ndarray
    click: np.ndarray
    click_only: np.ndarray
    buy_val: np.ndarray
    click_val: np.ndarray
    test: np.ndarray
    users: int
    items: int
    buy_counts: np.ndarray
    click_counts: np.ndarray
    fingerprint: str
    manifest: dict


def prepare_data(args):
    paths = locate_files(args.data_root, args.dataset)
    buy_raw, click_raw, test = [read_pairs(paths[k]) for k in ("buy", "click", "test")]
    # IDs from all files define the common catalog, not training interactions.
    users = int(max(x[:, 0].max() for x in (buy_raw, click_raw, test))) + 1
    items = int(max(x[:, 1].max() for x in (buy_raw, click_raw, test))) + 1
    if users * items > np.iinfo(np.int64).max:
        raise ValueError("IDs too large. Remap IDs to contiguous integer indices.")
    if contains(np.sort(pair_keys(buy_raw, items)), pair_keys(test, items)).any():
        raise ValueError("Original buy_train and buy_test overlap; fix the dataset first.")
    buy, buy_val = split_by_user(buy_raw, args.val_ratio, args.split_seed)
    click, click_val = split_by_user(click_raw, args.val_ratio, args.split_seed + 1)
    # Match the repo's assumption that a training purchase implies a click.
    click = np.unique(np.concatenate([click, buy]), axis=0)
    click_keys = np.sort(pair_keys(click, items))
    overlap = contains(click_keys, pair_keys(click_val, items))
    removed_click_val = int(overlap.sum())
    click_val = click_val[~overlap]
    buy_keys = np.sort(pair_keys(buy, items))
    click_only = click[~contains(buy_keys, pair_keys(click, items))]
    if not len(buy_val) or not len(click_val):
        raise ValueError("Validation is empty. Need users with >=2 interactions per behavior.")
    counts_buy = np.bincount(buy[:, 0], minlength=users)
    counts_click = np.bincount(click[:, 0], minlength=users)
    if counts_click.max() >= items:
        raise ValueError("A user has interacted with every item: no unseen negative exists.")
    manifest = {
        "implementation": IMPLEMENTATION_VERSION, "dataset": args.dataset,
        "files_sha256": {k: file_hash(v) for k, v in paths.items()},
        "split_seed": args.split_seed, "val_ratio": args.val_ratio,
        "protocol": "train_clicks_only_plus_train_purchases; train_purchase_mask_at_eval",
        "users": users, "items": items,
        "counts": {"buy_train": len(buy), "click_train": len(click),
                   "click_only": len(click_only), "buy_val": len(buy_val),
                   "click_val": len(click_val), "buy_test": len(test),
                   "click_val_removed_because_in_train": removed_click_val},
        "test_users_without_train_purchase": int(sum(
            counts_buy[u] == 0 for u in np.unique(test[:, 0]))),
    }
    return Data(buy, click, click_only, buy_val, click_val, test, users, items,
                counts_buy, counts_click, signature(manifest), manifest)


class MF(nn.Module):
    def __init__(self, users, items, dim):
        super().__init__()
        self.user_embedding = nn.Embedding(users, dim)
        self.item_embedding = nn.Embedding(items, dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, users, items):
        return (self.user_embedding(users) * self.item_embedding(items)).sum(-1)

    def ratings(self, users):
        return torch.sigmoid(self.user_embedding(users) @ self.item_embedding.weight.T)


class NegativeSampler:
    def __init__(self, excluded, items):
        self.items = items
        self.keys = np.sort(pair_keys(excluded, items))
        if np.bincount(excluded[:, 0]).max() >= items:
            raise ValueError("No negative item available for at least one user")

    def sample(self, users, rng):
        result = rng.integers(self.items, size=len(users), dtype=np.int64)
        rejected = contains(self.keys, users * self.items + result)
        attempts = 0
        while rejected.any():
            result[rejected] = rng.integers(self.items, size=int(rejected.sum()))
            rejected = contains(self.keys, users * self.items + result)
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("Negative sampling stalled; check catalog density")
        return result


def batches(pairs, size, rng):
    order = rng.permutation(len(pairs))
    for start in range(0, len(order), size):
        yield pairs[order[start:start + size]]


def tensor(array, device):
    return torch.as_tensor(array, dtype=torch.long, device=device)


def l2(model, users):
    # Same full-target-parameter regularizer and normalization as original repo.
    return sum(p.square().sum() for p in model.parameters()) / (2 * users)


def log(prob):
    return torch.log(prob + 1e-5)


def bernoulli_kl(teacher_logits, target_logits):
    p = torch.sigmoid(teacher_logits.detach())
    q = torch.sigmoid(target_logits)
    return p * (log(p) - log(q)) + (1 - p) * (log(1 - p) - log(1 - q))


def observation_log_likelihood(h1_logits, h2_logits, target_logits, labels, phase, c1, c2):
    """Matches DN/DP functions in the repository, including negative C2 sign."""
    q = torch.sigmoid(target_logits)
    if phase == "DN":
        h1 = torch.sigmoid(h1_logits)
        positive = log(h1) * q - c1 * (1 - q)
        negative = log(1 - h1) * q
    else:
        h2 = torch.sigmoid(h2_logits)
        positive = log(h2) * (1 - q)
        negative = log(1 - h2) * (1 - q) - c2 * q
    return torch.where(labels, positive, negative)


def support_weights(data, scale, epsilon, device):
    scores = []
    for counts in (data.buy_counts, data.click_counts):
        positive = counts[counts > 0]
        k = scale * (float(np.median(positive)) if len(positive) else 1.0)
        scores.append(counts / (counts + k))
    denominator = scores[0] + scores[1] + 2 * epsilon
    return tuple(torch.as_tensor(2 * (score + epsilon) / denominator,
                                 dtype=torch.float32, device=device) for score in scores)


@torch.no_grad()
def evaluate(model, truth_pairs, excluded_pairs, args, teacher=None, beta=1.0,
             buy_counts=None, return_users=False):
    model.eval()
    if teacher is not None:
        teacher.eval()
    truths = as_user_sets(truth_pairs)
    exclusions = as_user_sets(excluded_pairs)
    user_ids = sorted(truths)
    per_user = []
    max_k = min(max(args.top_k), model.item_embedding.num_embeddings)
    discounts = 1.0 / np.log2(np.arange(2, max(args.top_k) + 2))
    for start in range(0, len(user_ids), args.eval_batch_size):
        ids = user_ids[start:start + args.eval_batch_size]
        scores = model.ratings(tensor(ids, args.device))
        if teacher is not None:
            scores = beta * scores + (1 - beta) * teacher.ratings(tensor(ids, args.device))
        rows, columns = [], []
        for row, user in enumerate(ids):
            seen = exclusions.get(user, set())
            if seen & truths[user]:
                raise ValueError("Evaluation truth is masked by training positives")
            rows.extend([row] * len(seen))
            columns.extend(seen)
        if rows:
            scores[tensor(rows, args.device), tensor(columns, args.device)] = -torch.inf
        predictions = scores.topk(max_k, dim=1).indices.cpu().numpy()
        for user, prediction in zip(ids, predictions):
            true = truths[user]
            hits = np.asarray([int(item) in true for item in prediction], dtype=float)
            row = {"user": user}
            for k in args.top_k:
                row[f"recall@{k}"] = float(hits[:k].sum() / len(true))
                ideal = discounts[:min(k, len(true))].sum()
                row[f"ndcg@{k}"] = float((hits[:k] * discounts[:len(hits[:k])]).sum() / ideal)
            if buy_counts is not None:
                n = int(buy_counts[user])
                row["group"] = "0" if n == 0 else "1-5" if n <= 5 else "6-10" if n <= 10 else "11+"
            per_user.append(row)
    keys = [f"{metric}@{k}" for metric in ("recall", "ndcg") for k in args.top_k]
    result = {key: float(np.mean([row[key] for row in per_user])) for key in keys}
    result["users"] = len(per_user)
    if buy_counts is not None:
        result["groups"] = {}
        for group in ("0", "1-5", "6-10", "11+"):
            selected = [row for row in per_user if row["group"] == group]
            if selected:
                result["groups"][group] = {"users": len(selected), **{
                    key: float(np.mean([row[key] for row in selected])) for key in keys}}
    return (result, per_user) if return_users else result


def append_log(path, row):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_teacher(data, behavior, seed, args, folder):
    teacher_config = {
        "data": data.fingerprint, "behavior": behavior, "seed": seed,
        "dim": args.dim, "lr": args.lr, "lambda0": args.lambda0,
        "batch_size": args.batch_size, "epochs": args.pretrain_epochs,
        "patience": args.patience, "top_k": args.top_k,
        "torch": torch.__version__, "device": str(args.device),
    }
    tag = signature(teacher_config)
    checkpoint = folder / f"teacher_{behavior}_{seed}_{tag}.pt"
    model = MF(data.users, data.items, args.dim).to(args.device)
    if checkpoint.exists() and not args.retrain_teachers:
        model.load_state_dict(torch.load(checkpoint, map_location=args.device))
        print(f"Reusing teacher: {checkpoint.name}", flush=True)
    else:
        seed_everything(seed)
        model = MF(data.users, data.items, args.dim).to(args.device)
        pairs, val = ((data.buy, data.buy_val) if behavior == "buy"
                      else (data.click, data.click_val))
        sampler = NegativeSampler(pairs, data.items)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        rng = np.random.default_rng(seed)
        best, stale = -math.inf, 0
        started = time.perf_counter()
        (folder / f"teacher_{behavior}_{seed}_{tag}.jsonl").write_text("", encoding="utf-8")
        for epoch in range(args.pretrain_epochs):
            model.train()
            total, seen = 0.0, 0
            for batch in batches(pairs, args.batch_size, rng):
                users, positive = tensor(batch[:, 0], args.device), tensor(batch[:, 1], args.device)
                negative = tensor(sampler.sample(batch[:, 0], rng), args.device)
                loss = F.softplus(model(users, negative) - model(users, positive)).mean()
                loss = loss + args.lambda0 * l2(model, data.users)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)
                seen += len(batch)
            metrics = evaluate(model, val, pairs, args)
            score = metrics[f"recall@{max(args.top_k)}"]
            record = {"epoch": epoch + 1, "loss": total / seen, "validation": metrics}
            append_log(folder / f"teacher_{behavior}_{seed}_{tag}.jsonl", record)
            print(f"teacher={behavior} seed={seed} epoch={epoch+1} "
                  f"loss={total/seen:.5f} val={score:.5f}", flush=True)
            if score > best:
                best, stale = score, 0
                torch.save(model.state_dict(), checkpoint)
                save_json(checkpoint.with_suffix(".json"), {
                    "config": teacher_config, "best_epoch": epoch + 1,
                    "validation": metrics, "seconds_to_best": time.perf_counter() - started})
            else:
                stale += 1
            if stale >= args.patience:
                break
        metadata_path = checkpoint.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({"total_train_seconds": time.perf_counter() - started,
                         "epochs_trained": epoch + 1})
        save_json(metadata_path, metadata)
        model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tag


def train_variant(data, teachers, teacher_tags, seed, variant, args, base_folder):
    adaptive, warmup = VARIANTS[variant]
    config = {"implementation": IMPLEMENTATION_VERSION, "data": data.fingerprint,
              "teacher_tags": teacher_tags, "seed": seed, "variant": variant,
              "dim": args.dim, "batch_size": args.batch_size, "lr": args.lr,
              "lambda1": args.lambda1, "alpha": args.alpha, "c1": args.c1, "c2": args.c2,
              "beta": args.beta, "epochs": args.epochs, "patience": args.patience,
              "warmup_epochs": args.warmup_epochs if warmup else 0,
              "support_scale": args.support_scale if adaptive else None,
              "epsilon": args.epsilon if adaptive else None, "top_k": args.top_k,
              "torch": torch.__version__, "device": str(args.device)}
    folder = base_folder / f"{variant}_seed{seed}_{signature(config)}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        print(f"Already completed: {folder.name}; use --force to rerun", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))
    if result_path.exists():
        result_path.unlink()
    (folder / "history.jsonl").write_text("", encoding="utf-8")
    save_json(folder / "config.json", config)
    # All variants receive identical initial weights and sampling RNG streams.
    seed_everything(seed)
    target = MF(data.users, data.items, args.dim).to(args.device)
    helpers = nn.ModuleList([MF(data.users, data.items, args.dim) for _ in range(4)]).to(args.device)
    optimizers = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in [target, *helpers]]
    weights_buy, weights_click = support_weights(data, args.support_scale, args.epsilon, args.device)
    sampler = NegativeSampler(data.click, data.items)
    rng = np.random.default_rng(seed)
    best, stale, phase, best_epoch = -math.inf, 0, "DP", 0
    checkpoint = folder / "best_target.pt"
    started = time.perf_counter()
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    for epoch in range(args.epochs):
        target.train()
        helpers.train()
        alpha = args.alpha * (min(1.0, (epoch + 1) / args.warmup_epochs) if warmup else 1.0)
        total, seen = 0.0, 0
        # Same two-stage ordering as repo: purchases, then click-only pairs.
        for pairs, purchased in ((data.buy, True), (data.click_only, False)):
            for batch in batches(pairs, args.batch_size, rng):
                u_np = np.concatenate([batch[:, 0], batch[:, 0]])
                i_np = np.concatenate([batch[:, 1], sampler.sample(batch[:, 0], rng)])
                users, items = tensor(u_np, args.device), tensor(i_np, args.device)
                size = len(batch)
                click_labels = torch.cat([torch.ones(size), torch.zeros(size)]).bool().to(args.device)
                buy_labels = click_labels if purchased else torch.zeros_like(click_labels)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                logits = target(users, items)
                # helper order: purchase h1/h2, click h1/h2.
                # Only the active helper pair is differentiated and stepped.
                if phase == "DN":
                    buy_h1, click_h1 = helpers[0](users, items), helpers[2](users, items)
                    buy_h2 = click_h2 = torch.zeros_like(logits)
                else:
                    buy_h2, click_h2 = helpers[1](users, items), helpers[3](users, items)
                    buy_h1 = click_h1 = torch.zeros_like(logits)
                likelihood_buy = observation_log_likelihood(
                    buy_h1, buy_h2, logits, buy_labels, phase, args.c1, args.c2)
                likelihood_click = observation_log_likelihood(
                    click_h1, click_h2, logits, click_labels, phase, args.c1, args.c2)
                with torch.no_grad():
                    teacher_buy = teachers[0](users, items)
                    teacher_click = teachers[1](users, items)
                kl_buy = bernoulli_kl(teacher_buy, logits)
                kl_click = bernoulli_kl(teacher_click, logits)
                if adaptive:
                    kl_buy = weights_buy[users] * kl_buy
                    kl_click = weights_click[users] * kl_click
                terms = alpha * (kl_buy + kl_click) - likelihood_buy - likelihood_click
                # Original sums mean(pos) and mean(neg), rather than one joint mean.
                loss = terms[:size].mean() + terms[size:].mean()
                loss = loss + args.lambda1 * l2(target, data.users)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at {variant}, epoch {epoch+1}")
                loss.backward()
                optimizers[0].step()
                for helper_index in ((0, 2) if phase == "DN" else (1, 3)):
                    optimizers[helper_index + 1].step()
                phase = "DN" if phase == "DP" else "DP"
                total += float(loss.detach()) * size
                seen += size
        metrics = evaluate(target, data.buy_val, data.buy, args, teachers[0], args.beta)
        score = metrics[f"recall@{max(args.top_k)}"]
        record = {"epoch": epoch + 1, "loss": total / seen, "alpha": alpha,
                  "validation": metrics, "elapsed_seconds": time.perf_counter() - started}
        append_log(folder / "history.jsonl", record)
        print(f"{variant} seed={seed} epoch={epoch+1} alpha={alpha:.3g} "
              f"loss={total/seen:.5f} val={score:.5f}", flush=True)
        if score > best:
            best, stale, best_epoch = score, 0, epoch + 1
            torch.save(target.state_dict(), checkpoint)
            save_json(folder / "best_validation.json", record)
        else:
            stale += 1
        if stale >= args.patience:
            break
    train_seconds = time.perf_counter() - started
    target.load_state_dict(torch.load(checkpoint, map_location=args.device))
    metrics, rows = evaluate(target, data.test, data.buy, args, teachers[0], args.beta,
                             data.buy_counts, return_users=True)
    with open(folder / "test_per_user.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"dataset": args.dataset, "variant": variant, "seed": seed,
              "data_fingerprint": data.fingerprint, "config_fingerprint": signature(config),
              "best_epoch": best_epoch, "best_validation_recall": best,
              "target_train_seconds": train_seconds,
              "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(args.device))
              if args.device.type == "cuda" else None,
              "test": metrics, "folder": str(folder),
              "note": "Test is evaluated once after validation selection; not paper reproduction."}
    save_json(result_path, result)
    print(f"TEST {variant} seed={seed}: {json.dumps(metrics)}", flush=True)
    return result


def write_summary(results, output, top_k):
    rows = []
    keys = [f"{metric}@{k}" for metric in ("recall", "ndcg") for k in top_k]
    for result in results:
        rows.append({"dataset": result["dataset"], "variant": result["variant"],
                     "seed": result["seed"], "best_epoch": result["best_epoch"],
                     "target_train_seconds": result["target_train_seconds"],
                     **{k: result["test"][k] for k in keys}})
    pd.DataFrame(rows).to_csv(output / "comparison_per_seed.csv", index=False)
    aggregate = []
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        row = {"variant": variant, "seeds": len(selected)}
        for key in keys:
            values = [x[key] for x in selected]
            row[key + "_mean"] = float(np.mean(values))
            row[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            baseline = {x["seed"]: x[key] for x in rows if x["variant"] == "B0"}
            paired = [x for x in selected if x["seed"] in baseline]
            if paired:
                base_mean = float(np.mean([baseline[x["seed"]] for x in paired]))
                row[key + "_gain_pct_vs_B0"] = (
                    100 * (float(np.mean([x[key] for x in paired])) / base_mean - 1)
                    if base_mean else None)
        aggregate.append(row)
    pd.DataFrame(aggregate).to_csv(output / "comparison_mean_std.csv", index=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--dataset", choices=["beibei", "taobao"], default="beibei")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/mba_a_results"))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["B3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2020])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda0", type=float, default=1e-4)
    parser.add_argument("--lambda1", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--c1", type=float, default=100)
    parser.add_argument("--c2", type=float, default=100)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--support-scale", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force", action="store_true", help="Retrain already completed target runs")
    parser.add_argument("--retrain-teachers", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Validate and save splits without training")
    # Jupyter/Colab injects `-f <kernel.json>` when code is run in a cell.
    # Accept only this known launcher option; keep typo checking for other flags.
    parser.add_argument("-f", dest="_jupyter_kernel_file", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 0 < args.val_ratio < 1 or not 0 <= args.beta <= 1:
        parser.error("Require 0 < val-ratio < 1 and 0 <= beta <= 1")
    for name in ("epochs", "pretrain_epochs", "patience", "batch_size", "eval_batch_size",
                 "dim", "warmup_epochs", "lr", "support_scale", "epsilon"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if min(args.alpha, args.lambda0, args.lambda1, args.c1, args.c2) < 0:
        parser.error("Loss weights must be nonnegative")
    if not args.top_k or min(args.top_k) <= 0:
        parser.error("top-k must contain positive integers")
    args.top_k = sorted(set(args.top_k))
    args.variants = list(dict.fromkeys(args.variants))
    args.seeds = list(dict.fromkeys(args.seeds))
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA not available. Enable a Kaggle GPU or use --device cpu")
    return args


def main(argv=None):
    args = parse_args(argv)
    print(f"Python={platform.python_version()} Torch={torch.__version__} "
          f"NumPy={np.__version__} device={args.device}", flush=True)
    data = prepare_data(args)
    args.output.mkdir(parents=True, exist_ok=True)
    folder = args.output / f"{args.dataset}_{data.fingerprint}"
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "data_manifest.json", data.manifest)
    np.savez_compressed(folder / "splits.npz", buy_train=data.buy, click_train=data.click,
                        buy_val=data.buy_val, click_val=data.click_val, buy_test=data.test)
    print(json.dumps(data.manifest, indent=2), flush=True)
    if args.prepare_only:
        print(f"Prepared splits: {folder}")
        return
    teacher_folder = folder / "teachers"
    teacher_folder.mkdir(exist_ok=True)
    results = []
    for seed in args.seeds:
        teacher_started = time.perf_counter()
        buy_teacher, buy_tag = train_teacher(data, "buy", seed, args, teacher_folder)
        click_teacher, click_tag = train_teacher(data, "click", seed, args, teacher_folder)
        teacher_wall = time.perf_counter() - teacher_started
        print(f"Teacher preparation (may reuse cache): {teacher_wall:.1f}s", flush=True)
        for variant in args.variants:
            result = train_variant(data, (buy_teacher, click_teacher), (buy_tag, click_tag),
                                   seed, variant, args, folder)
            results.append(result)
            write_summary(results, folder, args.top_k)
    print(f"DONE. Results: {folder / 'comparison_mean_std.csv'}", flush=True)


if __name__ == "__main__":
    main()
