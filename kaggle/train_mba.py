"""MBA and an experimental adaptive extension, standalone on Kaggle.

Original MBA: Xin et al., SIGIR 2023, arXiv:2305.05585v1.
Objective adapted from https://github.com/LiuXiangYuan/MBA (MIT).
Copyright (c) 2023 LiuXiangYuan. See the accompanying LICENSE.

Dependencies: numpy, pandas, torch. No SciPy, internet, or repository import.
The extension is a research hypothesis, not a reproduced paper result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


PRESETS = {
    ("beibei", "mf"): dict(alpha=10.0, c1=100.0, c2=100.0, reg=1e-4, beta=0.7),
    ("taobao", "mf"): dict(alpha=1.0, c1=1.0, c2=1.0, reg=1e-4, beta=0.8),
    ("beibei", "lightgcn"): dict(alpha=10.0, c1=10.0, c2=10.0, reg=1e-6, beta=1.0),
    ("taobao", "lightgcn"): dict(alpha=1000.0, c1=1000.0, c2=1.0, reg=1e-6, beta=0.8),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def keys(pairs, n_items):
    return pairs[:, 0].astype(np.int64) * n_items + pairs[:, 1]


def contains(sorted_keys, query):
    if len(sorted_keys) == 0:
        return np.zeros(np.shape(query), dtype=bool)
    idx = np.searchsorted(sorted_keys, query)
    return (idx < len(sorted_keys)) & (sorted_keys[np.minimum(idx, len(sorted_keys) - 1)] == query)


def unique_pairs(pairs):
    return np.unique(np.asarray(pairs, dtype=np.int64).reshape(-1, 2), axis=0)


def exclude_pairs(pairs, excluded, n_items):
    return pairs[~contains(np.sort(keys(excluded, n_items)), keys(pairs, n_items))]


def split_by_user(pairs, fraction, seed):
    """Hold out per user, retaining >=1 training edge; never uses test labels."""
    pairs = unique_pairs(pairs)
    rng = np.random.default_rng(seed)
    holdout = np.zeros(len(pairs), dtype=bool)
    boundaries = np.r_[0, np.flatnonzero(np.diff(pairs[:, 0])) + 1, len(pairs)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        size = end - start
        if size >= 2:
            count = min(size - 1, max(1, int(round(size * fraction))))
            holdout[rng.choice(np.arange(start, end), size=count, replace=False)] = True
    return pairs[~holdout], pairs[holdout]


def read_pairs(path):
    """Accept the repository's TSV-with-.csv-extension and ordinary numeric CSV."""
    with open(path, encoding="utf-8-sig") as stream:
        first = stream.readline().strip()
    sep = "\t" if "\t" in first else "," if "," in first else r"\s+"
    fields = first.replace(",", " ").split()
    header = None if len(fields) >= 2 and all(x.lstrip("-").isdigit() for x in fields[:2]) else 0
    frame = pd.read_csv(path, sep=sep, header=header, usecols=[0, 1], dtype=np.int64)
    pairs = frame.to_numpy(dtype=np.int64)
    if len(pairs) == 0 or (pairs < 0).any():
        raise ValueError(f"Empty data or negative IDs: {path}")
    return unique_pairs(pairs)


def discover_data(root, dataset):
    root = Path(root)
    matches = sorted(root.rglob(f"{dataset}_buy_train.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected one {dataset}_buy_train.csv under {root}; found {matches}. "
                         "Set --data-root to the intended dataset folder.")
    folder = matches[0].parent
    paths = {name: folder / f"{dataset}_{name}.csv" for name in ("buy_train", "buy_test", "pv_train")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    # pv_test is intentionally unused in this validation-based protocol.
    return paths


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_data(paths, val_fraction, split_seed):
    raw = {k: read_pairs(p) for k, p in paths.items()}
    # Transductive ID vocabulary only. No held-out labels enter losses or graphs.
    user_ids = np.unique(np.concatenate([x[:, 0] for x in raw.values()]))
    item_ids = np.unique(np.concatenate([x[:, 1] for x in raw.values()]))
    mapped = {k: np.stack([np.searchsorted(user_ids, a[:, 0]),
                           np.searchsorted(item_ids, a[:, 1])], axis=1) for k, a in raw.items()}
    n_users, n_items = len(user_ids), len(item_ids)
    if contains(np.sort(keys(mapped["buy_train"], n_items)), keys(mapped["buy_test"], n_items)).any():
        raise ValueError("buy_train and buy_test share user-item pairs; fix the split first.")
    buy_fit, buy_val = split_by_user(mapped["buy_train"], val_fraction, split_seed)
    pv_fit_raw, pv_val = split_by_user(mapped["pv_train"], val_fraction, split_seed + 1)
    pv_fit = unique_pairs(np.concatenate([pv_fit_raw, buy_fit]))
    # Unioning buy_fit can otherwise put a PV validation pair back in the graph.
    pv_val = exclude_pairs(pv_val, pv_fit, n_items)
    clicked_only = exclude_pairs(pv_fit, buy_fit, n_items)
    if len(buy_val) == 0 or len(pv_val) == 0:
        raise ValueError("Need nonempty purchase and click validation; users need >=2 distinct items.")
    degree_buy = np.bincount(buy_fit[:, 0], minlength=n_users)
    degree_pv = np.bincount(pv_fit[:, 0], minlength=n_users)
    prior = len(buy_fit) / len(pv_fit)
    # Shrink sparse user conversion estimates towards the global training ratio.
    conversion = (degree_buy + 20.0 * prior) / (degree_pv + 20.0)
    reliability = (conversion / (conversion + prior)).astype(np.float32)
    data = dict(n_users=n_users, n_items=n_items, user_ids=user_ids, item_ids=item_ids,
                buy_fit=buy_fit, buy_val=buy_val, buy_train=mapped["buy_train"],
                buy_test=mapped["buy_test"], pv_fit=pv_fit, pv_val=pv_val,
                clicked_only=clicked_only, reliability=reliability, degree_buy=degree_buy)
    data["stats"] = dict(users=n_users, items=n_items, buy_fit=len(buy_fit), buy_val=len(buy_val),
                         buy_test=len(mapped["buy_test"]), pv_fit=len(pv_fit), pv_val=len(pv_val),
                         clicked_only=len(clicked_only),
                         test_users_without_fit_buy=int((degree_buy[np.unique(mapped["buy_test"][:, 0])] == 0).sum()))
    return data


class NegativeSampler:
    """Vectorized rejection sampling against TRAINING positives only."""
    def __init__(self, observed, n_users, n_items, rng, popularity_mix=0.0):
        self.n_items, self.rng = n_items, rng
        self.seen = np.sort(keys(observed, n_items))
        self.degree = np.bincount(observed[:, 0], minlength=n_users)
        self.mix = popularity_mix
        counts = np.bincount(observed[:, 1], minlength=n_items).astype(float)
        self.popularity = np.sqrt(counts + 1.0)
        self.popularity /= self.popularity.sum()

    def _draw(self, size):
        items = self.rng.integers(self.n_items, size=size)
        if self.mix:
            mask = self.rng.random(size) < self.mix
            items[mask] = self.rng.choice(self.n_items, size=int(mask.sum()), p=self.popularity)
        return items

    def sample(self, users, count=1):
        users = np.repeat(np.asarray(users, dtype=np.int64), count)
        if (self.degree[users] >= self.n_items).any():
            raise ValueError("A user has observed the entire catalog; no valid negative exists.")
        items = self._draw(len(users))
        bad = contains(self.seen, users * self.n_items + items)
        for _ in range(100):
            if not bad.any():
                break
            items[bad] = self._draw(int(bad.sum()))
            bad = contains(self.seen, users * self.n_items + items)
        # Bounded fallback for tiny or extremely dense data, rather than an infinite loop.
        for idx in np.flatnonzero(bad):
            available = np.flatnonzero(~contains(self.seen, users[idx] * self.n_items + np.arange(self.n_items)))
            items[idx] = self.rng.choice(available)
        return items.reshape(-1, count)


def make_graph(pairs, n_users, n_items, device):
    user = pairs[:, 0]
    item = pairs[:, 1] + n_users
    row, col = np.r_[user, item], np.r_[item, user]
    degree = np.bincount(row, minlength=n_users + n_items).astype(np.float32)
    values = 1.0 / np.sqrt(degree[row] * degree[col])
    graph = torch.sparse_coo_tensor(torch.from_numpy(np.stack([row, col])),
                                    torch.from_numpy(values),
                                    (n_users + n_items, n_users + n_items))
    return graph.coalesce().to(device)


class Recommender(nn.Module):
    def __init__(self, n_users, n_items, dim=32, layers=0, graph=None):
        super().__init__()
        self.n_users, self.n_items, self.layers = n_users, n_items, layers
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.1 if layers else 0.01)
        nn.init.normal_(self.item.weight, std=0.1 if layers else 0.01)
        # Graphs derive from this run's training split, never a stale on-disk cache.
        self.register_buffer("graph", graph, persistent=False)

    def encode(self):
        if not self.layers:
            return self.user.weight, self.item.weight
        out = torch.cat([self.user.weight, self.item.weight])
        total = out
        for _ in range(self.layers):
            out = torch.sparse.mm(self.graph, out)
            total = total + out
        total = total / (self.layers + 1)
        return total[:self.n_users], total[self.n_users:]


def dot(embedding, users, items):
    return (embedding[0][users] * embedding[1][items]).sum(dim=-1)


def bernoulli_kl(teacher, student, temperature=1.0):
    """KL(Bernoulli(teacher)||Bernoulli(student)), stable in logit space."""
    t = teacher.detach() / temperature
    s = student / temperature
    p = torch.sigmoid(t)
    return (p * (F.logsigmoid(t) - F.logsigmoid(s)) +
            (1 - p) * (F.logsigmoid(-t) - F.logsigmoid(-s))) * temperature**2


def denoising_nll(student, h1, h2, observed, phase, c1, c2):
    """Negative expectation, preserving the DP/DN formulas in loss_func.py.

    DP updates h2; DN updates h1. The source subtracts C2*p in DP,
    although the printed Eq.17/18 in the paper has a plus sign there.
    We follow the executable source and the negative-log interpretation.
    """
    p = torch.sigmoid(student)
    if phase == "DN":
        positive = -F.logsigmoid(h1) * p + c1 * (1 - p)
        negative = -F.logsigmoid(-h1) * p
    else:
        positive = -F.logsigmoid(h2) * (1 - p)
        negative = -F.logsigmoid(-h2) * (1 - p) + c2 * p
    return torch.where(observed.bool(), positive, negative)


def batch_regularizer(model, users, positive, negative):
    return (model.user(users).pow(2).sum() + model.item(positive).pow(2).sum() +
            model.item(negative).pow(2).sum()) / (2.0 * len(users))


def full_regularizer(model):
    # Same scale as the repository, needed for the source-style control.
    return sum(p.pow(2).sum() for p in model.parameters()) / (2.0 * model.n_users)


def group_items(pairs):
    groups = {}
    for user, item in pairs:
        groups.setdefault(int(user), []).append(int(item))
    return groups


def metrics_at_k(predictions, truth, ks):
    """Return per-user metrics; macro averaging is done by evaluate()."""
    output = {}
    for k in ks:
        discount = 1.0 / np.log2(np.arange(2, k + 2))
        recall, ndcg = [], []
        for row, relevant in zip(predictions, truth):
            hits = np.isin(row[:k], relevant).astype(float)
            recall.append(float(hits.sum() / len(relevant)))
            ndcg.append(float((hits * discount).sum() / discount[:min(k, len(relevant))].sum()))
        output[f"Recall@{k}"] = np.array(recall)
        output[f"NDCG@{k}"] = np.array(ndcg)
    return output


@torch.no_grad()
def evaluate(model, truth_pairs, exclude, ks=(10, 20), user_batch=256, item_chunk=8192,
             teacher=None, beta=1.0, user_limit=0, beta_by_user=None):
    """Exact full-catalog top-k with bounded score memory and cached encodings."""
    model.eval()
    embedding = model.encode()
    teacher_embedding = None
    if teacher is not None:
        teacher.eval()
        teacher_embedding = teacher.encode()
    truth, seen = group_items(truth_pairs), group_items(exclude)
    users = np.array(sorted(truth), dtype=np.int64)
    if user_limit:
        users = np.random.default_rng(0).permutation(users)[:user_limit]
        users.sort()
    if len(users) == 0:
        raise ValueError("No evaluable users.")
    max_k = max(ks)
    if model.n_items < max_k:
        raise ValueError("top_k exceeds the item catalog.")
    if any(model.n_items - len(seen.get(int(u), [])) < max_k for u in users):
        raise ValueError("Fewer than top_k unmasked candidates for an evaluated user.")
    device = embedding[0].device
    collected = {f"{metric}@{k}": [] for k in ks for metric in ("Recall", "NDCG")}
    for start in range(0, len(users), user_batch):
        batch_users = users[start:start + user_batch]
        uid = torch.as_tensor(batch_users, dtype=torch.long, device=device)
        scores_best = torch.full((len(uid), max_k), -float("inf"), device=device)
        items_best = torch.full((len(uid), max_k), -1, device=device, dtype=torch.long)
        for item_start in range(0, model.n_items, item_chunk):
            item_end = min(item_start + item_chunk, model.n_items)
            score = torch.sigmoid(embedding[0][uid] @ embedding[1][item_start:item_end].T)
            if teacher_embedding is not None:
                b = beta if beta_by_user is None else torch.as_tensor(beta_by_user[batch_users], device=device)[:, None]
                score = b * score + (1 - b) * torch.sigmoid(
                    teacher_embedding[0][uid] @ teacher_embedding[1][item_start:item_end].T)
            mask_row, mask_col = [], []
            for row, user in enumerate(batch_users):
                for item in seen.get(int(user), []):
                    if item_start <= item < item_end:
                        mask_row.append(row)
                        mask_col.append(item - item_start)
            if mask_row:
                score[mask_row, mask_col] = -float("inf")
            item_ids = torch.arange(item_start, item_end, device=device).expand(len(uid), -1)
            all_scores = torch.cat([scores_best, score], dim=1)
            all_items = torch.cat([items_best, item_ids], dim=1)
            scores_best, positions = torch.topk(all_scores, k=max_k, dim=1)
            items_best = all_items.gather(1, positions)
        metric = metrics_at_k(items_best.cpu().numpy(), [truth[int(u)] for u in batch_users], ks)
        for name, values in metric.items():
            collected[name].append(values)
    per_user = {name: np.concatenate(parts) for name, parts in collected.items()}
    return {name: float(values.mean()) for name, values in per_user.items()}, users, per_user


def tensor(array, device):
    return torch.as_tensor(array, dtype=torch.long, device=device)


def save_weights(path, model):
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, path)


def load_weights(path, model, device):
    # All checkpoints here contain tensors only, also usable with PyTorch >=2.6.
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch 1.10 in the original local environment.
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)


def append_log(path, record):
    with open(path, "a", encoding="utf8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def checked_step(loss, optimizer, parameters, clip):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss; inspect hyperparameters and data.")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient.")
    optimizer.step()


def pretrain(model, pairs, validation, observed, config, out, name, rng):
    sampler = NegativeSampler(observed, model.n_users, model.n_items, rng)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best, stale = -1.0, 0
    checkpoint = out / f"teacher_{name}.pt"
    for epoch in range(1, config.pretrain_epochs + 1):
        started = time.perf_counter()
        model.train()
        order = rng.permutation(len(pairs))
        total_loss, batches = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            batch = pairs[order[start:start + config.batch_size]]
            users, positive = tensor(batch[:, 0], config.device), tensor(batch[:, 1], config.device)
            negative = tensor(sampler.sample(batch[:, 0])[:, 0], config.device)
            optimizer.zero_grad(set_to_none=True)
            embedding = model.encode()  # One propagation per optimizer step.
            loss = F.softplus(dot(embedding, users, negative) - dot(embedding, users, positive)).mean()
            loss = loss + config.reg * full_regularizer(model)
            checked_step(loss, optimizer, list(model.parameters()), config.grad_clip)
            total_loss += float(loss.detach())
            batches += 1
            if config.max_steps and batches >= config.max_steps:
                break
        record = dict(stage=f"pretrain_{name}", epoch=epoch, loss=total_loss / batches,
                      seconds=time.perf_counter() - started)
        if epoch % config.eval_every == 0 or epoch == config.pretrain_epochs or epoch == 1:
            metric, _, _ = evaluate(model, validation, observed, config.top_k, config.eval_batch_size,
                                    config.item_chunk, user_limit=config.eval_user_limit)
            record.update(validation=metric)
            value = metric[config.select_metric]
            if value > best + config.min_delta:
                best, stale = value, 0
                save_weights(checkpoint, model)
            else:
                stale += 1
        append_log(out / "history.jsonl", record)
        print(json.dumps(record), flush=True)
        if stale >= config.patience:
            break
    load_weights(checkpoint, model, config.device)
    model.eval()
    model.requires_grad_(False)
    return model


def adaptive_click_weight(pv_logits, buy_logits, users, reliability, tau, floor):
    # Agreement of scores is a heuristic, not calibrated probability of correctness.
    agreement = torch.exp(-torch.abs(torch.sigmoid(pv_logits) - torch.sigmoid(buy_logits)) / tau)
    return floor + (1 - floor) * reliability[users] * agreement


def train_target(target, teachers, data, config, out, rng):
    # h1/h2 stay MF, matching the original default --h_model MF.
    heads = nn.ModuleDict({f"{behavior}_{index}": Recommender(data["n_users"], data["n_items"], config.dim)
                           for behavior in ("buy", "pv") for index in (1, 2)}).to(config.device)
    trainable = list(target.parameters()) + list(heads.parameters())
    optimizer = torch.optim.Adam(trainable, lr=config.lr)
    with torch.no_grad():
        teacher_emb = {name: tuple(x.detach() for x in model.encode()) for name, model in teachers.items()}
    reliability = torch.as_tensor(data["reliability"], device=config.device)
    sampler = NegativeSampler(data["pv_fit"], data["n_users"], data["n_items"], rng, config.popularity_mix)
    beta_by_user = None
    if config.variant == "adaptive" and config.fusion_strength:
        beta_by_user = np.clip(config.beta + config.fusion_strength * (data["reliability"] - 0.5), 0.5, 1.0)
    best, stale, step, best_epoch = -1.0, 0, 0, 0
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        target.train()
        heads.train()
        click = data["clicked_only"]
        if config.click_ratio >= 0:
            limit = min(len(click), int(config.click_ratio * len(data["buy_fit"])))
            click = click[rng.choice(len(click), size=limit, replace=False)]
        totals = dict(loss=0.0, kl=0.0, denoise=0.0, bpr=0.0)
        batches = 0
        for source, pairs in (("buy", data["buy_fit"]), ("pv", click)):
            order = rng.permutation(len(pairs))
            for start in range(0, len(order), config.batch_size):
                batch = pairs[order[start:start + config.batch_size]]
                users, pos = tensor(batch[:, 0], config.device), tensor(batch[:, 1], config.device)
                count = config.negative_candidates if source == "buy" else 1
                candidates = tensor(sampler.sample(batch[:, 0], count), config.device)
                optimizer.zero_grad(set_to_none=True)
                embedding = target.encode()
                if count > 1:
                    with torch.no_grad():
                        candidate_score = (embedding[0][users, None] * embedding[1][candidates]).sum(-1)
                        # Rank two among a small pool; reduces extreme-hard-negative risk.
                        choice = candidate_score.topk(min(2, count), dim=1).indices[:, -1]
                        negative = candidates.gather(1, choice[:, None]).squeeze(1)
                else:
                    negative = candidates[:, 0]
                all_users, items = torch.cat([users, users]), torch.cat([pos, negative])
                logits = dot(embedding, all_users, items)
                pv_logits = dot(teacher_emb["pv"], all_users, items)
                buy_logits = dot(teacher_emb["buy"], all_users, items)
                phase = "DP" if step % 2 == 0 else "DN"
                observed_pv = torch.cat([torch.ones_like(users), torch.zeros_like(users)])
                observed_buy = observed_pv if source == "buy" else torch.zeros_like(all_users)
                nll = torch.zeros_like(logits)
                for behavior, labels in (("buy", observed_buy), ("pv", observed_pv)):
                    # Only the active noise model is evaluated and gets gradients.
                    active = heads[f"{behavior}_{1 if phase == 'DN' else 2}"]
                    h = dot(active.encode(), all_users, items)
                    nll += denoising_nll(logits, h, h, labels, phase, config.c1, config.c2)
                kl_buy = bernoulli_kl(buy_logits, logits, config.temperature)
                kl_pv = bernoulli_kl(pv_logits, logits, config.temperature)
                if config.variant == "adaptive" and config.adaptive_kl:
                    weight = adaptive_click_weight(pv_logits, buy_logits, all_users, reliability,
                                                   config.agreement_tau, config.weight_floor)
                    kl_pv = kl_pv * weight
                ramp = min(1.0, epoch / max(1, config.warmup_epochs)) if config.variant == "adaptive" else 1.0
                # The repository sums means for positive and negative samples separately.
                denoise = 2.0 * nll.mean()
                kl = 2.0 * config.alpha * ramp * (kl_buy + kl_pv).mean()
                bpr = logits.new_zeros(())
                if source == "buy" and config.bpr_weight:
                    confidence = torch.ones_like(logits[:len(users)])
                    if config.confidence_bpr:
                        margin = buy_logits[:len(users)] - buy_logits[len(users):]
                        confidence = config.weight_floor + (1 - config.weight_floor) * torch.sigmoid(margin)
                    bpr = (confidence * F.softplus(logits[len(users):] - logits[:len(users)])).mean()
                if config.variant == "mba":
                    reg = full_regularizer(target)
                else:
                    # Regularize sampled target embeddings and active denoisers.
                    reg = batch_regularizer(target, users, pos, negative)
                    for behavior in ("buy", "pv"):
                        reg += batch_regularizer(heads[f"{behavior}_{1 if phase == 'DN' else 2}"], users, pos, negative)
                loss = denoise + kl + config.bpr_weight * bpr + config.target_reg * reg
                checked_step(loss, optimizer, trainable, config.grad_clip)
                for name, value in (("loss", loss), ("kl", kl), ("denoise", denoise), ("bpr", bpr)):
                    totals[name] += float(value.detach())
                step += 1
                batches += 1
                if config.max_steps and batches >= config.max_steps:
                    break
            if config.max_steps and batches >= config.max_steps:
                break
        record = dict(stage=config.variant, epoch=epoch, **{k: v / batches for k, v in totals.items()},
                      click_pairs=len(click), seconds=time.perf_counter() - started)
        if epoch % config.eval_every == 0 or epoch == 1 or epoch == config.epochs:
            metric, _, _ = evaluate(target, data["buy_val"], data["buy_fit"], config.top_k,
                                    config.eval_batch_size, config.item_chunk, teachers["buy"], config.beta,
                                    config.eval_user_limit, beta_by_user)
            value = metric[config.select_metric]
            record["validation"] = metric
            if value > best + config.min_delta:
                best, stale, best_epoch = value, 0, epoch
                save_weights(out / "target_best.pt", target)
            else:
                stale += 1
        # Inference snapshot of the latest epoch, plus the best selected on validation.
        save_weights(out / "target_last.pt", target)
        append_log(out / "history.jsonl", record)
        print(json.dumps(record), flush=True)
        if stale >= config.patience:
            break
    load_weights(out / "target_best.pt", target, config.device)
    return target, beta_by_user, best_epoch, best


def resolve_config(config):
    if config.run_label and not re.fullmatch(r"[A-Za-z0-9_-]+", config.run_label):
        raise ValueError("run_label may contain letters, numbers, underscores and hyphens only.")
    preset = PRESETS[(config.dataset, config.backbone)]
    for name, value in preset.items():
        if getattr(config, name) is None:
            setattr(config, name, value)
    if config.target_reg is None:
        config.target_reg = config.reg
    adaptive = config.variant == "adaptive"
    if config.bpr_weight is None:
        config.bpr_weight = 1.0 if adaptive else 0.0
    if config.negative_candidates is None:
        config.negative_candidates = 4 if adaptive else 1
    if config.click_ratio is None:
        config.click_ratio = (1.0 if config.dataset == "beibei" else 2.0) if adaptive else -1.0
    if config.popularity_mix is None:
        config.popularity_mix = 0.2 if adaptive else 0.0
    if config.fusion_strength is None:
        config.fusion_strength = 0.1 if adaptive else 0.0
    config.device = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Enable a Kaggle GPU or use --device cpu.")
    if config.cpu_threads:
        torch.set_num_threads(config.cpu_threads)
    if not 0 < config.val_fraction < 1 or not 0 <= config.popularity_mix <= 1:
        raise ValueError("Invalid validation fraction or popularity mixture.")
    if config.temperature <= 0 or config.agreement_tau <= 0 or not 0 <= config.weight_floor <= 1:
        raise ValueError("Invalid temperature, agreement_tau, or weight_floor.")
    if not 0 <= config.beta <= 1 or config.fusion_strength < 0:
        raise ValueError("Invalid fusion parameters.")
    if any(getattr(config, x) < 1 for x in ("epochs", "pretrain_epochs", "batch_size", "eval_every",
                                          "patience", "eval_batch_size", "item_chunk", "negative_candidates", "dim")):
        raise ValueError("Epoch counts, dimensions, batch sizes and patience must be positive.")
    if config.alpha < 0 or min(config.reg, config.target_reg, config.bpr_weight, config.c1, config.c2) < 0:
        raise ValueError("Loss weights must be nonnegative.")
    if config.grad_clip <= 0 or config.lr <= 0 or config.layers < 0:
        raise ValueError("Invalid learning rate, clipping threshold or layer count.")
    if min(config.max_steps, config.eval_user_limit, config.cpu_threads, config.warmup_epochs, config.min_delta) < 0:
        raise ValueError("Debug limits, CPU thread count, warmup and min_delta must be nonnegative.")
    config.top_k = sorted(set(config.top_k))
    if min(config.top_k) < 1 or config.select_metric not in {f"{m}@{k}" for m in ("Recall", "NDCG") for k in config.top_k}:
        raise ValueError("select_metric must be one of the requested ranking metrics.")
    return config


def run(config):
    config = resolve_config(config)
    seed_all(config.seed)
    rng = np.random.default_rng(config.seed)
    method = config.run_label or config.variant
    out = Path(config.output) / config.dataset / f"{method}_{config.backbone}_seed{config.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "history.jsonl").exists() or (out / "result.json").exists():
        raise FileExistsError(f"Run already exists: {out}. Choose another --output; existing results are preserved.")
    paths = discover_data(config.data_root, config.dataset)
    data = prepare_data(paths, config.val_fraction, config.split_seed)
    provenance = {name: dict(path=str(path), sha256=file_hash(path)) for name, path in paths.items()}
    meta = dict(config=vars(config), data=data["stats"], sources=provenance,
                environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                 pandas=pd.__version__, cuda=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name(0) if config.device.startswith("cuda") else None),
                protocol="fixed purchase test; validation from train only; pv_test unused; full catalog ranking")
    (out / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    np.savez_compressed(out / "splits.npz", **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})
    print(json.dumps(meta, indent=2), flush=True)
    layers = config.layers if config.backbone == "lightgcn" else 0
    buy_graph = make_graph(data["buy_fit"], data["n_users"], data["n_items"], config.device) if layers else None
    pv_graph = make_graph(data["pv_fit"], data["n_users"], data["n_items"], config.device) if layers else None

    def build(graph):
        return Recommender(data["n_users"], data["n_items"], config.dim, layers, graph).to(config.device)

    teacher_cache = None
    if config.teacher_cache:
        names = ("dataset", "backbone", "seed", "split_seed", "val_fraction", "dim", "layers", "reg", "lr",
                 "batch_size", "pretrain_epochs", "patience", "eval_every", "min_delta", "select_metric", "top_k",
                 "eval_user_limit", "max_steps", "device", "grad_clip")
        signature = dict(implementation="mba-kaggle-v1", settings={k: getattr(config, k) for k in names},
                         data={k: v["sha256"] for k, v in provenance.items()}, environment=meta["environment"])
        fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        teacher_cache = Path(config.teacher_cache) / fingerprint
        teacher_cache.mkdir(parents=True, exist_ok=True)
    cached = teacher_cache is not None and (teacher_cache / "complete.json").is_file()
    teachers = {}
    for name, graph in (("pv", pv_graph), ("buy", buy_graph)):
        seed_all(config.seed)  # Same initialization policy across control and extension.
        if cached:
            model = build(graph)
            load_weights(teacher_cache / f"teacher_{name}.pt", model, config.device)
            model.eval().requires_grad_(False)
            teachers[name] = model
            save_weights(out / f"teacher_{name}.pt", model)
            record = dict(stage=f"pretrain_{name}", reused_cache=str(teacher_cache))
            append_log(out / "history.jsonl", record)
            print(json.dumps(record), flush=True)
        else:
            teachers[name] = pretrain(build(graph), data[f"{name}_fit"], data[f"{name}_val"], data[f"{name}_fit"],
                                     config, out, name, np.random.default_rng(config.seed))
            if teacher_cache is not None:
                shutil.copy2(out / f"teacher_{name}.pt", teacher_cache / f"teacher_{name}.pt")
    if teacher_cache is not None and not cached:
        (teacher_cache / "complete.json").write_text(json.dumps(signature, indent=2), encoding="utf8")
    seed_all(config.seed + 1)
    target = build(buy_graph)
    if config.variant == "adaptive" and config.warm_start:
        target.load_state_dict(teachers["buy"].state_dict())
    target, beta_by_user, best_epoch, best_val = train_target(target, teachers, data, config, out, rng)
    if beta_by_user is not None:
        np.save(out / "beta_by_user.npy", beta_by_user)
    # The first and only use of purchase-test labels for model evaluation.
    metric, users, per_user = evaluate(target, data["buy_test"], data["buy_train"], config.top_k,
                                      config.eval_batch_size, config.item_chunk, teachers["buy"], config.beta,
                                      config.eval_user_limit, beta_by_user)
    baseline_metric, _, _ = evaluate(teachers["buy"], data["buy_test"], data["buy_train"], config.top_k,
                                     config.eval_batch_size, config.item_chunk, user_limit=config.eval_user_limit)
    degree = np.bincount(data["buy_train"][:, 0], minlength=data["n_users"])[users]
    np.savez_compressed(out / "test_per_user.npz", users=data["user_ids"][users], buy_train_degree=degree, **per_user)
    result = dict(dataset=config.dataset, variant=config.variant, method=method, backbone=config.backbone, seed=config.seed,
                  split_seed=config.split_seed, best_epoch=best_epoch, best_validation=best_val,
                  selection_metric=config.select_metric, test=metric, purchase_teacher_test=baseline_metric,
                  debug=bool(config.max_steps or config.eval_user_limit), users_evaluated=len(users),
                  protocol=meta["protocol"])
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    pd.DataFrame([dict(dataset=config.dataset, variant=config.variant, seed=config.seed, **metric)]).to_csv(out / "metrics.csv", index=False)
    print("FINAL " + json.dumps(result), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="/kaggle/input")
    p.add_argument("--output", default="/kaggle/working/mba_runs")
    p.add_argument("--teacher-cache", help="Reuse teachers only when hashes, split, settings and environment match.")
    p.add_argument("--dataset", choices=["taobao", "beibei"], default="taobao")
    p.add_argument("--variant", choices=["mba", "adaptive"], default="adaptive")
    p.add_argument("--run-label", help="Optional name for an ablation, also used in the output folder.")
    p.add_argument("--backbone", choices=["mf", "lightgcn"], default="mf")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--pretrain-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=4, help="Number of validation checks without improvement.")
    p.add_argument("--min-delta", type=float, default=1e-6)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--top-k", type=int, nargs="+", default=[10, 20])
    p.add_argument("--select-metric", default="Recall@20")
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--item-chunk", type=int, default=8192)
    p.add_argument("--alpha", type=float)
    p.add_argument("--c1", type=float)
    p.add_argument("--c2", type=float)
    p.add_argument("--reg", type=float)
    p.add_argument("--target-reg", type=float)
    p.add_argument("--beta", type=float)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--agreement-tau", type=float, default=0.25)
    p.add_argument("--weight-floor", type=float, default=0.2)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--bpr-weight", type=float)
    p.add_argument("--negative-candidates", type=int)
    p.add_argument("--click-ratio", type=float, help="Max clicked-only edges per purchase edge per epoch; -1 means all.")
    p.add_argument("--popularity-mix", type=float)
    p.add_argument("--fusion-strength", type=float)
    p.add_argument("--no-adaptive-kl", dest="adaptive_kl", action="store_false")
    p.add_argument("--no-confidence-bpr", dest="confidence_bpr", action="store_false")
    p.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    p.add_argument("--max-steps", type=int, default=0, help="Debug only: cap steps per epoch; marks results debug.")
    p.add_argument("--eval-user-limit", type=int, default=0, help="Debug only: cap evaluation users.")
    p.add_argument("--cpu-threads", type=int, default=0)
    p.set_defaults(adaptive_kl=True, confidence_bpr=True, warm_start=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
