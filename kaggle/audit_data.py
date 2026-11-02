"""Audit the provided preprocessed CSVs without loading any executable document content."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(root):
    output = {}
    for dataset in ["beibei", "taobao"]:
        paths = sorted(Path(root).rglob(f"{dataset}_*_*.csv"))
        tables = {p.stem[len(dataset) + 1:]: pd.read_csv(p, sep="\t", dtype=np.int64).iloc[:, :2].to_numpy()
                  for p in paths}
        if set(tables) != {"buy_train", "buy_test", "pv_train", "pv_test"}:
            raise ValueError(f"Expected exactly four source CSVs for {dataset}, got {paths}")
        users = int(max(a[:, 0].max() for a in tables.values()) + 1)
        items = int(max(a[:, 1].max() for a in tables.values()) + 1)
        sets = {k: set((a[:, 0] * items + a[:, 1]).tolist()) for k, a in tables.items()}
        degree = np.bincount(tables["buy_train"][:, 0], minlength=users)
        row = dict(users=users, items=items,
                   files={k: dict(rows=len(a), unique_pairs=len(sets[k])) for k, a in tables.items()},
                   sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                   buy_total=len(sets["buy_train"] | sets["buy_test"]),
                   pv_total=len(sets["pv_train"] | sets["pv_test"]),
                   train_purchase_density=len(sets["buy_train"]) / (users * items),
                   train_click_buy_ratio=len(sets["pv_train"]) / len(sets["buy_train"]),
                   buy_train_degree_quantiles=np.quantile(degree, [0, .25, .5, .75, .9, 1]).tolist(),
                   users_buy_train_le_3=int((degree <= 3).sum()),
                   users_buy_train_le_3_percent=float(100 * (degree <= 3).mean()),
                   train_test_buy_overlap=len(sets["buy_train"] & sets["buy_test"]),
                   train_test_pv_overlap=len(sets["pv_train"] & sets["pv_test"]),
                   buy_test_in_pv_train=len(sets["buy_test"] & sets["pv_train"]),
                   buy_test_in_pv_test=len(sets["buy_test"] & sets["pv_test"]),
                   source_pretrain_pv_union_size=len(sets["pv_train"] | sets["buy_train"]),
                   source_mba_pv_union_size=len(sets["pv_train"] | sets["pv_test"] | sets["buy_train"]))
        output[dataset] = row
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data")
    p.add_argument("--output", default="docs/data_audit.json")
    args = p.parse_args()
    result = audit(args.data_root)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))
