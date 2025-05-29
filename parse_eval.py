#!/usr/bin/env python3
"""
parse_eval.py   ·  single-process version with checkpointing
For each test protein:
  1. Build its rank-ordered site-score DataFrame (compute_rank_df)
  2. Run parse.parse() to obtain significant functions (empirical FDR < --cutoff)
  3. Choose the top-scoring significant function as the prediction
     • If none pass the cutoff, skip the protein (no prediction)

Outputs:
  - Per-protein predictions
  - Overall macro-precision / macro-recall / macro-F1
  - Saved as a single JSON file

Usage example (full test set)
-----------------------------
python parse_eval.py \
  --test_dataset  /scratch/users/tartici/PARSE/datasets/af2_human_proteome_combined/ \
  --test_csv      data/test_dataset.csv \
  --id_column     uniprot \
  --gt_column     description \
  --db            downloads/csa_site_db_nn.pkl \
  --func_sets     downloads/csa_function_sets_nn.pkl \
  --background    downloads/function_score_dists.pkl \
  --cutoff        0.001 \
  --output        results/parse_eval.json
  
"""

import argparse, json, pickle, os
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import torch

from raw_lmdb import RawLMDB
from parse     import compute_rank_df, parse as run_parse


# ----------------------------- helpers -------------------------------------- #
def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)

def checkpoint_path(base_out: str, n: int) -> str:
    """foo.json  →  foo.json.ckpt<n>.jsonl"""
    return f"{base_out}.ckpt{n}.jsonl"

# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", required=True)
    parser.add_argument("--test_csv",    required=True)
    parser.add_argument("--id_column",   default="cluster")
    parser.add_argument("--gt_column",   default="EC_uniprot")
    parser.add_argument("--db",          required=True)
    parser.add_argument("--func_sets",   required=True)
    parser.add_argument("--background",  required=True)
    parser.add_argument("--cutoff",      type=float, default=0.001)
    parser.add_argument("--limit",       type=int,   default=None)
    parser.add_argument("--checkpoint_every", type=int, default=0,
                        help="Write checkpoint every N predictions (0 = off)")
    parser.add_argument("--output",      required=True)
    args = parser.parse_args()

    # -------------------- load data ----------------------------------------- #
    db         = load_pickle(args.db)
    func_sets  = load_pickle(args.func_sets)
    bg_dists   = load_pickle(args.background)
    gt_df      = pd.read_csv(args.test_csv)
    id2gt      = dict(zip(gt_df[args.id_column], gt_df[args.gt_column]))

    ds = RawLMDB(args.test_dataset)
    if args.limit:
        ds = ds[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- main loop ----------------------------------------- #
    records, skipped = [], 0

    for rec in tqdm(ds, desc="Running PARSE"):
        pid = rec["id"].split("-", 2)[1]
        if pid not in id2gt:
            skipped += 1
            continue

        try:
            rank_df = compute_rank_df(rec, db)
            res_df  = run_parse(rank_df, func_sets, bg_dists, cutoff=args.cutoff)
            if res_df.empty:
                skipped += 1
                continue

            top = res_df.iloc[0]
            records.append(
                dict(id=pid,
                     ground_truth=id2gt[pid],
                     prediction=str(top["function"]),
                     emp_FDR=float(top["empirical_FDR"]),
                     score=float(top["score"]))
            )

            # -------------- checkpoint ---------------------------------- #
            if args.checkpoint_every and len(records) % args.checkpoint_every == 0:
                ck_file = checkpoint_path(args.output, len(records))
                pd.DataFrame(records).to_json(ck_file,
                                              orient="records",
                                              lines=True)
                tqdm.write(f"💾 checkpoint → {ck_file}")

            if args.limit and len(records) >= args.limit:
                break

        except Exception as e:
            tqdm.write(f"[{pid}] ⚠️  {e}")
            skipped += 1
            continue

    # -------------------- metrics + final save ------------------------------ #
    if not records:
        print("No predictions generated – check cutoff / inputs.")
        return

    df      = pd.DataFrame(records)
    y_true  = df["ground_truth"]
    y_pred  = df["prediction"]
    metrics = dict(
        precision   = precision_score(y_true, y_pred, average="macro"),
        recall      = recall_score   (y_true, y_pred, average="macro"),
        f1          = f1_score       (y_true, y_pred, average="macro"),
        n_proteins  = len(df),
        n_skipped   = skipped,
    )

    out = dict(method="PARSE",
               parameters=dict(cutoff=args.cutoff),
               metrics=metrics,
               predictions=records)

    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n✅  Finished. Results → {args.output}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
