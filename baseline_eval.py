#!/usr/bin/env python3
"""
baseline_eval.py

Evaluate simple COLLAPSE-only baselines against PARSE’s GSEA on your validation
and test splits. Supports max-score, mean_top_k, and Mann-Whitney U.
Outputs per-protein predictions plus overall precision/recall/F1 into a single JSON.
"""

import argparse, json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from raw_lmdb import RawLMDB
from parse import compute_rank_df
import utils
import math
from scipy.stats import mannwhitneyu
from sklearn.metrics import precision_score, recall_score, f1_score
import multiprocessing as mp

import os 
os.environ["OMP_NUM_THREADS"] = "1" # OpenMP (NumPy/Scipy) 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def load_pickle(path):
    with open(path,'rb') as f:
        return pickle.load(f)

def collapse_max(rank_df, function_sets, **kw):
    """Baseline = max site-score per function."""
    site2score = dict(zip(rank_df.site, rank_df.score))
    out = {}
    for func, sites in function_sets.items():
        scores = [ site2score[s] for s in sites if s in site2score ]
        out[func] = max(scores) if scores else -np.inf
    return out

def collapse_mean_top_pct(rank_df, function_sets, pct=0.1, **kw):
    """
    Baseline = mean of the top PERCENT of site-scores per function.
    pct: float in (0,1], fraction of the class to average; default 0.1 (10%).
    At least one site is always selected.
    """
    site2score = dict(zip(rank_df.site, rank_df.score))
    out = {}
    for func, sites in function_sets.items():
        # collect scores for this function
        scores = [site2score[s] for s in sites if s in site2score]
        if not scores:
            out[func] = -np.inf
            continue

        # sort descending and pick top ceil(pct * N)
        scores.sort(reverse=True)
        N = len(scores)
        k = max(1, math.ceil(pct * N))
        top_scores = scores[:k]

        out[func] = float(np.mean(top_scores))
    return out


def _parallel_mwu_task(args):
    func, sites, site2score = args
    in_scores = np.array([site2score[s] for s in sites if s in site2score])
    out_scores = np.array([site2score[s] for s in site2score if s not in sites])
    if len(in_scores) >= 1 and len(out_scores) >= 1:
        p = mannwhitneyu(in_scores, out_scores, alternative='greater').pvalue
        return func, float(-np.log10(p + 1e-300))
    else:
        return func, float('-inf')

def collapse_mannwhitney(rank_df, function_sets, **kwargs):
    site2score = dict(zip(rank_df.site, rank_df.score))
    args_list = [(func, sites, site2score) for func, sites in function_sets.items()]

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    pool = mp.get_context("fork").Pool(n_workers)
    results = pool.map(_parallel_mwu_task, args_list)

    return dict(results)

BASELINES = {
    'max':           collapse_max,
    'mean_top_pct':  collapse_mean_top_pct,  
    'mannwhitney':   collapse_mannwhitney
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', required=True)
    p.add_argument('--func_sets', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--gt_csv', required=True)
    p.add_argument('--id_column', default='pdb_id')
    p.add_argument('--gt_column', default='description')
    p.add_argument('--baseline', choices=BASELINES.keys(), default='max')
    p.add_argument('--pct', type=float, default=0.1)
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint_every', type=int, default=100,
                   help="Save intermediate results every N proteins")
    args = p.parse_args()

    verbose = True

    if verbose:
        for arg, value in vars(args).items():
            print(f'{arg}: {value}', flush=True)

    # 1) Load everything
    db            = load_pickle(args.db)
    if verbose:
        print(f"Loaded the database {args.db}")
    function_sets = load_pickle(args.func_sets)
    if verbose:
        print(f"Loaded the func_sets {args.func_sets}")
    gt_df = pd.read_csv(args.gt_csv)
    if verbose:
        print(f"Loaded the ground truth df {args.gt_csv}")
    # build mapping ID -> true label
    id2gt = dict(zip(gt_df[args.id_column], gt_df[args.gt_column]))
    #lmdb_ds       = load_dataset(args.dataset, 'lmdb')
    lmdb_ds = RawLMDB(args.dataset)
    if verbose:
        print(f"Loaded the lmdb dataset {args.dataset}")

    collapse_fn = BASELINES[args.baseline]

    # After you do: lmdb_ds = load_dataset(args.dataset, 'lmdb')
    ids = lmdb_ds.ids  # all record IDs


    n = len(lmdb_ds)
    matched = 0
    records = []
    for idx in tqdm(range(n), desc="Scoring proteins"):
        try:
            rec = lmdb_ds[idx]
            raw_id = rec['id'] 
            prot_id = raw_id.split('-', 2)[1]
        except:
            continue
        if prot_id not in id2gt:
            continue

        

        try:
            rank_df = compute_rank_df(rec, db)
            func_scores = collapse_fn(rank_df, function_sets, pct=args.pct)
            pred_func  = max(func_scores, key=func_scores.get)
            pred_score = func_scores[pred_func]
            records.append({
                'id': prot_id,
                'ground_truth': id2gt[prot_id],
                'prediction': pred_func,
                'score': pred_score
            })
        except Exception as e:
            print(f"Raised the following exception but continuing still: {e}", flush=True)
            continue

        matched += 1
        tqdm.write(f"Matched {matched:5d}: {prot_id} → {id2gt[prot_id]}", end='\r')

        if args.checkpoint_every and matched % args.checkpoint_every == 0:
            cp_path = f"{args.output}_checkpoint_{matched}.json"
            with open(cp_path, 'w') as cp_fh:
                json.dump({'partial_predictions': records}, cp_fh, indent=2)
            print(f"\n✅ Saved checkpoint to {cp_path}")

    out_df = pd.DataFrame(records)
    y_true = out_df['ground_truth']
    y_pred = out_df['prediction']
    metrics = {
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'n_proteins': len(out_df)
    }

    result = {
        'baseline': args.baseline,
        'parameters': {'pct': args.pct},
        'metrics': metrics,
        'predictions': records
    }
    with open(f"{args.output}_baseline_{args.baseline}.json", 'w') as fh:
        json.dump(result, fh, indent=2)

    print(f"\nDone. Wrote predictions+metrics for '{args.baseline}' → {args.output}")
    print("Metrics:", metrics)

if __name__=='__main__':
    main()


"""
python baseline_eval.py \
  --db         downloads/csa_site_db_nn.pkl \
  --func_sets  downloads/csa_function_sets_nn.pkl \
  --dataset    /scratch/users/tartici/PARSE/datasets/af2_human_proteome_combined/ \
  --gt_csv     data/test_dataset.csv \
  --id_column  uniprot \
  --gt_column  description \
  --baseline   mean_top_pct \
  --pct        0.10 \
  --output     results/baseline_pct10
"""
