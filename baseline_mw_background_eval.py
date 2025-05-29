#!/usr/bin/env python3
"""
baseline_mw_eval.py

Evaluate an empirical Mann–Whitney baseline on your test set, using
the precomputed background from build_mw_background.py.

For each test protein:
  - compute its Mann-Whitney U score per function
  - empirical p = fraction(bg_scores >= observed_score)
  - predict the function with the smallest empirical p
Outputs per-protein predictions + overall precision/recall/F1 in JSON.

Usage:
  python baseline_mw_eval.py \
    --test_dataset  /scratch/users/tartici/PARSE/datasets/af2_human_proteome_combined/ \
    --test_csv      data/test_dataset.csv \
    --id_column     uniprot \
    --gt_column     description \
    --db            downloads/csa_site_db_nn.pkl \
    --func_sets     downloads/csa_function_sets_nn.pkl \
    --bg_pkl        data/bg_mannwhitneyU_distributions.pkl \
    --output        results/baseline_mw_empirical.json


for smoke test
    python baseline_mw_eval.py \
          --test_dataset /scratch/users/tartici/PARSE/datasets/af2_human_proteome_combined/ \
          --test_csv     data/test_dataset.csv \
          --id_column    uniprot \
          --gt_column    description \
          --db           downloads/csa_site_db_nn.pkl \
          --func_sets    downloads/csa_function_sets_nn.pkl \
          --bg_pkl       data/bg_mannwhitneyU_distributions.pkl \
          --limit        20 \
          --output       results/smoketest_baseline_mw_empirical.json

    
"""

import argparse, json, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from atom3d.datasets import load_dataset
from parse import compute_rank_df
from scipy.stats import mannwhitneyu
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from raw_lmdb import RawLMDB  # use this instead of load_dataset




def load_pickle(path):
    with open(path,'rb') as f:
        return pickle.load(f)


def protein_mw_scores(rank_df, function_sets):
    """
    Given a rank_df for one protein, compute the
    Mann–Whitney U score for each function.
    """
    # map site->score
    site2s = dict(zip(rank_df['site'], rank_df['score']))
    scores = {}
    for func, sites in function_sets.items():
        in_scores  = np.array([site2s[s] for s in sites if s in site2s])
        out_scores = np.array([site2s[s] for s in site2s if s not in sites])
        if len(in_scores) >= 1 and len(out_scores) >= 1:
            u_stat = mannwhitneyu(in_scores, out_scores, alternative='greater').statistic
            scores[func] = u_stat
        else:
            scores[func] = -np.inf  # or maybe np.nan if more appropriate
    return scores

def compute_mwu_u_vec(scores, func_idx):
        order = np.argsort(scores, kind='mergesort')
        ranks = np.empty_like(order, dtype=np.int32)
        ranks[order] = np.arange(1, len(scores)+1, dtype=np.int32)
        u = {}
        for f, idxs in func_idx.items():
            n1 = idxs.size
            if n1 < 1 or n1 >= len(scores):
                u[f] = -np.inf
                continue
            R1 = ranks[idxs].sum()
            u[f] = R1 - (n1 * (n1 + 1) // 2)
        return u

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_dataset', required=True,
                   help="LMDB root of test embeddings")
    p.add_argument('--test_csv',    required=True,
                   help="CSV of test ground truth")
    p.add_argument('--id_column',   default='cluster',
                   help="Column in test_csv matching rec['id']")
    p.add_argument('--gt_column',   default='EC_uniprot',
                   help="Column in test_csv holding the true function")
    p.add_argument('--db',          required=True,
                   help="Pickle of csa_site_db_nn.pkl")
    p.add_argument('--func_sets',   required=True,
                   help="Pickle of csa_function_sets_nn.pkl")
    p.add_argument('--bg_pkl',      required=True,
                   help="Pickle of bg_mannwhitney_dists.pkl")
    p.add_argument('--limit', type=int, default=None,
               help="Stop after this many proteins (for smoke test)")
    p.add_argument('--output',      required=True,
                   help="JSON file to write predictions + metrics")
    args = p.parse_args()

    # load everything
    db            = load_pickle(args.db)
    function_sets = load_pickle(args.func_sets)
    bg            = load_pickle(args.bg_pkl)   # dict: func->np.array(scores)
    test_gt       = pd.read_csv(args.test_csv)
    id2gt         = dict(zip(test_gt[args.id_column], test_gt[args.gt_column]))
    # Load test LMDB and move DB embeddings to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = RawLMDB(args.test_dataset)
    
    db_emb = torch.from_numpy(np.vstack(db['embeddings'])).to(device)  # (M,D)
    db_norm = db_emb.norm(dim=1)  # (M,)
    site_ids = [ f"{p}_{r}" for p,r in zip(db['pdbs'], db['resids']) ]
    name2idx = { site:i for i,site in enumerate(site_ids) }
    
    func_idx = {
        f: np.array([name2idx[s] for s in sites if s in name2idx], dtype=np.int32)
        for f, sites in function_sets.items()
    }
    
    



    
    records = []
    seen = 0
    for i, rec in enumerate(tqdm(ds, desc="Evaluating empirical MW")):
        
        pid = rec['id'].split('-',2)[1]
        if pid not in id2gt:
            continue
        true_f = id2gt[pid]
    
        emb = torch.tensor(rec['embeddings'], dtype=torch.float32, device=device)
        conf = torch.tensor(rec['confidence'], dtype=torch.float32, device=device)
        emb = emb[conf >= 70]
        if emb.shape[0] == 0:
            continue
    
        a_norm = emb.norm(dim=1, keepdim=True)
        cosims = (emb @ db_emb.T) / (a_norm * db_norm[None, :]).clamp(min=1e-8)
        site_scores = cosims.max(dim=0).values.cpu().numpy()
    
        u_scores = compute_mwu_u_vec(site_scores, func_idx)
    
        # empirical p-values
        epvs = {}
        for func, obs in u_scores.items():
            dist = bg.get(func, None)
            epv = 1.0 if dist is None or len(dist)==0 or obs==-np.inf else np.sum(dist >= obs) / len(dist)
            epvs[func] = epv
    
        pred_f  = min(epvs, key=epvs.get)
        pred_epv= epvs[pred_f]
        """records.append({
            'id': pid, 'ground_truth': true_f,
            'prediction': pred_f, 'emp_pval': pred_epv,
            'raw_score': u_scores[pred_f]
        })"""
        records.append({
            'id':          str(pid),
            'ground_truth': str(true_f),
            'prediction':  str(pred_f),
            'emp_pval':    float(pred_epv),
            'raw_score':   float(u_scores[pred_f])
        })


        seen += 1
        if args.limit and seen >= args.limit:
            break


    # assemble and compute metrics
    df = pd.DataFrame(records)
    y_true = df['ground_truth']
    y_pred = df['prediction']
    metrics = {
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall':    recall_score   (y_true, y_pred, average='macro'),
        'f1':        f1_score       (y_true, y_pred, average='macro'),
        'n_proteins': len(df)
    }

    # dump JSON
    out = {
        'baseline':   'mannwhitney_empirical',
        'metrics':    metrics,
        'predictions':records
    }
    with open(args.output,'w') as fh:
        json.dump(out, fh, indent=2)

    print(f"\nDone. Wrote results → {args.output}")
    print("Metrics:", metrics)

if __name__=='__main__':
    main()



