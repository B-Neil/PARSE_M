#!/usr/bin/env python3
"""
build_mw_background_fast.py

Construct an empirical Mann–Whitney background distribution for each function
using your validation set.  This version GPUs the cosine similarity and
vectorizes the U‐statistic so you can process O(10⁴) proteins in minutes.

Usage:

  # smoke‐test first 100 proteins only:
  ./build_mw_background.py \
    --val_dataset    /path/to/val_lmdb \
    --val_csv        data/val_dataset.csv \
    --id_column      uniprot \
    --gt_column      EC \
    --db             downloads/csa_site_db_nn.pkl \
    --func_sets      downloads/csa_function_sets_nn.pkl \
    --limit          100 \
    --sample_rate    1 \
    --output         results/bg_mannwhitney_dists_test.pkl

  # full run (no limit):
  ./build_mw_background.py \
    --val_dataset    /path/to/val_lmdb \
    --val_csv        data/val_dataset.csv \
    --id_column      uniprot \
    --gt_column      EC \
    --db             downloads/csa_site_db_nn.pkl \
    --func_sets      downloads/csa_function_sets_nn.pkl \
    --output         results/bg_mannwhitney_dists.pkl
"""
import argparse, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from raw_lmdb import RawLMDB
from parse import compute_rank_df  # we'll bypass this for speed
from scipy.stats import mannwhitneyu

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def compute_mwu_u_vec(scores, func_idx):
    """
    Given:
      - scores: np.ndarray, shape (M,) of site‐scores for this protein
      - func_idx: dict f -> np.ndarray([i0,i1,...]) of db‐indices belonging to f
    
    Returns:
      - u_stat: dict f -> U (raw Mann–Whitney U statistic) or -inf if invalid
    """
    M = scores.shape[0]
    # 1) get 1..M ranks
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, M+1, dtype=np.int32)

    u = {}
    for f, idxs in func_idx.items():
        n1 = idxs.size
        if n1 < 1 or n1 >= M:
            u[f] = -np.inf
            continue
        # sum of ranks of in‐class
        R1 = ranks[idxs].sum()
        # U = R1 - n1*(n1+1)/2
        u_val = R1 - (n1*(n1+1)//2)
        u[f] = u_val
    return u

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--val_dataset', required=True,
                   help="LMDB root of validation embeddings")
    p.add_argument('--val_csv', required=True,
                   help="CSV of validation ground truth")
    p.add_argument('--id_column', default='uniprot',
                   help="Column in val_csv matching rec['id']")
    p.add_argument('--gt_column', default='description',
                   help="Column in val_csv holding the true function")
    p.add_argument('--db',        required=True,
                   help="Pickle of csa_site_db_nn.pkl")
    p.add_argument('--func_sets', required=True,
                   help="Pickle of csa_function_sets_nn.pkl")
    p.add_argument('--limit',     type=int, default=None,
                   help="Stop after this many proteins")
    p.add_argument('--sample_rate', type=int, default=1,
                   help="Process only every Nth protein (1=all, 10=10%)")
    p.add_argument('--output',    required=True,
                   help="Where to write bg_mannwhitney_dists.pkl")
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
   
    for arg, value in vars(args).items():
        print(f'{arg}: {value}', flush=True)

    # 1) load CSA DB and function‐to‐site maps
    db = load_pickle(args.db)
    func_sets = load_pickle(args.func_sets)

    # flatten DB into a single tensor
    # assume db has keys: 'pdbs', 'resids', 'embeddings'
    db_emb = torch.from_numpy(np.vstack(db['embeddings'])).to(device)  # shape (M, D)
    M, D = db_emb.shape
    db_norm = db_emb.norm(dim=1)  # (M,)

    # build flat list of site‐ids in the same order
    site_ids = [ f"{pdb}_{rid}" for pdb, rid in zip(db['pdbs'], db['resids']) ]

    # precompute func → np.array of indices into 0..M
    name2idx = { site: i for i,site in enumerate(site_ids) }
    func_idx = {}
    for f, sites in func_sets.items():
        # filter out any sites not in DB (just in case)
        idxs = np.array([name2idx[s] for s in sites if s in name2idx], dtype=np.int32)
        func_idx[f] = idxs

    # 2) load ground‐truth map
    gt_df = pd.read_csv(args.val_csv)
    id2gt = dict(zip(gt_df[args.id_column], gt_df[args.gt_column]))

    # 3) open LMDB
    ds = RawLMDB(args.val_dataset)
    print("Dataset size:", len(ds))

    # 4) init background accumulator
    bg = { f: [] for f in func_sets }

    seen = 0
    total = len(ds)
    for i, rec in enumerate(tqdm(ds, desc="Building MW background")):
        if i % args.sample_rate != 0:
            continue

        # rec['id'] is like "AF-<uniprot>-model..."
        prot_id = rec['id'].split('-',2)[1]
        true_f  = id2gt.get(prot_id)
        print(f"prot_id {prot_id}")
        print(f"true_f {true_f}")
        if true_f is None:
            continue

        # 5) pull protein embeddings + normalize
        emb = torch.as_tensor(rec['embeddings'], dtype=torch.float32, device=device)   # (n_res, D)
        
        conf = torch.as_tensor(np.asarray(rec['confidence'], dtype=np.float32),
                               dtype=torch.float32, device=device)

        if args.limit is not None:
            print(f"\nemb.shape {emb.shape}", flush=True)
            print(f"conf.shape {conf.shape}", flush=True)
            print(f"conf {conf}\n", flush=True)
            
        # filter by pLDDT >= 70
        mask = conf >= 70
        emb = emb[mask]
    
        
        if emb.numel()==0:
            continue

        # 6) compute cosine similarity matrix: (n_res, M)
        #    cos(a,b) = (a·b)/(||a|| ||b||)
        a_norm = emb.norm(dim=1, keepdim=True)     # (n_res,1)
        num    = emb @ db_emb.T                   # (n_res,M)
        den    = (a_norm * db_norm[None,:]).clamp(min=1e-8)
        cosims = num / den                        # (n_res,M)

        # 7) collapse: max over residues → site‐score vector (M,)
        site_scores = cosims.max(dim=0).values    # (M,)
         

        # 8) vectorized U‐statistic for *all* funcs
        u_scores = compute_mwu_u_vec(site_scores.cpu().numpy(), func_idx)
        if args.limit is not None:
            print(f"site_scores {site_scores}", flush=True)
            print(f"u_scores {u_scores}", flush=True)

        # 9) accumulate to per‐function background (skip the true function)
        for f, u in u_scores.items():
            if f != true_f:
                bg[f].append(u)

        seen += 1
        if args.limit and seen >= args.limit:
            break

    # 10) finalize → numpy arrays
    for f in bg:
        bg[f] = np.array(bg[f], dtype=float)

    # 11) dump
    save_pickle(bg, args.output)
    print(f"Saved background distributions for {len(bg)} functions → {args.output}")

if __name__=="__main__":
    main()
