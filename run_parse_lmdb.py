import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from atom3d.datasets import load_dataset
from collapse.utils import pdb_from_fname
import argparse
import parse
import utils

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./data/datasets/af2_dark_proteome/full')
    parser.add_argument('--db', type=str, default='./data/csa_site_db_nn.pkl', help='Reference database embeddings')
    parser.add_argument('--function_sets', type=str, default='./data/csa_function_sets_nn.pkl', help='Reference function sets')
    parser.add_argument('--background', type=str, default='./data/function_score_dists.pkl', help='Function-specific background distributions')
    parser.add_argument('--cutoff', type=float, default=0.001, help='FDR cutoff for reporting results')
    parser.add_argument('--out_path', type=str, default='./data/results/parse_af2_dark.tsv', help='Output path for results (tsv)')
    parser.add_argument('--split_id', type=int, default=0)
    parser.add_argument('--num_splits', type=int, default=1)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, 'lmdb')

    db = utils.deserialize(args.db)
    function_sets = utils.deserialize(args.function_sets)
    background_dists = utils.deserialize(args.background)

    if args.num_splits > 1:
        # Replaced .pkl with .tsv for split logic
        out_path = args.out_path.replace('.tsv', f'_{args.split_id}.tsv')

        split_idx = np.array_split(np.arange(len(dataset)), args.num_splits)[args.split_id - 1]
        print(f'Processing split {args.split_id} with {len(split_idx)} examples...')

        dataset = torch.utils.data.Subset(dataset, split_idx)
    else:
        out_path = args.out_path
        print(f'Processing full dataset with {len(dataset)} examples...')

    results = {}
    for pdb_data in tqdm(dataset):
        protein = pdb_data['id']
        rnk = parse.compute_rank_df(pdb_data, db)
        result = parse.parse(rnk, function_sets, background_dists, args.cutoff)
        results[protein] = result.copy()
        
    # --- NEW TSV EXPORT LOGIC ---
    print(f"Converting results to TSV...")
    try:
        # Check if the result for each protein is a DataFrame
        if isinstance(list(results.values())[0], pd.DataFrame):
            df_final = pd.concat(results, names=['protein_id', 'original_index'])
        else:
            # If the result is a standard dictionary of scores/functions
            df_final = pd.DataFrame.from_dict(results, orient='index')
            df_final.index.name = 'protein_id'
            
        df_final.to_csv(out_path, sep='\t')
        print(f"Successfully saved TSV to {out_path}")
        
    except Exception as e:
        print(f"Error converting to TSV: {e}")
        fallback_path = out_path.replace('.tsv', '.pkl')
        print(f"Saving as pickle instead to {fallback_path}...")
        utils.serialize(results, fallback_path)
