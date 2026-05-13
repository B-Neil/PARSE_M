import os
import time
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm  # Make sure tqdm is installed
from collapse import process_pdb, embed_protein, initialize_model
from atom3d.datasets import load_dataset
from raw_lmdb import RawLMDB
import parse
import utils

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, default=None, help='Input PDB file')
    parser.add_argument('--pdb_dir', type=str, default=None, help='Input directory containing PDB files')
    parser.add_argument('--precomputed_id', type=str, default=None, help='ID for accessing precomputed embeddings')
    parser.add_argument('--precomputed_lmdb', type=str, default=None, help='Precomputed embeddings in LMDB format')
    parser.add_argument('--chain', type=str, default=None, help='Input PDB chain to annotate')
    parser.add_argument('--db', type=str, default='./data/csa_site_db_nn.pkl', help='Reference database embeddings')
    parser.add_argument('--function_sets', type=str, default='./data/csa_function_sets_nn.pkl', help='Reference function sets')
    parser.add_argument('--background', type=str, default='./data/function_score_dists.pkl', help='Function-specific background distributions')
    parser.add_argument('--cutoff', type=float, default=0.001, help='FDR cutoff for reporting results')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU to embed proteins')
    parser.add_argument('--out_path', type=str, default='results.pkl', help='Path to save results')
    
    args = parser.parse_args()
    
    start_total = time.time()
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    
    # Load reference data
    db = utils.deserialize(args.db)
    function_sets = utils.deserialize(args.function_sets)
    background_dists = utils.deserialize(args.background)
    
    pdb_files = []
    if args.pdb:
        pdb_files.append(args.pdb)
    elif args.pdb_dir:
        for f in os.listdir(args.pdb_dir):
            if f.endswith('.pdb') or f.endswith('.ent'):
                pdb_files.append(os.path.join(args.pdb_dir, f))

    if pdb_files:
        model = initialize_model(device=device)
        all_results = {}

        pbar = tqdm(pdb_files, desc="Annotating Proteins", unit="pdb")
        
        for pdb_path in pbar:
            fname = os.path.basename(pdb_path)
            pbar.set_description(f"Processing {fname[:20]}...") # Update bar text
            
            try:
                #GPU Embedding
                pdb_df = process_pdb(pdb_path, chain=args.chain, include_hets=False)
                embed_data = embed_protein(pdb_df, model, device, include_hets=False)
                embed_data['id'] = fname
            
                #CPU Ranking/Parsing
                rnk = parse.compute_rank_df(embed_data, db)
                results = parse.parse(rnk, function_sets, background_dists, cutoff=args.cutoff)
                
                all_results[fname] = results
            except Exception as e:
                tqdm.write(f"Error in {fname}: {e}")

        # Final save
        utils.serialize(all_results, args.out_path)
        print(f'\nDone! Processed {len(all_results)} proteins.')
        print(f'Results saved to: {args.out_path}')

    elif args.precomputed_id and args.precomputed_lmdb:
        pdb_dataset = RawLMDB(args.precomputed_lmdb)
        idx = pdb_dataset.ids_to_indices([args.precomputed_id])[0]
        embed_data = pdb_dataset[idx]
        rnk = parse.compute_rank_df(embed_data, db)
        results = parse.parse(rnk, function_sets, background_dists, cutoff=args.cutoff)
        print(results)
    
    print(f'Total time: {time.time() - start_total:.2f} seconds')
