#!/usr/bin/env python3
"""
merge_lmdb_shards.py

Merge a set of LMDB “shards” (tmp_1, tmp_2, …) into a single LMDB
where each record is keyed by its internal `rec['id']` string.  This
avoids the 0–N key collisions you saw when you simply concatenated
numeric keys.

Usage:
    python merge_lmdb_shards.py \
      "/scratch/users/tartici/PARSE/datasets/af2_human_proteome_full_fixed/tmp_*" \
      /scratch/users/tartici/PARSE/datasets/af2_human_proteome_combined
"""
import argparse
import glob
import gzip
import lmdb
import os
import pickle
import sys

META_KEYS = {b'id_to_idx', b'idx_to_id', b'num_examples', b'serialization_format'}

def is_gzip(buf: bytes) -> bool:
    return buf.startswith(b'\x1f\x8b')

def main():
    p = argparse.ArgumentParser(
        description="Merge LMDB shard directories into one LMDB keyed by record['id']")
    p.add_argument('shard_pattern',
                   help="Glob pattern matching your tmp_* shard dirs, e.g. '/…/tmp_*'")
    p.add_argument('combined_dir',
                   help="Empty (or non‐existent) directory to write the merged LMDB into")
    args = p.parse_args()

    # Expand the glob into a sorted list of shard dirs
    shard_dirs = sorted(glob.glob(args.shard_pattern))
    if not shard_dirs:
        print("❌ No shards found for pattern:", args.shard_pattern, file=sys.stderr)
        sys.exit(1)

    # Make sure combined_dir exists and is empty
    os.makedirs(args.combined_dir, exist_ok=True)
    # lmdb.open will create data.mdb, lock.mdb inside that dir
    env = lmdb.open(args.combined_dir, map_size=2**40, subdir=True,
                    readahead=False, writemap=True, sync=False)

    with env.begin(write=True) as dst_txn:
        for shard in shard_dirs:
            print(f"Merging shard {shard!r} …", flush=True)
            src_env = lmdb.open(shard, readonly=True, lock=False, subdir=True)
            with src_env.begin() as src_txn:
                cursor = src_txn.cursor()
                for key, val in cursor:
                    if key in META_KEYS:
                        continue
                    # decompress & unpickle
                    raw = gzip.decompress(val) if is_gzip(val) else val
                    rec = pickle.loads(raw)
                    rec_id = rec.get('id')
                    if rec_id is None:
                        # skip anything without an 'id'
                        continue
                    # ensure it's bytes:
                    k = rec_id.encode('utf-8') if isinstance(rec_id, str) else rec_id

                    # re‐serialize the entire record
                    new_raw = pickle.dumps(rec, protocol=4)
                    new_val = gzip.compress(new_raw)

                    # put into combined LMDB under the record‐ID key
                    dst_txn.put(k, new_val)
            src_env.close()

    env.close()
    print("✅ Finished merging into:", args.combined_dir)

if __name__ == '__main__':
    main()
