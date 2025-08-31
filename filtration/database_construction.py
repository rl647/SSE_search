#%%
import itertools
import string
from collections import defaultdict
import numpy as np
from collections import Counter
import multiprocessing as mp
import os 
import sys
import pickle
import gc 
import rocksdb
import psutil


#%%
# 1) SSE alphabet and maps
alphabet = list(string.ascii_lowercase + string.ascii_uppercase)
lu = {ch: i for i, ch in enumerate(alphabet)}
ul = {i: ch for i, ch in enumerate(alphabet)}
directions = np.arange(-3,3)

# 2) Generate all ±2‐similar variants of a k‐mer (same‐case enforced)
def similar_kmers(kmer, directions=directions):
    sims = set()
    L = len(kmer)
    for offs in itertools.product(directions, repeat=L):
        cand = []
        ok = True
        for ch, off in zip(kmer, offs):
            idx = lu[ch] + off
            if idx < 0 or idx >= len(alphabet):
                ok = False
                break
            new = ul[idx]
            # enforce same case
            if (ch.islower() ^ new.islower()) or (ch.isupper() ^ new.isupper()):
                ok = False
                break
            cand.append(new)
        if ok:
            sims.add(''.join(cand))
        # print(sims)
    return sims
# Initialize RocksDB databases
def init_rocksdb(db_path):
    opts = rocksdb.Options()
    opts.create_if_missing = True
    return rocksdb.DB(db_path, opts)



# Helper to append new entries to RocksDB
def append_to_rocksdb(db, updates):
    batch = rocksdb.WriteBatch()
    keys = list(updates.keys())
    # for key, new_items in updates.items():
    for key in keys:
        new_items = updates[key]
        key_bytes = key.encode()
        existing = db.get(key_bytes)
        if existing:
            try:
                # Deserialize list from DB and convert to set of tuples
                existing_set = set(tuple(item) for item in pickle.loads(existing))
            except TypeError:
                # Handle legacy data that might already be stored as set
                existing_set = set(pickle.loads(existing))
        else:
            existing_set = set()
            # items = list(new_items)
        items = existing_set.union(new_items)

        batch.put(key_bytes, pickle.dumps(items))
        del updates[key]
    db.write(batch)
    gc.collect()

# Monitor memory and flush buffers if needed
def check_memory(buffer_kmer, buffer_space, threshold=50):
    if psutil.virtual_memory().percent >= threshold:
        if buffer_kmer:
            append_to_rocksdb(kmer_db, buffer_kmer)
            buffer_kmer.clear()
        if buffer_space:
            append_to_rocksdb(space_db, buffer_space)
            buffer_space.clear()
        gc.collect()
        return True
    return False

# Build indexes with incremental RocksDB storage
def build_indexes_incremental(sequences, k=3):
    buffer_kmer = defaultdict(set)
    buffer_space = defaultdict(set)
    total = len(sequences)
    
    for idx, (seq_id, seq) in enumerate(sequences.items(), 1):

        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            buffer_kmer[kmer].add((seq_id, i))
            for sim in similar_kmers(kmer):
                buffer_kmer[sim].add((seq_id, i))
            
            # Process spaced indices every 2 positions
            if i <= len(seq) - 5:
                space_kmer = seq[i] + seq[i+2] + seq[i+4]
                buffer_space[space_kmer].add((seq_id, i))
                for sim in similar_kmers(space_kmer):
                    buffer_space[sim].add((seq_id, i))
        
        # Periodically check memory
        if idx % 3000 == 0 or idx == total:
            print(f"Flushed buffers at sequence {idx}/{total}")
            if check_memory(buffer_kmer, buffer_space):
                print(f"Flushed buffers at sequence {idx}/{total}")
                buffer_space.clear()
                buffer_kmer.clear()
                gc.collect()
    
    # Final flush for any remaining entries
    if buffer_kmer or buffer_space:
        append_to_rocksdb(kmer_db, buffer_kmer)
        append_to_rocksdb(space_db, buffer_space)
        print("Final flush completed")
if __name__ == '__main__':
    database_dir = sys.argv[1]
    sse_path = sys.argv[2]
    # Create DB directories if they don't exist
    os.makedirs(f'{database_dir}/kmer_db', exist_ok=True)
    os.makedirs(f'{database_dir}/space_db', exist_ok=True)
    kmer_db = init_rocksdb(f'{database_dir}/kmer_db')
    space_db = init_rocksdb(f'{database_dir}/space_db')
    sse = {}
    with open(sse_path) as f:
        for line in f:
            s=line.strip().split('\t')
            
            for line in f:
                s=line.strip().split('\t')
                sse[s[0]] = s[3]
                
    build_indexes_incremental(sse, k=3)

