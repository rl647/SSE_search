#%%
import argparse
import pickle
import multiprocessing as mp
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import rocksdb


def create_arg_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Find candidate sequences from RocksDB using k-mer and spaced-seed searches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Paths ---
    parser.add_argument("db_path", type=Path, help="Path to the main database directory containing 'kmer_db' and 'space_db'.")
    parser.add_argument("query_path", type=Path, help="Path to the input query SSE file.")
    parser.add_argument("output_path", type=Path, help="Path for the output file containing candidate pairs.")

    # --- Search Parameters ---
    parser.add_argument("--kmer_size", type=int, default=3, help="Size of k-mers for sliding window search.")
    parser.add_argument("--min_diag_hits", type=int, default=3, help="Minimum diagonal hits required for a sequence to be considered in the first pass.")
    parser.add_argument("--min_total_score", type=int, default=5, help="Minimum total score for a sequence to be a final candidate.")
    parser.add_argument("--min_query_len", type=int, default=5, help="Minimum length of a query sequence to be processed.")

    # --- System Parameters ---
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of worker processes to use.")
    return parser


def load_queries(filepath: Path, min_len: int) -> Dict[str, str]:
    """
    Loads query sequences from a file.

    Args:
        filepath: Path to the tab-separated query file (id\tsequence).
        min_len: The minimum sequence length to be included.

    Returns:
        A dictionary mapping sequence IDs to sequences.
    """
    queries = {}
    print(f"Loading queries from: {filepath}")
    try:
        with filepath.open('r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2 and len(parts[1]) >= min_len:
                    queries[parts[0]] = parts[1]
    except FileNotFoundError:
        print(f"Error: Query file not found at {filepath}")
        return {}
    print(f"Loaded {len(queries)} sequences.")
    return queries


class CandidateFinder:
    """
    A worker class to find candidate sequences from RocksDB databases.
    Each worker instance in a process will have its own database handles.
    """
    def __init__(self, db_path: Path):
        """
        Initializes the finder but does not open DBs yet.
        DBs are opened on the first call within a process.
        """
        self.kmer_db_path = str(db_path / 'kmer_db')
        self.space_db_path = str(db_path / 'space_db')
        self.kmer_db = None
        self.space_db = None

    def _connect(self):
        """Initializes read-only RocksDB connections."""
        if self.kmer_db is None:
            print(f"Worker {os.getpid()}: Connecting to databases...")
            opts = rocksdb.Options(create_if_missing=False)
            self.kmer_db = rocksdb.DB(self.kmer_db_path, opts, read_only=True)
            self.space_db = rocksdb.DB(self.space_db_path, opts, read_only=True)

    def _fetch_list(self, db: rocksdb.DB, key: str) -> list:
        """Fetches and unpickles a list from the database."""
        raw = db.get(key.encode('utf-8'))
        return pickle.loads(raw) if raw else []

    def find(self, item: Tuple[str, str], k: int, min_diag_hits: int, min_total_score: int) -> Tuple[str, List[str]]:
        """
        Processes a single query item to find candidate sequences.
        """
        self._connect()  # Ensure DBs are connected
        key, query = item
        query_len = len(query)
        diag_hits = defaultdict(int)

        # 1. K-mer sliding window search
        for i in range(query_len - k + 1):
            kmer = query[i:i+k]
            for seq_id, pos in self._fetch_list(self.kmer_db, kmer):
                diag_hits[(seq_id, i - pos)] += 1

        # 2. Spaced seed search (e.g., pattern 1-1-1-1)
        if query_len >= 5:
            for i in range(query_len - 4):
                seed = query[i] + query[i+2] + query[i+4]
                for seq_id, pos in self._fetch_list(self.space_db, seed):
                    diag_hits[(seq_id, i - pos)] += 1

        # 3. Two-stage filtering
        # First pass: count total hits for sequences with enough diagonal hits
        first_pass_counts = Counter()
        lc = max(min_diag_hits, query_len // 3)
        for (seq_id, _), count in diag_hits.items():
            if count >= lc:
                first_pass_counts[seq_id] += count

        # Second pass: filter by total score
        final_candidates = []
        lv = max(min_total_score, query_len // 2)
        for seq_id, total_score in first_pass_counts.items():
            if total_score >= lv and seq_id != key:
                final_candidates.append(seq_id)

        return key, final_candidates


def worker_find_candidates(item: Tuple[str, str], db_path: Path, k: int, min_diag: int, min_total: int) -> Tuple[str, List[str]]:
    """
    A top-level function wrapper for the pool, which instantiates
    the finder class and calls its method.
    """
    finder = CandidateFinder(db_path)
    return finder.find(item, k, min_diag, min_total)


def main():
    """Main function to orchestrate the candidate search."""
    parser = create_arg_parser()
    args = parser.parse_args()

    queries = load_queries(args.query_path, args.min_query_len)
    if not queries:
        return

    # Use functools.partial to create a worker function with fixed arguments
    worker_func = partial(
        worker_find_candidates,
        db_path=args.db_path,
        k=args.kmer_size,
        min_diag=args.min_diag_hits,
        min_total=args.min_total_score
    )

    print(f"Starting search with {args.workers} worker processes...")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with mp.Pool(processes=args.workers) as pool, args.output_path.open('w') as fout:
        # Use imap_unordered for efficiency, as results can be written as they complete
        for key, candidates in pool.imap_unordered(worker_func, queries.items(), chunksize=100):
            if candidates:
                fout.write(f"{key}\t" + "\t".join(candidates) + "\n")

    print(f"Processing complete. Results saved to {args.output_path}")


if __name__ == '__main__':
    main()
