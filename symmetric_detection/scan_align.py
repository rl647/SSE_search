#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
"""
This script performs a two-stage analysis on protein structures.
1.  Aligns secondary structure element (SSE) sequences derived from protein chains
    to identify potentially similar substructures.
2.  Calculates the TM-score for these aligned substructures using US-align to
    quantify their 3D structural similarity.
The process is parallelized to efficiently handle large datasets.
"""

import argparse
import os
import re
import shutil
import string
import subprocess
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import multiprocessing as mp
import numpy as np

# Append local module path
BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)
sys.path.append(str(BASE_DIR / 'local_alignment_s'))
try:
    import local_alignment as la
except ImportError:
    print("Error: The 'local_alignment' module could not be found.")
    sys.exit(1)

# --- Constants ---
HELIX_CHARS = list(string.ascii_lowercase)
SHEET_CHARS = list(string.ascii_uppercase)#

ALL_CHARS = HELIX_CHARS + SHEET_CHARS + ['-', '=']
#%%
# --- Argument Parsing ---
# /home/runfeng/Dropbox/structure_searching/SSE_search/examples/sse/sse.txt
# /media/runfeng/86719ed4-89e5-44b4-ab1e-f5e7b808e4ed1/split_pdb
# /home/runfeng/US_align
# /home/runfeng/Dropbox/structure_searching/SSE_search/examples/symm_canner

def create_arg_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Run SSE alignment and TM-score calculation for protein substructures."
    )
    parser.add_argument("ss_path", type=Path, help="Path to the file containing SSE sequences and ranges.")
    parser.add_argument("pdb_path", type=Path, help="Path to the directory containing PDB files.")
    parser.add_argument("us_align_path", type=Path, help="Path to the US-align executable.")
    parser.add_argument("result_path", type=Path, help="Path to the directory for storing results.")
    return parser

# --- Data Loading ---

def load_similarity_matrix(filepath: Path) -> Dict[str, Dict[str, float]]:
    """
    Loads the SSE alignment scoring matrix from a text file.

    The file is expected to have:
    - 2 header lines.
    - A square matrix for SSE substitution scores.
    - A line for gap open penalties.
    - A line for gap extension penalties.
    """
    print(f"Loading similarity matrix from: {filepath}")
    with open(filepath, 'r') as f:
        lines = f.readlines()[2:]

    num_sse_types = len(HELIX_CHARS) + len(SHEET_CHARS)
    matrix_lines = [line.strip().split() for line in lines[:num_sse_types]]
    score_matrix = np.array(matrix_lines, dtype=float)

    gap_open = np.array(lines[num_sse_types].strip().split(), dtype=float)
    gap_extension = np.array(lines[num_sse_types + 1].strip().split(), dtype=float)

    sim_matrix = {char1: {char2: 0.0 for char2 in ALL_CHARS} for char1 in ALL_CHARS}
    sse_chars = HELIX_CHARS + SHEET_CHARS

    for i, char1 in enumerate(sse_chars):
        for j, char2 in enumerate(sse_chars):
            sim_matrix[char1][char2] = score_matrix[i, j]
        sim_matrix[char1]['-'] = gap_open[i]
        sim_matrix[char1]['='] = gap_extension[i]
        sim_matrix['-'][char1] = gap_open[i]
        sim_matrix['='][char1] = gap_extension[i]

    return sim_matrix

def load_sse_data(filepath: Path) -> Tuple[Dict[str, str], Dict[str, List[np.ndarray]]]:
    """Loads SSE sequences and residue ranges from the provided file."""
    sse_sequences = {}
    sse_ranges = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            protein_id = parts[0]
            sse_sequences[protein_id] = parts[-2]
            ranges_str = parts[-1].strip().split(' ')
            sse_ranges[protein_id] = [np.array(r.split('-'), dtype=int) for r in ranges_str]
    return sse_sequences, sse_ranges

def load_pdb_files(pdb_path: Path, protein_ids: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """Pre-loads all required PDB files into memory."""
    print("Loading PDB files...")
    pdb_data = {}
    for protein_id in protein_ids:
        pdb_file = pdb_path / f"{protein_id}.pdb"
        if not pdb_file.exists():
            print(f"Warning: PDB file not found for {protein_id}. Skipping.")
            continue
        
        pdb_data[protein_id] = defaultdict(list)
        with open(pdb_file) as f:
            for line in f:
                if line.startswith('ATOM'):
                    # Key by residue sequence number and insertion code
                    residue_key = line[22:27].strip()
                    pdb_data[protein_id][residue_key].append(line)
    return pdb_data

# --- Worker Functions for Multiprocessing ---

def perform_alignment(task: Tuple, sim_matrix: Dict) -> Optional[Dict[str, Any]]:
    """
    Worker function to perform local alignment for a single SSE pair.
    Filters results based on score and length.
    """
    pair_id, ss1, ss2, sr1, sr2 = task
    results = la.local_alignment(ss1, ss2, sr1, sr2, sim_matrix)
    
    if results.normalized_score < 0.0:
        return None

    aligned_sse1 = ''.join(results.aligned_ss1.strip().split('_')[::2]).replace('-', '')
    aligned_sse2 = ''.join(results.aligned_ss2.strip().split('_')[::2]).replace('-', '')

    if len(aligned_sse1) < 3 or len(aligned_sse2) < 3:
        return None

    # Find the start and end indices of the aligned segments
    try:
        start_idx1 = ss1.index(aligned_sse1)
        end_idx1 = start_idx1 + len(aligned_sse1) - 1
        start_idx2 = ss2.index(aligned_sse2)+end_idx1
        end_idx2 = start_idx2 + len(aligned_sse2) - 1
    except ValueError:
        # This can happen if the aligned sequence is not a contiguous substring
        return None

    return {
        "pair_id": pair_id,
        "protein_id": pair_id[:6],
        "score": results.normalized_score,
        "sse_range1": (start_idx1, end_idx1),
        "sse_range2": (start_idx2, end_idx2),
        "full_ss1": ss1,
        "full_ss2": ss2,
    }

def calculate_tm_score(
    alignment: Dict,
    sse_ranges: Dict,
    pdb_data: Dict,
    split_pdb_path: Path,
    us_align_path: Path
) -> Tuple[str, float]:
    """
    Worker function to extract PDBs, run US-align, parse the score,
    and reliably clean up temporary files.
    """
    pair_id = alignment["pair_id"]
    protein_id = alignment["protein_id"]
    pair_pdb_dir = split_pdb_path / pair_id
    
    try:
        # Create a dedicated subdirectory for this pair's PDB files
        pair_pdb_dir.mkdir(exist_ok=True)
        
        pdb1_path = pair_pdb_dir / f"{protein_id}_1.pdb"
        pdb2_path = pair_pdb_dir / f"{protein_id}_2.pdb"

        # Extract residue numbers and write substructure PDB files
        protein_sse_ranges = sse_ranges[protein_id]
        start_res1 = protein_sse_ranges[alignment["sse_range1"][0]][0]
        end_res1 = protein_sse_ranges[alignment["sse_range1"][1]][1]
        start_res2 = protein_sse_ranges[alignment["sse_range2"][0]][0]
        end_res2 = protein_sse_ranges[alignment["sse_range2"][1]][1]

        with open(pdb1_path, 'w') as f:
            for res_num in range(start_res1, end_res1 + 1):
                if str(res_num) in pdb_data[protein_id]:
                    f.writelines(pdb_data[protein_id][str(res_num)])

        with open(pdb2_path, 'w') as f:
            for res_num in range(start_res2, end_res2 + 1):
                if str(res_num) in pdb_data[protein_id]:
                    f.writelines(pdb_data[protein_id][str(res_num)])

        # Run US-align and parse output
        cmd = [str(us_align_path), str(pdb1_path), str(pdb2_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        tm_score_match = re.search(r"TM-score=\s*(\d\.\d+)", result.stdout)
        
        tm_score = float(tm_score_match.group(1)) if tm_score_match else 0.0

    except (subprocess.CalledProcessError, FileNotFoundError, AttributeError):
        tm_score = -1.0  # Indicate an error
        
    finally:
        # This block will ALWAYS run, ensuring cleanup
        if pair_pdb_dir.exists():
            shutil.rmtree(pair_pdb_dir, ignore_errors=True)
            
    return (pair_id, tm_score)

# --- Main Execution ---

def main():
    """Main function to orchestrate the alignment and scoring pipeline."""
    parser = create_arg_parser()
    args = parser.parse_args()
    cpu = mp.cpu_count()
    # --- 1. Setup ---
    args.result_path.mkdir(exist_ok=True)
    matrix_path = BASE_DIR / "alignment_matrix.txt"
    if not all([args.ss_path.exists(), args.pdb_path.is_dir(), matrix_path.exists()]):
        print("Error: One or more input paths are invalid. Please check them.")
        sys.exit(1)

    # --- 2. Load Data ---
    sim_matrix = load_similarity_matrix(matrix_path)
    sse_sequences, sse_ranges = load_sse_data(args.ss_path)
    pdb_data = load_pdb_files(args.pdb_path, list(sse_sequences.keys()))

    # --- 3. Generate Alignment Tasks ---
    alignment_tasks = []
    for protein_id, ss_sequence in sse_sequences.items():
        # Consider all split points, ensuring each fragment has at least 3 SSEs
        for i in range(len(ss_sequence) - 5):
            split_point = i + 3
            pair_id = f"{protein_id}_{split_point}"
            ss1 = ss_sequence[:split_point]
            ss2 = ss_sequence[split_point:]
            sr1 = sse_ranges[protein_id][:split_point]
            sr2 = sse_ranges[protein_id][split_point:]
            alignment_tasks.append((pair_id, ss1, ss2, sr1, sr2))
    
    print(f"Generated {len(alignment_tasks)} alignment tasks.")

    # --- 4. Perform Alignments in Parallel ---
    print("Performing SSE alignments...")
    with Pool(cpu) as pool:
        # Use partial to "bake in" the sim_matrix argument
        align_func = partial(perform_alignment, sim_matrix=sim_matrix)
        # pool.map is efficient for this kind of task
        alignment_results = pool.map(align_func, alignment_tasks)

    # Filter out unsuccessful alignments
    valid_alignments = [res for res in alignment_results if res is not None]
    print(f"Found {len(valid_alignments)} valid alignments.")

    if not valid_alignments:
        print("No valid alignments found. Exiting.")
        sys.exit(0)

    # --- 5. Calculate TM-Scores in Parallel ---
    print("Calculating TM-scores for valid alignments...")
    split_pdb_path = args.result_path / "split_pdb_files"
    if split_pdb_path.exists():
        shutil.rmtree(split_pdb_path) # Clean up from previous runs
    split_pdb_path.mkdir()

    tm_score_tasks = valid_alignments
    
    with Pool(cpu) as pool:
        score_func = partial(
            calculate_tm_score,
            sse_ranges=sse_ranges,
            pdb_data=pdb_data,
            split_pdb_path=split_pdb_path,
            us_align_path=args.us_align_path
        )
        tm_score_results = pool.map(score_func, tm_score_tasks)

    tm_scores_map = dict(tm_score_results)
    
    # --- 6. Write Final Report ---
    output_file = args.result_path / "final_results.txt"
    print(f"Writing final results to {output_file}")
    with open(output_file, 'w') as f:
        header = [
            "Pair_ID", "Protein_ID", "Alignment_Score",
            "SSE_Range_1", "SSE_Range_2", "TM_Score"
        ]
        f.write("\t".join(header) + "\n")
        
        for alignment in valid_alignments:
            pair_id = alignment["pair_id"]
            tm_score = tm_scores_map.get(pair_id, "N/A")
            range1_str = f"{alignment['sse_range1'][0]}-{alignment['sse_range1'][1]}"
            range2_str = f"{alignment['sse_range2'][0]}-{alignment['sse_range2'][1]}"
            
            row = [
                pair_id,
                alignment["protein_id"],
                f"{alignment['score']:.4f}",
                range1_str,
                range2_str,
                f"{tm_score:.4f}" if isinstance(tm_score, float) else tm_score,
            ]
            f.write("\t".join(row) + "\n")

    print("Processing complete.")

if __name__ == '__main__':
    main()