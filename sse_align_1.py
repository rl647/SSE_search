#%%

import argparse
import string
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.append(f'{base_dir}/local_alignment_s')
import local_alignment as lc1
import multiprocessing as mp
from multiprocessing import Pool
sys.path.append(f'{base_dir}')
from ML import HybridResNetBiLSTM
from alignment_featurizaton import *
model_path = f'{base_dir}/Bilstm_model.pth'
# --- Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "npu" if torch.npu.is_available() else "cpu")
HELIX_CHARS = list(string.ascii_lowercase)
SHEET_CHARS = list(string.ascii_uppercase)
ALL_CHARS = HELIX_CHARS + SHEET_CHARS + ['-', '=']

#%%
def create_arg_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(description="Run sequence alignment and TM-score prediction.")
    parser.add_argument("ss_path", type=Path, help="Path to the directory ")
    parser.add_argument("--align_threshold", type=float, default=0.35, help="Minimum aligned score to be written to the output file.")
    parser.add_argument("--tm_threshold", type=float, default=0.5, help="Minimum predicted TM-score to be written to the output file.")
    parser.add_argument("--ML", type=bool, default=1, help="")

    return parser


def load_similarity_matrix(filepath: Path) -> Dict[str, Dict[str, float]]:
    print(f"Loading similarity matrix from: {filepath}")
    with open(filepath,'r') as f:
        # Skip header lines
        lines = f.readlines()[2:]

    num_sse_types = len(HELIX_CHARS) + len(SHEET_CHARS)

    # Parse the matrix, gap open, and gap extension scores
    matrix_lines = [line.strip().split() for line in lines[:num_sse_types]]
    gap_open = np.array(lines[num_sse_types].strip().split(), dtype=float)
    gap_extension = np.array(lines[num_sse_types + 1].strip().split(), dtype=float)
    score_matrix = np.array(matrix_lines, dtype=float)

    # Prepare the matrix in the dictionary format required by the alignment function
    sim_matrix = {char1: {char2: 0.0 for char2 in ALL_CHARS} for char1 in ALL_CHARS}

    for i, char1 in enumerate(HELIX_CHARS + SHEET_CHARS):
        for j, char2 in enumerate(HELIX_CHARS + SHEET_CHARS):
            sim_matrix[char1][char2] = score_matrix[i, j]
        sim_matrix[char1]['-'] = gap_open[i]
        sim_matrix[char1]['='] = gap_extension[i]
        sim_matrix['-'][char1] = gap_open[i]
        sim_matrix['='][char1] = gap_extension[i]

    return sim_matrix


def load_model(model_path: Path, device: torch.device) -> HybridResNetBiLSTM:
    """Loads the pre-trained HybridResNetBiLSTM model."""
    print(f"Loading model from: {model_path}")
    model = HybridResNetBiLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval() # Set model to evaluation mode
    return model


def run_predictions(
    model: HybridResNetBiLSTM,
    results_path: Path,
    output_file: Path,
    tm_threshold: float
):

    print(f"Processing alignment files in: {results_path}")
    all_predictions = []

    for protein_file in results_path.glob("*.txt"):
        protein_name = protein_file.stem
        alignments_data = []
        labels = []
        pairs_seen = set()
        align_map = {}

        with protein_file.open('r') as f:
            for line in f:
                parts = line.strip().split('\t')

                # Avoid duplicate pairs (e.g., (A,B) and (B,A))
                pair = tuple(sorted((protein_name, parts[0])))
                if pair in pairs_seen:
                    continue

                pairs_seen.add(pair)
                original_pair = (protein_name, parts[0])
                align_str1, align_str2 = parts[1], parts[2]

                # Featurize alignment for the model
                align_features = alignment_to_features(align_str1, align_str2, osc_helix, osc_sheet)
                # print(align_features)
                alignments_data.append(torch.tensor(align_features, dtype=torch.float))
                labels.append(original_pair)
                align_map[original_pair] = (align_str1, align_str2)

        if not alignments_data:
            # print(f"No valid alignments found for {protein_name}, skipping.")
            continue

        # Create DataLoader and run evaluation
        dataset = SequenceAlignmentDataset(alignments_data, labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=custom_pad_sequences)

        with torch.no_grad(): # Ensure no gradients are calculated
            predictions, _ = evaluate(model, loader, DEVICE)

        # Collect predictions that meet the threshold
        for i, score in enumerate(predictions):
            if score >= tm_threshold:
                label = labels[i]
                as1, as2 = align_map[label]
                all_predictions.append(f"{label[0]}\t{label[1]}\t{as1}\t{as2}\t{score[0]:.3f}\n")

    # Write all filtered predictions to a single file
    print(f"Writing {len(all_predictions)} predictions to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w') as f_out:
        f_out.writelines(all_predictions)

def batch_alignment(batch, sim_matrix):
    parser = create_arg_parser()
    args = parser.parse_args()
    for i, (pair0,pair1, ss0,ss1,sr1,sr2) in enumerate(batch):
        results = lc1.local_alignment(ss0, ss1, sr1, sr2, sim_matrix)
        as1 = results.aligned_ss1
        as2 = results.aligned_ss2
        score = results.normalized_score
        if score>=float(args.align_threshold):
            as11 = as1.strip().split('_')[::2]
            as22 = as2.strip().split('_')[::2]
            r0 = ''.join(as11).replace('-','')
            r1 = ''.join(as22).replace('-','')
            if len(r0)>=3 and len(r1)>=3:
                r0_range = ss0.index(r0) if r0 != '' else 0
                r1_range = ss1.index(r1) if r1 != '' else 0
                predicted_range = [[r0_range, r0_range + len(r0) - 1], [r1_range, r1_range + len(r1) - 1]]
                with open(f'{args.ss_path}/alignment_results/{pair0[:6]}.txt','a') as f:
                    f.write(pair1+'\t'+ f'\t{as1}\t{as2}\t{predicted_range[0][0]}-{predicted_range[0][1]}\t{predicted_range[1][0]}-{predicted_range[1][1]}\t')
                    f.write(ss0+'\t'+ss1+'\t'+f'{score:.3f}'+'\n')

def create_batches(paired_deteails, batch_size,cpu):

    return [paired_deteails[i * batch_size:(i + 1) * batch_size] for i in range(cpu+1)]
def compute_fitness(paired_deteails,sim_matrix,cpu):
    fitnesses = []
    with Pool(cpu) as pool:
        batch_size = len(paired_deteails)//cpu   
        pairs = create_batches(paired_deteails, batch_size,cpu)
        results = pool.starmap(batch_alignment, [(batch, sim_matrix) for batch in pairs])
    return fitnesses

def main():
    """Main function to orchestrate the alignment and prediction workflow."""
    parser = create_arg_parser()
    args = parser.parse_args()
    matrix_path = f"{base_dir}/alignment_matrix.txt"
    sim_matrix = load_similarity_matrix(matrix_path)
    model = load_model(model_path, DEVICE)

    print("Running initial local alignment...")
    alignment_results_path = args.ss_path / "alignment_results"
    sse_path = args.ss_path / "sse"/"sse.txt"
    ML = args.ML
    sse = {}
    sse_range = {}
    with open(sse_path) as f:
        for line in f:
            s=line.strip().split('\t')
            sse[s[0]] = s[-2]
            sse_range[s[0]] = [np.array(e.strip().split('-'),dtype=int)for i,e in enumerate(s[-1].strip().split(' '))]
    pairs = []
    with open(args.ss_path / "sp.txt") as f:
        for line in f:
            s=line.strip().split('\t')
            pairs.append([s[0],s[1],sse[s[0]],sse[s[1]],sse_range[s[0]],sse_range[s[1]]])
    cpu  = mp.cpu_count()
    compute_fitness(pairs,sim_matrix,cpu)
    tm_output = args.ss_path / "tm.txt"
    alignment_results_path.mkdir(exist_ok=True)
    # --- Step 3: Run predictions on alignment results ---
    if ML:
        run_predictions(model, alignment_results_path, tm_output, args.tm_threshold)

    print("Processing complete.")

import time
if __name__ == '__main__':
    t1 = time.time()    
    main()
    print(time.time()-t1)

# %%
