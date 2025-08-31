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
sys.path.append(f'{base_dir}/local_alignment_s2')
import local_alignment_v2 as lc2

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
    parser.add_argument("ss_path", type=Path, help="Path to the directory containing ss1.txt, ssr1.txt, etc.")
    parser.add_argument("--align_threshold", type=float, default=0.35, help="Minimum aligned score to be written to the output file.")
    parser.add_argument("--tm_threshold", type=float, default=0.5, help="Minimum predicted TM-score to be written to the output file.")
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
    alignments_data = []
    labels = []
    pairs_seen = set()
    align_map = {}
    for protein_file in results_path.glob("*.txt"):
        protein_name = protein_file.stem
        

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
        print(f"No valid alignments found for {protein_name}, skipping.")
        return

    # Create DataLoader and run evaluation
    dataset = SequenceAlignmentDataset(alignments_data, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=custom_pad_sequences)

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


def main():
    """Main function to orchestrate the alignment and prediction workflow."""
    parser = create_arg_parser()
    args = parser.parse_args()
    matrix_path = f"{base_dir}/alignment_matrix.txt"
    # --- Step 1: Load resources ---
    sim_matrix = load_similarity_matrix(matrix_path)
    model = load_model(model_path, DEVICE)

    # --- Step 2 (Optional): Run initial alignment if needed ---
    # This part is commented out as it seems you are processing pre-existing
    # alignment files. If you need to run the alignment first, you can
    # uncomment and adapt this section.
    #
    print("Running initial local alignment...")
    alignment_results_path = args.ss_path / "alignment_results"
    sse_path = args.ss_path / "sse"
    tm_output = args.ss_path / "tm.txt"
    alignment_results_path.mkdir(exist_ok=True)
    lc2.align(
        str(sse_path / "ss1.txt"),
        str(sse_path / "ssr1.txt"),
        str(sse_path / "ss2.txt"),
        str(sse_path / "ssr2.txt"),
        str(alignment_results_path),
        sim_matrix,
        threshold=float(args.align_threshold)
    )

    # --- Step 3: Run predictions on alignment results ---
    run_predictions(model, alignment_results_path, tm_output, args.tm_threshold)

    print("Processing complete.")


import time
if __name__ == '__main__':
    t1 = time.time()    
    main()
    print(time.time()-t1)
# %%
