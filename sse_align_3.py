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
    parser.add_argument("alignment_results", type=Path, help="Path to the alignment_results ")
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
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=False), strict=True)
    model.eval() # Set model to evaluation mode
    return model

import shutil
def run_predictions(
    model: HybridResNetBiLSTM,
    results_path: Path,
    output_file: Path,
    tm_threshold: float
):

    print(f"Processing alignment files in: {results_path}")
    all_predictions = []
    # print(results_path)
    # results_path = '/media/runfeng/86719ed4-89e5-44b4-ab1e-f5e7b808e4ed1/structure_searching/conformational_changes/alignment_results'
    for protein_file in results_path.glob("*.txt"):
        protein_name = protein_file.stem
        alignments_data = []
        labels = []
        pairs_seen = set()
        align_map = {}

        with protein_file.open('r') as f:
            for line in f:
                parts = list(filter(None,(line.strip().split('\t'))))

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
        loader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=custom_pad_sequences,num_workers=max(1, mp.cpu_count() // 2), pin_memory=True)# A good starting point
        with torch.no_grad(): # Ensure no gradients are calculated
            predictions, _ = evaluate(model, loader, DEVICE)
        # Collect predictions that meet the threshold
        for i, score in enumerate(predictions):
            # print(score)
            if score >= tm_threshold:
                label = labels[i]
                as1, as2 = align_map[label]
                all_predictions.append(f"{label[0]}\t{label[1]}\t{as1}\t{as2}\t{score[0]:.3f}\n")
        # print(len(predictions),len(all_predictions))
    shutil.rmtree('/media/runfeng/86719ed4-89e5-44b4-ab1e-f5e7b808e4ed1/structure_searching/conformational_changes/alignment_results',ignore_errors=True)
    os.makedirs('/media/runfeng/86719ed4-89e5-44b4-ab1e-f5e7b808e4ed1/structure_searching/conformational_changes/alignment_results',exist_ok=True)

    # Write all filtered predictions to a single file
    print(f"Writing {len(all_predictions)} predictions to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open('/media/runfeng/86719ed4-89e5-44b4-ab1e-f5e7b808e4ed1/structure_searching/conformational_changes/tm.txt','a') as f_out:
        f_out.writelines(all_predictions)

def batch_alignment(batch, sim_matrix):
    batch_results = []
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
                result_line = pair1+'\t'+ f'\t{as1}\t{as2}\t{predicted_range[0][0]}-{predicted_range[0][1]}\t{predicted_range[1][0]}-{predicted_range[1][1]}\t'
                result_line += (ss0+'\t'+ss1+'\t'+f'{score:.3f}'+'\n')
                batch_results.append((f"{pair0[:6]}.txt", result_line))
    return batch_results
                # with open(f'/media/runfeng/86719ed4-89e5-44b4-ab1e-f5e7b808e4ed1/structure_searching/conformational_changes/alignment_results/{pair0[:6]}.txt','a') as f:
                #     f.write(pair1+'\t'+ f'\t{as1}\t{as2}\t{predicted_range[0][0]}-{predicted_range[0][1]}\t{predicted_range[1][0]}-{predicted_range[1][1]}\t')
                #     f.write(ss0+'\t'+ss1+'\t'+f'{score:.3f}'+'\n')

def create_batches(paired_deteails, batch_size,cpu):

    return [paired_deteails[i * batch_size:(i + 1) * batch_size] for i in range(cpu+1)]
def compute_fitness(paired_deteails,sim_matrix,cpu):
    fitnesses = []
    with Pool(cpu) as pool:
        batch_size = len(paired_deteails)//cpu   
        pairs = create_batches(paired_deteails, batch_size,cpu)
        results_from_pool = pool.starmap(batch_alignment, [(batch, sim_matrix) for batch in pairs])
    print("Writing alignment results to disk...")
    # Use a dictionary to efficiently group lines for each file
    output_files = defaultdict(list)
    for result_list in results_from_pool:
        for filename, line in result_list:
            output_files[filename].append(line)

    # Write to each file in an efficient batch
    for filename, lines in output_files.items():
        with open(f'/media/runfeng/86719ed4-89e5-44b4-ab1e-f5e7b808e4ed1/structure_searching/conformational_changes/alignment_results/{filename}', 'a') as f:
            f.writelines(lines)
from collections import defaultdict
def main():
    """Main function to orchestrate the alignment and prediction workflow."""
    parser = create_arg_parser()
    args = parser.parse_args()
    matrix_path = f"{base_dir}/alignment_matrix.txt"
    sim_matrix = load_similarity_matrix(matrix_path)
    model = load_model(model_path, DEVICE)

    print("Running initial local alignment...")
    alignment_results_path = args.alignment_results 
    sse_path = args.ss_path/"sse.txt"
    ML = args.ML
    sse = {}
    sse_range = {}
    with open(sse_path) as f:
        for line in f:
            s=line.strip().split('\t')
            sse[s[0]] = s[1]
            # sse_range[s[1]] = [0,0]
            sse_range[s[0]] = [np.array(e.strip().split('-'),dtype=int)for i,e in enumerate(s[-1].strip().split(' '))]
    pairs = []
    mappings = defaultdict(set)
    with open('/home/runfeng/Dropbox/conformational_changes/database/sse/sse.txt') as f:
        for line in f:
            s=line.strip().split('\t')
            mappings[s[1]].update(set(s[2:]))
    print(len(sse))
    finish = set()
    with open('/media/runfeng/86719ed4-89e5-44b4-ab1e-f5e7b808e4ed1/structure_searching/conformational_changes/finish.txt') as f:
        for line in f:
            s=line.strip().split('\t')
            finish.add(s[0])

    cpu  = mp.cpu_count()
    access = []
    with open(args.ss_path / "selected_pairs.txt") as f:
        for line in f:
            s=line.strip().split('\t')
            if len(s)<2:
                continue
            if s[0] in finish:
                continue
            access.append(s[0])
            s0 = mappings[s[0]]
            
            for i, e in enumerate(s[1:]):
                if e in finish:
                    continue
                se = mappings[e]
                for e0 in s0:
                    for e2 in se:
                        if e0 in sse and e2 in sse: 
                            pairs.append([e0,e2,sse[e0],sse[e2],sse_range[e0],sse_range[e2]])
            finish.add(s[0])
            tm_output = args.ss_path / "tm.txt"
            if len(pairs)>=5000000:
                print('alignment start ',len(pairs))
                compute_fitness(pairs,sim_matrix,cpu)
                pairs = []
                
                run_predictions(model, alignment_results_path, tm_output, args.tm_threshold)
                with open('/media/runfeng/86719ed4-89e5-44b4-ab1e-f5e7b808e4ed1/structure_searching/conformational_changes/finish.txt','a') as f:
                    f.write('\n'.join(access)+'\n')
                access = []
                # break
            # break
    print('alignment start ',len(pairs))
    compute_fitness(pairs,sim_matrix,cpu)
    pairs = []
    tm_output = args.ss_path / "tm.txt"
    alignment_results_path.mkdir(exist_ok=True)
    run_predictions(model, alignment_results_path, tm_output, args.tm_threshold)
    with open('/media/runfeng/86719ed4-89e5-44b4-ab1e-f5e7b808e4ed1/structure_searching/conformational_changes/finish.txt','a') as f:
        f.write('\n'.join(access)+'\n')
    access = []
    # access = []
    # --- Step 3: Run predictions on alignment results ---
    # if ML:
    #     

    print("Processing complete.")

import time
if __name__ == '__main__':
    t1 = time.time()    
    main()
    print(time.time()-t1)
    

# %%
# from collections import defaultdict
# "/home/runfeng/Dropbox/conformational_changes/database/sp.txt"
# mapping = defaultdict(list)
# with open('/home/runfeng/Dropbox/conformational_changes/database/r_sse1.txt') as f:
#     for line in f:
#         s=line.strip().split('\t')
#         mapping[s[1]].append(s[2])
# print(len(mapping))
# pairs = {}
    
#%%

# sse,sse_range = {},{}
# with open('/home/runfeng/Dropbox/conformational_changes/database/sse/sse.txt') as f:
#     for line in f:
#         s=line.strip().split('\t')
#         for i,e in enumerate(s[2:]):
#             sse[e] = ''
#             sse_range[e] = ''
# print(len(sse))
# import sys
# sys.path.append('/home/runfeng/Dropbox/scripts/')
# from sse_extraction import *
# for key, val in sse.items():
#     sse_range[key],sse[key] = ss_extraction(key)
# # %%

# with open('/home/runfeng/Dropbox/conformational_changes/database/sse.txt','w')as f:
#     for key, val in sse.items():
#         f.write(key+'\t'+val+'\t')
#         for i, e in enumerate(sse_range[key]):
#             f.write(f'{e[0]}-{e[1]} ')
#         f.write('\n')
# #%%













