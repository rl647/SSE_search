#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:25:27 2024

@author: runfeng
"""
#%%
import os
from Bio import Align
from matplotlib.ticker import PercentFormatter
from Bio.Emboss.Applications import NeedleCommandline
from Bio import SeqIO
import string
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBParser
import os
from Bio.PDB.PDBExceptions import PDBConstructionWarning 
import warnings
import subprocess as sp
import gzip
import shutil
from Bio.PDB import MMCIFParser, PDBParser
from multiprocessing import Pool
import string
import sys
from multiprocessing import Pool
from functools import partial
#%%

warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.PDBParser")
warnings.filterwarnings("ignore", category=PDBConstructionWarning)
warnings.filterwarnings("ignore")

#%%
def sec(protein,p):
    error_log=[]
    
    try:
        structure = p.get_structure("mutant2", f"{protein}")
        model = structure[0]
        # print(model)
        dssp = DSSP(model, f"{protein}", dssp='mkdssp')
    
        seq = ''
        sec_structure = ''
        for z in range(len(dssp)):
            a_key = list(dssp.keys())[z]
            seq += dssp[a_key][1]
            # print(dssp[a_key][2])
            if dssp[a_key][2] == 'B' or dssp[a_key][2] == 'E':
                sec_structure += 'E'                    
            elif dssp[a_key][2] == 'G' or dssp[a_key][2] == 'H':
                sec_structure += 'H'
            else:
    
                sec_structure += 'C'
        # print(sec_structure)
        return seq,sec_structure
       
        
    except Exception as e:
        error_log.append((protein, str(e)))
        # print(v,str(e))
        return False, False

H = list(string.ascii_lowercase)
E = list(string.ascii_uppercase)
HE = H+E

score = {}
for i, e in enumerate(E):
    if i <= 8:
        score[i+2] = e
    elif i > 8 and i <= 18:
        score[(i*2)-6] = e
    elif i > 18:
        score[(i*3)-24] = e
finish = set()

#%%
p1 = MMCIFParser()
p2 = PDBParser()
target='/home/runfeng/Documents/afdb'
afd = ''
def process_protein_file(file_path, p1_param, p2_param, score_map):

    try:
        filename = os.path.basename(file_path)
        key = filename[:-4]  # Remove extension (e.g., '.pdb')

        # 1. Determine parameters and get secondary structure string
        p = p2_param if filename.endswith('.pdb') else p1_param
        sequence, sec_structure = sec(file_path, p)

        if not sec_structure:
            return key, "" # Return empty score for empty structure

        # 2. Parse secondary structure string into segments (retains original logic)
        ss_elements = []
        ss_range = []
        c = 1
        # Loop until the second to last element
        for i, e in enumerate(sec_structure[:-1]):
            # This block handles the final two elements of the sequence
            if i == len(sec_structure) - 2:
                if e == sec_structure[-1]:
                    c += 1
                    ss_elements.append(e + str(c))
                else:
                    if e != 'C' and c == 1:
                        ss_elements.append('C' + str(2))
                    elif sec_structure[-1] != 'C':
                        ss_range.append([str(i+1-c),str(i+1)])
                        ss_elements.append(e + str(c))
                        ss_elements.append('C' + str(1))
                    else:
                        ss_elements.append(e + str(c + 1))
                        ss_range.append([str(i+1-c),str(i+2)])
            # This block handles all other elements
            else:
                if e == sec_structure[i+1]:
                    c += 1
                elif e != sec_structure[i+1]:
                    if e == 'C':
                        ss_elements.append(e + str(c))
                        c = 1
                    else:
                        if c == 1:
                            ss_elements.append('C' + str(c))
                            c = 1
                        else:
                            ss_elements.append(e + str(c))
                            ss_range.append([str(i+1-c),str(i+1)])
                            c = 1

        # 3. Convert segments to sse 
        css = []
        for ele in ss_elements:
            if ele[0] != 'C':
                i2 = int(ele[1:])
                cs = None
                if i2 in score_map:
                    cs = score_map[i2]
                elif i2 + 1 in score_map:
                    cs = score_map[i2 + 1]
                elif i2 + 2 in score_map:
                    cs = score_map[i2 + 2]
                elif i2 > 51: # Cap at 51
                    cs = score_map[51]
                if cs:
                    # Adjust case based on element type
                    css.append(cs if ele[0] == 'E' else cs.lower())
        
        final_sse = ''.join(css)
        return key,sequence, sec_structure, final_sse, ss_range

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None, None, None, None


# --- Main Execution Block ---
if __name__ == '__main__':
    protein_path = sys.argv[1]
    # protein_path = '/home/runfeng/test'
    num_processes = int(sys.argv[2])
    # num_processes = 10
    output = sys.argv[3]
    output = '/home/runfeng/sse.txt'
    files_to_process = [
        os.path.join(protein_path, f)
        for f in os.listdir(protein_path)
        if f.endswith(('.pdb', '.cif')) # Add any other extensions you use
    ]
    print(f"Found {len(files_to_process)} protein files to process.")


    worker_func = partial(process_protein_file, p1_param=p1, p2_param=p2, score_map=score)


    with Pool(processes=num_processes) as pool:
        # The 'map' function runs the worker_func on every file in files_to_process
        results = pool.map(worker_func, files_to_process)

    protein_results = {key: [v1,v2,v3,v4] for key, v1,v2,v3,v4 in results if key is not None}

    print("\n--- Processing Complete ---")
    print("Final Results:")
    with open(f'{output}','w') as f:
        for key, val in protein_results.items():
            f.write(key+'\t'+'\t'.join(val[:-1])+'\t')
            for e in val[-1]:
                f.write('-'.join(e)+' ')
            f.write('\n')

#%%
