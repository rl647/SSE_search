# SSE-search: Ultra-Fast Protein Structure Searching & Analysis

`SSE-search` is a high-performance computational pipeline for protein structure analysis. It introduces an ultra-compact representation of protein structures using their Secondary Structure Elements (SSEs), enabling tertiary structure comparison and searching at speeds ~200 times faster than state-of-the-art tools.

Beyond rapid searching, the pipeline is a powerful tool for analyzing protein architecture, capable of detecting internal symmetry, repeated motifs, and structural rearrangements like circular permutations.

The methods are described in detail in our preprints:
* **Compression of Protein Secondary Structures Enables Ultra-Fast and Accurate Structure Searching**
* **Detection of Protein Symmetry and Structural Rearrangements using Secondary Structure Elements**



---

## Key Features

* **🚀 Ultra-Fast Searching:** Compares protein structures with a ~200x speedup over similar tools.
* **🎯 High Accuracy:** Achieves high search accuracy despite significant data compression.
* **🧬 Symmetry Detection:** Identifies internal symmetry and repeated structural motifs.
* **🔄 Rearrangement Analysis:** Detects circular permutations and other structural rearrangements.

---

## Installation

### 1. Clone the Repository
First, clone this repository to your local machine:
```bash
git clone [https://github.com/rl647/SSE_search.git](https://github.com/rl647/SSE_search.git)
cd SSE_search
```

### 2. Set Up the Environment

We strongly recommend using **Conda** to manage the dependencies, as this simplifies the installation of specific versions of scientific software.

```bash
# Create a new conda environment
conda create -n sse_env python=3.12

# Activate the environment
conda activate sse_env

# Install Python packages from requirements.txt
pip install -r requirements.txt

# Install dssp (mkdssp) version 4.0.4 from the bioconda channel
conda install -c bioconda dssp=4.0.4
```
**Note:** A `requirements.txt` file has been provided in the repository to simplify the installation of Python packages.

### 3. Install External Dependencies

You can use US-align for final structural alignment and validation.
* **US-align:** Please download and compile it from the official website: [zhanggroup.org/US-align/](https://zhanggroup.org/US-align/)

---

## Usage Workflow

The pipeline consists of several stages, from data preparation to final analysis.

### Stage 1: Data Preparation

**1. Extract Secondary Structure Elements (SSEs):**
Convert a directory of PDB files into SSE files.
```bash
python3 dataset_preparation/sse_extraction.py /path/to/pdb/folder
```

### Stage 2: Pre-filtration (for large database searches)

**2. Build a Seeding Database:**
Generate space seeds and k-mer seeds for the pre-filtration stage.
```bash
python3 filtration/database_construction.py /path/to/save/data /path/to/sse_files
```

**3. Run the Filtration:**
Search a query SSE against the database to find promising candidate pairs.
```bash
python filtration/filtration.py /path/to/your_db /path/to/your_query.sse /path/to/output.txt
```

### Stage 3: SSE Alignment & Analysis

**4. Perform SSE Alignments and Predict TM-scores:**
There are two scripts for this, depending on your input format.
* **`sse_align1.py`** (Uses pre-filtered pairs):
    ```bash
    # `path1` should be a directory containing `sse.txt` (all SSEs) and `sp.txt` (candidate pairs from filtration).
    python3 sse_align1.py /path/to/input/folder
    ```
* **`sse_align2.py`** (Directly compares two database):
    ```bash
    # `path2` should contain `ss1.txt`, `ssr1.txt` (query) and `ss2.txt`, `ssr2.txt` (target).
    python3 sse_align2.py /path/to/input/folder
    ```

### Stage 4: Symmetry & Rearrangement Detection

**5. Replicate SSEs for Self-Comparison:**
Prepare an SSE file for detecting internal duplications.
```bash
python3 sse_replication.py /path/to/sse.txt /path/to/replicated_sse.txt /path/to/output_folder
```

**6. Run the Duplication Pipeline:**
Use the output from the replication step to find symmetric or circularly permuted structures.
```bash
# The path here is the output folder from the `sse_replication.py` script.
python3 sse_align2.py /path/to/output_folder
```

**7. Perform Full Self-Scanning Alignment:**
Use `scan_align.py` to exhaustively scan a protein against itself and validate findings with US-align.
```bash
python scan_align.py /path/to/sse.txt /path/to/pdb_database /path/to/results_folder --us_align_path /path/to/US-align
```
---

## How It Works

The pipeline converts a protein's 3D coordinates into a compressed sequence of its Secondary Structure Elements (SSEs). Each SSE is represented by its type (α-helix, β-sheet), length, and spatial information. This compact representation allows for rapid comparison using algorithms that analyze these sequences to find similarities, symmetries, and rearrangements, bypassing the need for computationally intensive 3D alignment for initial searches.

---

## Citation

If you use `SSE-search` in your research, please cite our work:

```bibtex
@article{Lin2025compression,
  title={Compression of Protein Secondary Structures Enables Ultra-Fast and Accurate Structure Searching},
  author={Lin, Runfeng and Ahnert, Sebastian E.},
  journal={bioRxiv},
  year={2025},
  doi={}
}

@article{Lin2025detection,
  title={Detection of Protein Symmetry and Structural Rearrangements using Secondary Structure Elements},
  author={Lin, Runfeng and Ahnert, Sebastian E.},
  journal={bioRxiv},
  year={2025},
  doi={}
}
```

---

## Contact

Runfeng Lin - [rl647@cam.ac.uk](mailto:rl647@cam.ac.uk)
Sebastian Ahnert - [sea31@cam.ac.uk](mailto:sea31@cam.ac.uk)

Project Link: [https://github.com/rl647/SSE_search](https://github.com/rl647/SSE_search)
