# Molecular Clustering Tools

A suite of tools for molecular clustering based on SMILES strings. This project supports multiple clustering methods, including distance-based clustering, Tanimoto similarity, and two-stage clustering, with optional GPU acceleration.

## Features

- **Molecular Fingerprinting**: Morgan fingerprints with customizable parameters.
- **Clustering Approaches**: Distance-based, similarity-based, and two-stage clustering.
- **GPU Acceleration**: CUDA support for large datasets.
- **Multi-threshold Analysis**: Clustering with multiple similarity thresholds.
- **Memory Optimization**: Streaming mode for large datasets.
- **Progress Tracking**: Detailed progress bars and verbose reporting.

## Installation

### Prerequisites
```bash
pip install pandas numpy rdkit-pypi scikit-learn tqdm networkx
```

### Optional GPU Support
```bash
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x
```

### PyTorch Alternative
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Distance-based Clustering
```bash
python run_clustering.py --input molecules.csv --output results.csv --threshold 0.3 --verbose
```

### Similarity-based Clustering
```bash
python run_simclustering.py --input molecules.csv --output results.csv --sim_threshold 0.7 --verbose
```

### Two-Stage Clustering
```bash
python run_clustering.py --input molecules.csv --reference-file reference.csv --output results.csv --two-stage --verbose
```

### GPU Acceleration
```bash
python run_clustering.py --input molecules.csv --output results.csv --cuda --verbose
python run_simclustering.py --input molecules.csv --output results.csv --cuda --sim_threshold 0.7 --verbose
```

## Input and Output Formats

### Input Format
CSV file with at least the following columns:
- **SMILES**: Molecular structures in SMILES format.
- **ID**: Unique identifier for each molecule.

Example:
```csv
molecule_id,smiles
CHEMBL001,CCO
CHEMBL002,CCC
CHEMBL003,CCCO
```

### Output Format

#### Distance-based Clustering
- `molecule_id`: Molecule identifier.
- `smiles`: SMILES string.
- `cluster_id`: Cluster number.
- `cluster_size`: Number of molecules in the cluster.

#### Similarity-based Clustering
- `molecule_id`: Molecule identifier.
- `cluster_members_X_X`: Molecules in the cluster (threshold-specific).
- `similarity_scores_X_X`: Similarity scores (threshold-specific).

#### Two-Stage Clustering
- `molecule_id`: Molecule identifier.
- `reference_cluster_id`: Assigned reference cluster ID.
- `similarity_to_reference`: Similarity score to the assigned reference cluster.

## Performance Optimization

- **Streaming Mode**: Processes molecules without storing the full similarity matrix.
- **Chunked Processing**: Breaks large datasets into manageable chunks.
- **GPU Acceleration**: Offloads computation to GPU when available.

### Example for Large Datasets
```bash
python run_clustering.py --input large_dataset.csv --output results.csv --cuda --chunk-size 50
python run_simclustering.py --input large_dataset.csv --output results.csv --streaming --cuda
```

## Hardware Requirements

- **CPU Mode**: 8GB+ RAM recommended for datasets >5K molecules.
- **GPU Mode**: 4GB+ VRAM recommended for datasets >10K molecules.

## Repository Structure

```
ClusMol/
├── run_simclustering.py         # Similarity-based clustering
├── run_clustering.py            # Distance-based and two-stage clustering
├── smiles_clustering.py         # Core clustering library
├── requirements.txt             # Dependencies
├── example_datasets/            # Example datasets
└── tests/                       # Unit tests
```

## Troubleshooting

- **GPU Issues**: Ensure correct CUDA version is installed.
- **Memory Issues**: Use streaming mode or reduce chunk size.
- **Column Detection Issues**: Specify column names explicitly using `--smiles_col` and `--id_col`.

## Benchmarks

| Dataset Size | CPU Time | GPU Time |
|--------------|----------|----------|
| 1K molecules | 2-5 min  | 30-60 sec|
| 10K molecules| 1-2 hrs  | 10-15 min|
| 50K molecules| 6-12 hrs | 1-2 hrs  |

## Tutorials

### Basic Workflow

1. Prepare your data in CSV format.
2. Run clustering using the appropriate script and parameters.
3. Analyze the output for insights into molecular clustering.

## Parameters Explained

### Common Arguments (Both Tools)
- `--input`, `-i`: Input CSV file containing SMILES.
- `--output`, `-o`: Output CSV file for results.
- `--smiles_col`: Name of SMILES column (default: 'smiles').
- `--id_col`: Name of molecule ID column (default: 'molecule_id').
- `--radius`, `-r`: Morgan fingerprint radius (default: 2).
- `--n_bits`, `-b`: Fingerprint bit size (default: 2048).
- `--cuda`, `--gpu`: Use GPU acceleration.
- `--verbose`, `-v`: Enable verbose output.

### Distance-based Clustering (`run_clustering.py`)
- `--threshold`, `-t`: Distance threshold (default: 0.3, similarity ≥ 0.7).
- `--force-full-clustering`: Force full clustering without sampling.
- `--sample-size`: Sample size for large datasets (default: 10000).
- `--chunk-size`: Chunk size for memory management (default: 1000).
- `--two-stage`: Use two-stage clustering.
- `--reference-file`: Reference molecules for two-stage clustering.

### Similarity-based Clustering (`run_simclustering.py`)
- `--sim_threshold`, `-t`: Similarity threshold(s) (default: [0.6, 0.7]).
- `--streaming`: Use memory-efficient streaming mode.
- `--force_full_matrix`: Force full matrix calculation.
- `--max_molecules`, `-m`: Limit to first N molecules for testing.
- `--random_sample`: Use random sampling instead of first N molecules.

## Input Data Format

Your CSV file should contain at minimum:

- **SMILES column**: Molecular structures in SMILES format.
- **ID column**: Unique identifier for each molecule.

### Example Usage

```bash
--input ExampleL.csv: Your input file
--output Example_Clustered.csv: Output file name
--sim_threshold 0.6 0.7: Run both thresholds and combine results
--smiles_col "SMILES": Column name for SMILES strings
--id_col "Molecule ID": Column name for molecule identifiers
--cuda: Use GPU acceleration
--verbose: Show detailed progress
```
