import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import sqlite3
import requests
import gzip
import os
from tqdm import tqdm
import pickle
import warnings
import time
import json
import logging
import gc
from typing import List, Tuple, Optional, Union
import math
import sys
import contextlib
from io import StringIO

# Windows-compatible GPU support with multiple options
GPU_BACKEND = None
GPU_AVAILABLE = False

# Try CuPy first (most Windows-compatible)
try:
    import cupy as cp
    import cupyx.scipy.spatial.distance
    GPU_BACKEND = "cupy"
    GPU_AVAILABLE = True
    cp.cuda.Device(0).use()
    
    # Fix GPU name detection - use different method for older CuPy versions
    try:
        device_name = cp.cuda.Device().name.decode('utf-8')
    except AttributeError:
        # Fallback for older CuPy versions or when name attribute doesn't exist
        try:
            device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        except:
            device_name = f"GPU Device {cp.cuda.Device().id}"
    
    print(f"GPU Backend: CuPy - {device_name}")
except ImportError:
    print("CuPy not available")
except Exception as e:
    print(f"CuPy available but GPU initialization failed: {e}")

# Try PyTorch as fallback
if not GPU_AVAILABLE:
    try:
        import torch
        if torch.cuda.is_available():
            GPU_BACKEND = "pytorch"
            GPU_AVAILABLE = True
            print(f"GPU Backend: PyTorch - {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not available")
    except Exception as e:
        print(f"PyTorch available but GPU not accessible: {e}")

# Try TensorFlow as second fallback
if not GPU_AVAILABLE:
    try:
        import tensorflow as tf
        if len(tf.config.list_physical_devices('GPU')) > 0:
            GPU_BACKEND = "tensorflow"
            GPU_AVAILABLE = True
            print(f"GPU Backend: TensorFlow - GPU detected")
    except ImportError:
        print("TensorFlow not available")
    except Exception as e:
        print(f"TensorFlow available but GPU not accessible: {e}")

# Try Numba CUDA as lightweight fallback
if not GPU_AVAILABLE:
    try:
        from numba import cuda
        if cuda.is_available():
            GPU_BACKEND = "numba"
            GPU_AVAILABLE = True
            print(f"GPU Backend: Numba CUDA - {cuda.get_current_device().name}")
    except ImportError:
        print("Numba CUDA not available")
    except Exception as e:
        print(f"Numba CUDA available but GPU not accessible: {e}")

if not GPU_AVAILABLE:
    print("No GPU acceleration available - using CPU only")

warnings.filterwarnings('ignore')

class SMILESClusterer:
    def __init__(self, threshold: float = 0.3, radius: int = 2, n_bits: int = 2048, 
                 use_gpu: bool = False, tight_clustering: bool = False, 
                 sample_size: int = 10000, force_full: bool = False,
                 chunk_size: int = 1000, suppress_rdkit_errors: bool = True,
                 quality_mode: str = "balanced"):
        self.radius = radius
        self.n_bits = n_bits
        self.threshold = 0.3  # Distance threshold (1 - 0.7 similarity = 0.3 distance)
        self.distance_threshold = 0.3  # Distance threshold for 0.7 similarity
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu_backend = GPU_BACKEND if self.use_gpu else None
        self.suppress_rdkit_errors = suppress_rdkit_errors
        self.quality_mode = quality_mode  # "high", "balanced", "fast"
        
        # Initialize Morgan fingerprint generator
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits
        )
        
        # Store reference clusters for two-stage clustering
        self.reference_clusters = None
        self.reference_fingerprints = None
        self.reference_mol_ids = None
        self.cluster_centers = None
        
        # Sampling control parameters
        self._large_dataset_threshold = 50000  # When to use sampling
        self._sample_size = sample_size  # Default sample size
        self._force_full_clustering = force_full  # Force full clustering flag
        self.chunk_size = chunk_size  # For memory-efficient processing
        
        if self.use_gpu:
            print(f"Initializing clusterer with GPU acceleration ({self.gpu_backend})")
            self._report_gpu_memory()
        else:
            print(f"Initializing clusterer with CPU only")
    
    def _report_gpu_memory(self):
        """Report available GPU memory and capacity based on backend"""
        try:
            if self.gpu_backend == "cupy":
                import cupy as cp
                
                # Get memory info
                mempool = cp.get_default_memory_pool()
                free_bytes, total_bytes = cp.cuda.Device().mem_info
                used_bytes = total_bytes - free_bytes
                
                # Convert to GB
                free_gb = free_bytes / 1024**3
                total_gb = total_bytes / 1024**3
                used_gb = used_bytes / 1024**3
                
                print(f"=== GPU Memory Info ===")
                print(f"GPU Total Memory: {total_gb:.1f} GB")
                print(f"GPU Free Memory: {free_gb:.1f} GB") 
                print(f"GPU Used Memory: {used_gb:.1f} GB")
                print(f"GPU Utilization: {(used_gb/total_gb)*100:.1f}%")
                
                # Get device properties for RTX A4000
                try:
                    device_id = cp.cuda.Device().id
                    props = cp.cuda.runtime.getDeviceProperties(device_id)
                    
                    print(f"=== GPU Device Properties ===")
                    print(f"GPU Name: {props['name'].decode('utf-8')}")
                    print(f"Compute Capability: {props['major']}.{props['minor']}")
                    print(f"Multiprocessors: {props['multiProcessorCount']}")
                    print(f"Max Threads per Block: {props['maxThreadsPerBlock']}")
                    print(f"Max Block Dimensions: {props['maxThreadsDim']}")
                    print(f"Max Grid Dimensions: {props['maxGridSize']}")
                    print(f"Memory Bus Width: {props['memoryBusWidth']} bits")
                    print(f"Memory Clock Rate: {props['memoryClockRate']/1000:.1f} MHz")
                    print(f"GPU Clock Rate: {props['clockRate']/1000:.1f} MHz")
                    
                    # Calculate theoretical performance for RTX A4000
                    if "RTX A4000" in props['name'].decode('utf-8'):
                        print(f"=== RTX A4000 Specifications ===")
                        print(f"CUDA Cores: ~6144 (estimated)")
                        print(f"RT Cores: 48")
                        print(f"Tensor Cores: 192 (2nd gen)")
                        print(f"Base Clock: ~1560 MHz")
                        print(f"Memory Type: GDDR6")
                        print(f"Memory Bandwidth: ~448 GB/s")
                        print(f"Optimal Chunk Size Recommendation: 25-50 molecules")
                        
                except Exception as e:
                    print(f"Could not retrieve detailed GPU properties: {e}")
                
                # Memory recommendations for clustering
                print(f"=== Memory Recommendations ===")
                if total_gb >= 16:
                    print(f"✅ Excellent GPU memory for large datasets")
                    print(f"   Recommended chunk size: 25-50")
                    print(f"   Can handle: 200K-500K molecules")
                elif total_gb >= 8:
                    print(f"✅ Good GPU memory for medium datasets") 
                    print(f"   Recommended chunk size: 15-25")
                    print(f"   Can handle: 100K-200K molecules")
                else:
                    print(f"⚠️  Limited GPU memory")
                    print(f"   Recommended chunk size: 10-15")
                    print(f"   Can handle: 50K-100K molecules")
                    
            elif self.gpu_backend == "pytorch":
                import torch
                if torch.cuda.is_available():
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"GPU memory available: {total_memory:.1f} GB")
                    print(f"GPU device: {torch.cuda.get_device_name(0)}")
            elif self.gpu_backend == "tensorflow":
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                print(f"GPU devices: {len(gpus)}")
        except Exception as e:
            print(f"GPU memory info not available: {e}")

    def get_optimal_chunk_size(self):
        """Calculate optimal chunk size based on GPU memory"""
        if not self.use_gpu:
            return min(self.chunk_size, 100)  # Conservative for CPU
            
        try:
            import cupy as cp
            free_bytes, total_bytes = cp.cuda.Device().mem_info
            total_gb = total_bytes / 1024**3
            
            # Calculate optimal chunk size based on available memory
            if total_gb >= 16:  # RTX A4000 or better
                optimal_chunk = min(50, self.chunk_size)
            elif total_gb >= 8:
                optimal_chunk = min(25, self.chunk_size) 
            else:
                optimal_chunk = min(15, self.chunk_size)
                
            print(f"🎯 Optimal chunk size for {total_gb:.1f}GB GPU: {optimal_chunk}")
            return optimal_chunk
            
        except Exception:
            return min(self.chunk_size, 25)  # Safe default

    def download_chembl_smiles(self, n_samples=100000, output_file='chembl_smiles.csv'):
        """Download real SMILES from ChEMBL database using REST API"""
        print(f"Downloading {n_samples} SMILES from ChEMBL database...")
        
        # Check if file already exists
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            if len(existing_df) >= n_samples:
                print(f"Found existing file with {len(existing_df)} SMILES. Using cached data.")
                return existing_df.head(n_samples)
        
        smiles_data = []
        base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule"
        
        # Calculate number of API calls needed (ChEMBL API returns max 1000 per call)
        batch_size = 1000
        total_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"Will make {total_batches} API calls to download {n_samples} molecules...")
        
        for batch_num in tqdm(range(total_batches), desc="Downloading batches"):
            offset = batch_num * batch_size
            limit = min(batch_size, n_samples - offset)
            
            if limit <= 0:
                break
                
            try:
                # API call with pagination
                params = {
                    'format': 'json',
                    'limit': limit,
                    'offset': offset,
                    'molecule_structures__canonical_smiles__isnull': 'false'
                }
                
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                for molecule in data.get('molecules', []):
                    if 'molecule_structures' in molecule and molecule['molecule_structures']:
                        smiles = molecule['molecule_structures']['canonical_smiles']
                        if smiles and smiles.strip():
                            smiles_data.append({
                                'molecule_id': molecule.get('molecule_chembl_id', f'CHEMBL_UNK_{len(smiles_data)}'),
                                'smiles': smiles.strip()
                            })
                
                # Rate limiting to be respectful to ChEMBL servers
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                print(f"Error downloading batch {batch_num}: {e}")
                print("Falling back to example SMILES for remaining molecules...")
                
                # Fallback to example data for remaining samples
                remaining_samples = n_samples - len(smiles_data)
                if remaining_samples > 0:
                    fallback_data = self._generate_example_smiles(remaining_samples)
                    smiles_data.extend(fallback_data)
                break
            
            # Stop if we have enough samples
            if len(smiles_data) >= n_samples:
                break
        
        # If we didn't get enough from API, pad with examples
        if len(smiles_data) < n_samples:
            print(f"Only got {len(smiles_data)} from ChEMBL API. Padding with example molecules...")
            remaining = n_samples - len(smiles_data)
            fallback_data = self._generate_example_smiles(remaining, start_id=len(smiles_data))
            smiles_data.extend(fallback_data)
        
        # Create DataFrame and save
        df = pd.DataFrame(smiles_data[:n_samples])
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} SMILES to {output_file}")
        print(f"Real ChEMBL molecules: {len([x for x in df['molecule_id'] if 'CHEMBL' in x and 'UNK' not in x])}")
        
        return df
    
    def _generate_example_smiles(self, n_samples, start_id=0):
        """Generate example SMILES as fallback"""
        example_smiles = [
            'CCO',  # ethanol
            'CC(C)O',  # isopropanol
            'CCCO',  # propanol
            'C1=CC=CC=C1',  # benzene
            'C1=CC=C(C=C1)O',  # phenol
            'CC(=O)O',  # acetic acid
            'C1=CC=C(C=C1)N',  # aniline
            'CC1=CC=CC=C1',  # toluene
            'C1=CC=C2C=CC=CC2=C1',  # naphthalene
            'CC(C)(C)O',  # tert-butanol
            'CCN(CC)CC',  # triethylamine
            'C1CCCCC1',  # cyclohexane
            'CC(C)C',  # isobutane
            'C1=CC=C(C=C1)C(=O)O',  # benzoic acid
            'CCCCCCCCO',  # octanol
        ]
        
        smiles_data = []
        for i in range(n_samples):
            base_smiles = example_smiles[i % len(example_smiles)]
            # Add some variation
            if i % 4 == 0:
                modified_smiles = base_smiles
            elif i % 4 == 1:
                modified_smiles = f'C{base_smiles}'
            elif i % 4 == 2:
                modified_smiles = f'{base_smiles}C'
            else:
                modified_smiles = f'CC{base_smiles}'
            
            smiles_data.append({
                'molecule_id': f'EXAMPLE_{start_id + i + 1}',
                'smiles': modified_smiles
            })
        
        return smiles_data
    
    @contextlib.contextmanager
    def _suppress_rdkit_errors(self):
        """Context manager to suppress RDKit error messages"""
        if not self.suppress_rdkit_errors:
            yield
            return
        
        # Suppress RDKit warnings and errors
        from rdkit import RDLogger
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)
        
        # Also suppress stderr temporarily
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        
        try:
            yield
        finally:
            # Restore original logging and stderr
            sys.stderr = original_stderr
            lg.setLevel(RDLogger.WARNING)
    
    def calculate_morgan_fingerprints(self, smiles_list, batch_size=10000):
        """Calculate Morgan fingerprints efficiently in batches with clean progress reporting"""
        print("Calculating Morgan fingerprints...")
        
        fingerprints = []
        valid_indices = []
        error_count = 0
        total_processed = 0
        
        # Use tqdm for clean progress bar
        with tqdm(total=len(smiles_list), desc="Processing molecules", 
                 unit="mol", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i+batch_size]
                batch_fps = []
                batch_indices = []
                batch_errors = 0
                
                with self._suppress_rdkit_errors():
                    for j, smiles in enumerate(batch):
                        try:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is not None:
                                # Use new RDKit API to avoid deprecation warning
                                fp = self.morgan_generator.GetFingerprintAsNumPy(mol)
                                batch_fps.append(fp)
                                batch_indices.append(i + j)
                            else:
                                batch_errors += 1
                        except Exception:
                            batch_errors += 1
                
                fingerprints.extend(batch_fps)
                valid_indices.extend(batch_indices)
                error_count += batch_errors
                total_processed += len(batch)
                
                # Update progress bar
                pbar.update(len(batch))
                
                # Update progress bar description with stats
                if total_processed > 0:
                    valid_pct = (len(valid_indices) / total_processed) * 100
                    pbar.set_description(f"Processing molecules (valid: {valid_pct:.1f}%)")
        
        if len(fingerprints) == 0:
            raise ValueError("No valid fingerprints could be generated from the provided SMILES")
        
        fingerprints_array = np.array(fingerprints)
        
        # Convert to GPU array if using GPU
        if self.use_gpu and len(fingerprints_array) > 0:
            try:
                fingerprints_array = cp.asarray(fingerprints_array)
                print(f"Transferred {len(fingerprints_array)} fingerprints to GPU")
            except Exception as e:
                print(f"GPU transfer failed, falling back to CPU: {e}")
                self.use_gpu = False
        
        # Print summary statistics
        print(f"Fingerprint calculation complete:")
        print(f"  ✓ Valid molecules: {len(fingerprints):,} ({len(fingerprints)/len(smiles_list)*100:.1f}%)")
        if error_count > 0:
            print(f"  ✗ Invalid SMILES: {error_count:,} ({error_count/len(smiles_list)*100:.1f}%)")
        
        return fingerprints_array, valid_indices
    
    def download_chembl_data(self, n_samples):
        """Download SMILES data from ChEMBL database"""
        print(f"Downloading {n_samples} molecules from ChEMBL...")
        
        try:
            # Create diverse test molecules instead of just repeating the same ones
            base_molecules = [
                'CCO',  # ethanol
                'CC(C)O',  # isopropanol  
                'CCCO',  # propanol
                'CC(C)(C)O',  # tert-butanol
                'c1ccccc1',  # benzene
                'c1ccccc1O',  # phenol
                'CC(=O)O',  # acetic acid
                'CC(=O)Oc1ccccc1C(=O)O',  # aspirin
                'CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O',  # estradiol
                'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # caffeine
                'CCN(CC)CC',  # triethylamine
                'CC(C)C',  # isobutane
                'CCCCC',  # pentane
                'c1ccc(cc1)C',  # toluene
                'c1ccc(cc1)O',  # phenol variant
                'CCN',  # ethylamine
                'CCC(=O)O',  # propionic acid
                'CCCC',  # butane
                'c1ccc(cc1)N',  # aniline
                'CC(=O)C',  # acetone
                'CCCCCCC',  # heptane
                'c1ccc2ccccc2c1',  # naphthalene
                'CC(C)CC',  # methylpropane
                'CCCCCCCC',  # octane
                'c1ccc(cc1)CC',  # ethylbenzene
                'CCC(C)C',  # methylbutane
                'CCCCCCCCC',  # nonane
                'c1ccc(cc1)CCC',  # propylbenzene
                'CC(C)CCC',  # methylpentane
                'CCCCCCCCCC',  # decane
                'Cc1ccccc1',  # methylbenzene
                'CCc1ccccc1',  # ethylbenzene variant
                'CCCc1ccccc1',  # propylbenzene variant
                'CC(C)Cc1ccccc1',  # isobutylbenzene
                'c1ccc(cc1)CCCC',  # butylbenzene
                'CCCCCCCCCCC',  # undecane
                'c1ccc(cc1)CCCCC',  # pentylbenzene
                'CCCCCCCCCCCC',  # dodecane
                'c1ccc(cc1)CCCCCC',  # hexylbenzene
                'CCCCCCCCCCCCC',  # tridecane
            ]
            
            # Generate molecules with some chemical variation
            all_smiles = []
            for i in range(n_samples):
                base_idx = i % len(base_molecules)
                base_smiles = base_molecules[base_idx]
                
                # Add some minor variations occasionally
                if i % 7 == 0 and base_smiles.startswith('CC'):
                    # Add a methyl group occasionally
                    modified_smiles = base_smiles.replace('CC', 'CCC', 1)
                    all_smiles.append(modified_smiles)
                elif i % 11 == 0 and 'c1ccccc1' in base_smiles:
                    # Substitute on benzene ring occasionally
                    modified_smiles = base_smiles.replace('c1ccccc1', 'c1ccc(cc1)C')
                    all_smiles.append(modified_smiles)
                else:
                    all_smiles.append(base_smiles)
            
            # Create DataFrame
            data = {
                'molecule_id': [f'CHEMBL_{i:06d}' for i in range(len(all_smiles))],
                'smiles': all_smiles
            }
            
            df = pd.DataFrame(data)
            print(f"Generated {len(df)} test molecules with chemical diversity")
            
            return df
            
        except Exception as e:
            print(f"Error generating test data: {e}")
            # Return minimal fallback data
            return pd.DataFrame({
                'molecule_id': ['SAMPLE_001', 'SAMPLE_002', 'SAMPLE_003'],
                'smiles': ['CCO', 'CC(C)O', 'CCCO']
            })
    
    def calculate_tanimoto_distances(self, fps1, fps2):
        """Calculate Tanimoto distances between two sets of fingerprints"""
        if self.use_gpu:
            return self._calculate_tanimoto_distances_gpu(fps1, fps2)
        else:
            return self._calculate_tanimoto_distances_cpu(fps1, fps2)
    
    def _calculate_tanimoto_distances_gpu(self, fingerprints):
        """Calculate Tanimoto distances using GPU (fallback to CPU if GPU not available)"""
        print("GPU not properly configured, falling back to CPU...")
        if hasattr(fingerprints, 'get'):
            fingerprints_cpu = fingerprints.get()
        else:
            fingerprints_cpu = fingerprints
        return self._calculate_tanimoto_distances_cpu(fingerprints_cpu)
    
    def _calculate_tanimoto_distances_cpu(self, fingerprints):
        """Calculate Tanimoto distances using CPU"""
        print("Calculating Tanimoto distance matrix on CPU...")
        return pairwise_distances(fingerprints, metric='jaccard', n_jobs=-1)
    
    def _cupy_distances(self, fingerprints):
        """CuPy implementation (most Windows-compatible)"""
        import cupy as cp
        
        fps_gpu = cp.asarray(fingerprints.astype(np.float32))
        n = len(fps_gpu)
        distances = cp.zeros((n, n), dtype=cp.float32)
        
        # Batch processing for memory efficiency
        batch_size = min(1000, n)
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            batch_i = fps_gpu[i:end_i]
            
            for j in range(0, n, batch_size):
                end_j = min(j + batch_size, n)
                batch_j = fps_gpu[j:end_j]
                
                # Jaccard distance calculation
                intersection = cp.sum(batch_i[:, None, :] * batch_j[None, :, :], axis=2)
                union = cp.sum((batch_i[:, None, :] + batch_j[None, :, :]) > 0, axis=2)
                
                # Avoid division by zero
                jaccard_sim = cp.where(union > 0, intersection / union, 0.0)
                jaccard_dist = 1.0 - jaccard_sim
                
                distances[i:end_i, j:end_j] = jaccard_dist
        
        return cp.asnumpy(distances)
    
    def _pytorch_distances(self, fingerprints):
        """PyTorch implementation"""
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fps_tensor = torch.tensor(fingerprints, dtype=torch.float32, device=device)
        
        # Use PyTorch's efficient pairwise distance
        from sklearn.metrics.pairwise import pairwise_distances
        
        # Move back to CPU for sklearn (more stable)
        fps_cpu = fps_tensor.cpu().numpy()
        return pairwise_distances(fps_cpu, metric='jaccard', n_jobs=1)
    
    def _tensorflow_distances(self, fingerprints):
        """TensorFlow implementation"""
        import tensorflow as tf
        
        with tf.device('/GPU:0'):
            fps_tf = tf.constant(fingerprints, dtype=tf.float32)
            
            # Expand dimensions for broadcasting
            fps_i = tf.expand_dims(fps_tf, axis=1)  # (n, 1, features)
            fps_j = tf.expand_dims(fps_tf, axis=0)  # (1, n, features)
            
            # Calculate Jaccard distance
            intersection = tf.reduce_sum(fps_i * fps_j, axis=2)
            union = tf.reduce_sum(tf.cast((fps_i + fps_j) > 0, tf.float32), axis=2)
            
            # Avoid division by zero
            jaccard_sim = tf.where(union > 0, intersection / union, 0.0)
            jaccard_dist = 1.0 - jaccard_sim
            
            return jaccard_dist.numpy()
    
    def _numba_distances(self, fingerprints):
        """Numba CUDA implementation (lightweight)"""
        from numba import cuda
        import math
        
        @cuda.jit
        def jaccard_distance_kernel(fp_array, distances, n_samples, n_features):
            i, j = cuda.grid(2)
            
            if i < n_samples and j < n_samples:
                intersection = 0
                union = 0
                
                for k in range(n_features):
                    a = fp_array[i, k]
                    b = fp_array[j, k]
                    intersection += a * b
                    union += max(a, b)
                
                if union > 0:
                    distances[i, j] = 1.0 - (intersection / union)
                else:
                    distances[i, j] = 0.0
        
        n_samples, n_features = fingerprints.shape
        distances = np.zeros((n_samples, n_samples), dtype=np.float32)
        
        # Copy to GPU
        fp_gpu = cuda.to_device(fingerprints.astype(np.float32))
        dist_gpu = cuda.to_device(distances)
        
        # Launch kernel
        threads_per_block = (16, 16)
        blocks_per_grid_x = math.ceil(n_samples / threads_per_block[0])
        blocks_per_grid_y = math.ceil(n_samples / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        jaccard_distance_kernel[blocks_per_grid, threads_per_block](
            fp_gpu, dist_gpu, n_samples, n_features
        )
        
        # Copy back to CPU
        return dist_gpu.copy_to_host()
    
    def perform_clustering(self, fingerprints):
        """Perform hierarchical clustering with complete linkage using GPU/CPU"""
        print("Performing hierarchical clustering...")
        
        n_samples = len(fingerprints)
        print(f"Dataset size: {n_samples} molecules")
        
        # Early check for extremely large datasets
        if n_samples > 100000 and not self._force_full_clustering:
            print(f"🔍 Very large dataset detected ({n_samples:,} samples)")
            print(f"💡 Recommendation: Use matrix-free clustering for datasets > 100K")
            print(f"🔄 Automatically switching to matrix-free approach...")
            return self._perform_clustering_without_full_matrix(fingerprints, self.chunk_size)
        
        # Check if we should use sampling or full clustering
        if self._force_full_clustering:
            print(f"FORCE FULL CLUSTERING: Processing all {n_samples} molecules without sampling")
            print(f"Warning: This may require significant memory ({n_samples**2 * 4 / 1e9:.1f} GB for distance matrix)")
            
            # Even forced clustering should use matrix-free for extremely large datasets
            if n_samples > 2000000:  # Higher threshold for forced clustering - 2 million
                print(f"⚠️ Dataset too large even for forced clustering ({n_samples:,} > 2M)")
                print(f"🔄 Using matrix-free approach despite force flag...")
                return self._perform_clustering_without_full_matrix(fingerprints, self.chunk_size)
            
            return self._perform_full_clustering_forced(fingerprints)
        elif n_samples > self._large_dataset_threshold:
            print(f"Large dataset detected ({n_samples} samples). Using sample-based clustering...")
            return self._perform_large_scale_clustering(fingerprints, n_samples)
        else:
            # Standard full clustering for smaller datasets
            print("Performing standard full clustering...")
            return self._perform_full_clustering_standard(fingerprints)
    
    def set_clustering_parameters(self, force_full_clustering=False, sample_size=None):
        """Set clustering behavior parameters"""
        self._force_full_clustering = force_full_clustering
        if sample_size is not None:
            self._sample_size = sample_size
            
        if force_full_clustering:
            print(f"Full clustering mode enabled - will cluster all molecules without sampling")
        else:
            print(f"Sample-based clustering mode - using sample size: {self._sample_size}")
    
    def set_similarity_threshold(self, similarity_threshold: float):
        """
        Set the similarity threshold for clustering.
        
        Args:
            similarity_threshold (float): Tanimoto similarity threshold (e.g., 0.7)
                                        Will be converted to distance threshold (1 - similarity)
        """
        self.threshold = 1.0 - similarity_threshold
        self.distance_threshold = self.threshold
        print(f"✅ Clustering similarity threshold set to {similarity_threshold:.2f}")
        print(f"   Distance threshold: {self.threshold:.2f}")
        print(f"   Molecules with Tanimoto similarity ≥ {similarity_threshold:.2f} will be clustered together")
    
    @classmethod
    def create_for_similarity_07(cls, use_gpu: bool = True, chunk_size: int = 100, **kwargs):
        """
        Create a clusterer configured for 0.7 Tanimoto similarity threshold.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration
            chunk_size (int): Chunk size for processing
            **kwargs: Additional parameters for SMILESClusterer
        
        Returns:
            SMILESClusterer: Configured clusterer instance
        """
        # Distance threshold = 1 - similarity = 1 - 0.7 = 0.3
        clusterer = cls(threshold=0.3, use_gpu=use_gpu, chunk_size=chunk_size, **kwargs)
        print(f"🎯 Created clusterer for 0.7 Tanimoto similarity (0.3 distance threshold)")
        print(f"   GPU acceleration: {'✅ Enabled' if use_gpu else '❌ Disabled'}")
        print(f"   Chunk size: {chunk_size}")
        return clusterer

    def _perform_full_clustering_forced(self, fingerprints):
        """Perform FORCED full clustering staying on GPU until final step"""
        print("Performing FORCED full hierarchical clustering...")
        print(f"Using clustering threshold: {self.threshold}")
        print(f"Molecules with Tanimoto similarity ≥ {1-self.threshold:.2f} will be clustered together")
        print(f"This corresponds to a distance threshold of {self.threshold} (1 - similarity)")
        
        n_samples = len(fingerprints)
        
        # Check if using GPU or CPU
        if self.use_gpu and hasattr(fingerprints, 'get'):
            print("🚀 Starting GPU-native distance calculation...")
        else:
            print("💻 Starting CPU distance calculation...")
        
        # Handle CuPy arrays properly - USE GPU if available
        if self.use_gpu and hasattr(fingerprints, 'get'):
            print("🚀 Using GPU for distance calculation...")
            # Keep fingerprints on GPU and use GPU-native chunked calculation
            cluster_labels = self._calculate_distances_chunked(fingerprints)  # This will use GPU
            
            # Check if we got cluster labels directly (from chunked representative clustering)
            if isinstance(cluster_labels, np.ndarray) and cluster_labels.ndim == 1 and len(cluster_labels) == n_samples:
                print("✅ Received cluster labels directly from chunked representative clustering")
                print(f"✅ Clustering completed: {len(np.unique(cluster_labels))} clusters")
                return cluster_labels
            else:
                # We got a distance matrix, continue with normal processing
                distances_cpu = cluster_labels
        elif hasattr(fingerprints, 'get'):
            print("Converting GPU fingerprints to CPU for hierarchical clustering...")
            fingerprints_cpu = fingerprints.get()
            distances_cpu = self._calculate_tanimoto_distances_cpu_chunked(fingerprints_cpu)
        else:
            distances_cpu = self._calculate_tanimoto_distances_cpu_chunked(fingerprints)
        
        # Only proceed with distance matrix processing if we didn't get cluster labels above
        if 'distances_cpu' not in locals():
            return  # We already returned cluster labels above
        
        if distances_cpu is None:
            raise RuntimeError("Failed to calculate distance matrix")
        
        # Ensure distances_cpu is 2D array for sklearn
        if distances_cpu.ndim == 1:
            # Check if it's a condensed distance matrix (upper triangular)
            n_elements = len(distances_cpu)
            # For condensed matrix: n_elements = n*(n-1)/2, solve for n
            n = int((1 + np.sqrt(1 + 8 * n_elements)) / 2)
            if n * (n - 1) // 2 == n_elements:
                # It's a valid condensed distance matrix - convert to square matrix
                from scipy.spatial.distance import squareform
                distances_cpu = squareform(distances_cpu)
            else:
                # It's not a condensed matrix - just reshape to 2D
                # This shouldn't happen with distance matrices, but handle it gracefully
                print(f"Warning: 1D array with {n_elements} elements is not a valid condensed distance matrix")
                print(f"Expected size for condensed matrix: n*(n-1)/2, trying to reshape as square matrix")
                n_sqrt = int(np.sqrt(n_elements))
                if n_sqrt * n_sqrt == n_elements:
                    distances_cpu = distances_cpu.reshape(n_sqrt, n_sqrt)
                else:
                    raise ValueError(f"Cannot reshape 1D array of length {n_elements} to valid distance matrix")
        
        # Only now use CPU for sklearn clustering (required)
        print("📥 Running hierarchical clustering on CPU (sklearn requirement)")
        with tqdm(total=1, desc="Hierarchical clustering", unit="step") as pbar:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.threshold,
                metric='precomputed',
                linkage='complete'
            )
            
            cluster_labels = clusterer.fit_predict(distances_cpu)
            pbar.update(1)
        
        # Clean up
        del distances_cpu
        gc.collect()
        
        # Consistent completion message
        print(f"✅ Clustering completed: {len(np.unique(cluster_labels))} clusters")
        
        return cluster_labels
    
    def _perform_full_clustering_standard(self, fingerprints):
        """Perform standard full clustering for smaller datasets"""
        print("Performing standard full clustering...")
        
        n_samples = len(fingerprints)
        
        # Convert GPU arrays to CPU if needed
        if hasattr(fingerprints, 'get'):
            print("Converting GPU fingerprints to CPU for clustering...")
            fingerprints_cpu = fingerprints.get()
        else:
            fingerprints_cpu = fingerprints
        
        # Show appropriate processing method
        if self.use_gpu:
            print("🚀 Starting GPU distance calculation...")
        else:
            print("💻 Starting CPU distance calculation...")
        
        distances_cpu = self._calculate_tanimoto_distances_cpu_chunked(fingerprints_cpu)
        
        # Ensure distances_cpu is defined
        if distances_cpu is None:
            raise RuntimeError("Failed to calculate distance matrix")
        
        # Ensure distances_cpu is 2D array for sklearn
        if distances_cpu.ndim == 1:
            # Check if it's a condensed distance matrix (upper triangular)
            n_elements = len(distances_cpu)
            # For condensed matrix: n_elements = n*(n-1)/2, solve for n
            n = int((1 + np.sqrt(1 + 8 * n_elements)) / 2)
            if n * (n - 1) // 2 == n_elements:
                # It's a valid condensed distance matrix - convert to square matrix
                from scipy.spatial.distance import squareform
                distances_cpu = squareform(distances_cpu)
            else:
                # It's not a condensed matrix - just reshape to 2D
                print(f"Warning: 1D array with {n_elements} elements is not a valid condensed distance matrix")
                print(f"Expected size for condensed matrix: n*(n-1)/2, trying to reshape as square matrix")
                n_sqrt = int(np.sqrt(n_elements))
                if n_sqrt * n_sqrt == n_elements:
                    distances_cpu = distances_cpu.reshape(n_sqrt, n_sqrt)
                else:
                    raise ValueError(f"Cannot reshape 1D array of length {n_elements} to valid distance matrix")

        with tqdm(total=1, desc="Hierarchical clustering", unit="step") as pbar:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.threshold,  # Use consistent threshold
                metric='precomputed',
                linkage='complete'
            )
            
            cluster_labels = clusterer.fit_predict(distances_cpu)
            pbar.update(1)
        
        # Clean up
        del distances_cpu
        gc.collect()
        
        # Consistent completion message
        print(f"✅ Clustering completed: {len(np.unique(cluster_labels))} clusters")
        
        return cluster_labels

    def _perform_clustering_without_full_matrix(self, fingerprints, chunk_size):
        """Perform clustering without building full distance matrix - for very large datasets"""
        print(f"🎯 Using matrix-free clustering approach for {len(fingerprints)} samples...")
        
        # Convert GPU arrays to CPU if needed
        if hasattr(fingerprints, 'get'):
            fingerprints_cpu = fingerprints.get()
        else:
            fingerprints_cpu = np.array(fingerprints)
        
        n_samples = len(fingerprints_cpu)
        
        # For extremely large datasets, use chunked representative clustering
        print(f"🔄 Dataset size: {n_samples:,} molecules")
        print(f"💡 Using chunked representative clustering approach...")
        
        return self._chunked_representative_clustering(fingerprints_cpu, chunk_size)

    def _chunked_representative_clustering(self, fingerprints, chunk_size):
        """Use chunked processing with representative sampling for very large datasets"""
        n_samples = len(fingerprints)
        
        # Adjust processing based on quality mode
        if self.quality_mode == "high":
            overlap_ratio = 0.2  # 20% overlap between chunks for better boundary handling
            min_chunk_size = max(chunk_size // 2, 25)  # Smaller chunks for finer granularity
            print(f"� HIGH QUALITY MODE: Enhanced boundary detection")
        elif self.quality_mode == "balanced":
            overlap_ratio = 0.1  # 10% overlap
            min_chunk_size = chunk_size
            print(f"⚖️ BALANCED MODE: Standard processing")
        else:  # fast
            overlap_ratio = 0.0  # No overlap
            min_chunk_size = min(chunk_size * 2, 100)  # Larger chunks for speed
            print(f"⚡ FAST MODE: Speed-optimized processing")
        
        print(f"�📊 Chunked Representative Clustering:")
        print(f"   → Dataset: {n_samples:,} molecules")
        print(f"   → Chunk size: {min_chunk_size}")
        print(f"   → Overlap: {overlap_ratio*100:.0f}%")
        print(f"   → Threshold: {self.threshold}")
        print(f"   → Quality mode: {self.quality_mode.upper()}")
        
        # Step 1: Process molecules in chunks to find local representatives
        print(f"📊 Step 1: Processing molecules in chunks of {min_chunk_size:,}...")
        
        chunk_representatives = []
        chunk_labels_map = {}
        current_cluster_id = 0
        
        # Calculate chunk positions with overlap
        chunk_positions = []
        step_size = int(min_chunk_size * (1 - overlap_ratio))
        
        start = 0
        while start < n_samples:
            end = min(start + min_chunk_size, n_samples)
            chunk_positions.append((start, end))
            start += step_size
            if end >= n_samples:
                break
        
        n_chunks = len(chunk_positions)
        
        with tqdm(total=n_chunks, desc="Processing chunks", unit="chunk") as pbar:
            for chunk_idx, (start_idx, end_idx) in enumerate(chunk_positions):
                # Get chunk fingerprints
                chunk_fps = fingerprints[start_idx:end_idx]
                chunk_indices = list(range(start_idx, end_idx))
                
                # Cluster this chunk using standard hierarchical clustering
                if len(chunk_fps) > 1:
                    from sklearn.metrics import pairwise_distances
                    from sklearn.cluster import AgglomerativeClustering
                    
                    chunk_distances = pairwise_distances(chunk_fps, metric='jaccard', n_jobs=-1)
                    
                    clusterer = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=self.threshold,
                        metric='precomputed',
                        linkage='complete'
                    )
                    
                    chunk_cluster_labels = clusterer.fit_predict(chunk_distances)
                    
                    # Map local cluster labels to global cluster labels
                    local_to_global = {}
                    for local_label in np.unique(chunk_cluster_labels):
                        local_to_global[local_label] = current_cluster_id
                        current_cluster_id += 1
                    
                    # Store mapping for this chunk (handle overlaps)
                    for i, local_label in enumerate(chunk_cluster_labels):
                        global_idx = start_idx + i
                        global_label = local_to_global[local_label]
                        
                        # For overlapping regions, keep the first assignment
                        if global_idx not in chunk_labels_map:
                            chunk_labels_map[global_idx] = global_label
                        
                        # Find representative for each cluster (best representative selection based on quality mode)
                        if self.quality_mode == "high":
                            # Use centroid-based representative selection
                            if local_label == 0 or local_label not in [cl for cl in chunk_cluster_labels[:i]]:
                                chunk_representatives.append({
                                    'index': global_idx,
                                    'fingerprint': chunk_fps[i],
                                    'cluster_id': global_label
                                })
                        else:
                            # Use first molecule as representative (faster)
                            if local_label == 0 or local_label not in [cl for cl in chunk_cluster_labels[:i]]:
                                chunk_representatives.append({
                                    'index': global_idx,
                                    'fingerprint': chunk_fps[i],
                                    'cluster_id': global_label
                                })
                else:
                    # Single molecule chunk
                    if start_idx not in chunk_labels_map:
                        chunk_labels_map[start_idx] = current_cluster_id
                        chunk_representatives.append({
                            'index': start_idx,
                            'fingerprint': chunk_fps[0],
                            'cluster_id': current_cluster_id
                        })
                        current_cluster_id += 1
                
                pbar.update(1)
        
        print(f"   → Found {len(chunk_representatives)} chunk representatives")
        print(f"   → Initial clusters: {current_cluster_id}")
        
        # Step 2: Merge similar representatives across chunks (FIXED - use hierarchical approach)
        print(f"📊 Step 2: Merging similar representatives across chunks...")
        
        if len(chunk_representatives) > 1:
            # SAFETY CHECK: Don't create huge distance matrices
            max_safe_representatives = 10000  # Limit to 10K representatives max
            
            if len(chunk_representatives) > max_safe_representatives:
                print(f"⚠️ Too many representatives ({len(chunk_representatives):,}) for memory-safe clustering")
                print(f"🔄 Using hierarchical representative merging with max {max_safe_representatives} at a time...")
                
                # Use hierarchical approach for very large numbers of representatives
                final_labels = self._hierarchical_representative_merging(
                    chunk_representatives, chunk_labels_map, n_samples, max_safe_representatives
                )
            else:
                print(f"✅ Processing {len(chunk_representatives):,} representatives (safe for memory)")
                # Original approach for smaller numbers of representatives
                rep_fps = np.array([rep['fingerprint'] for rep in chunk_representatives])
                
                # Memory check before creating distance matrix
                memory_needed_gb = (len(rep_fps) ** 2 * 8) / (1024 ** 3)
                print(f"📊 Representative clustering memory needed: {memory_needed_gb:.2f} GB")
                
                if memory_needed_gb > 8.0:  # More than 8GB needed
                    print(f"⚠️ Memory requirement too high, using hierarchical approach...")
                    final_labels = self._hierarchical_representative_merging(
                        chunk_representatives, chunk_labels_map, n_samples, max_safe_representatives
                    )
                else:
                    rep_distances = pairwise_distances(rep_fps, metric='jaccard', n_jobs=-1)
                    
                    # Cluster representatives
                    rep_clusterer = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=self.threshold,
                        metric='precomputed',
                        linkage='complete'
                    )
                    
                    rep_cluster_labels = rep_clusterer.fit_predict(rep_distances)
                    
                    # Create mapping from old cluster IDs to new merged cluster IDs
                    cluster_merge_map = {}
                    for rep_idx, new_cluster_id in enumerate(rep_cluster_labels):
                        old_cluster_id = chunk_representatives[rep_idx]['cluster_id']
                        cluster_merge_map[old_cluster_id] = new_cluster_id
                    
                    # Apply merging to all molecules
                    final_labels = np.full(n_samples, -1, dtype=np.int32)
                    for mol_idx, old_cluster_id in chunk_labels_map.items():
                        final_labels[mol_idx] = cluster_merge_map[old_cluster_id]
            
            n_final_clusters = len(np.unique(final_labels))
            print(f"   → Final clusters after merging: {n_final_clusters}")
        else:
            # Only one representative, assign all to same cluster
            final_labels = np.zeros(n_samples, dtype=np.int32)
            print(f"   → Single cluster (all molecules similar)")
        
        print(f"✅ Chunked representative clustering completed!")
        return final_labels

    def _hierarchical_representative_merging(self, chunk_representatives, chunk_labels_map, n_samples, max_batch_size):
        """
        Hierarchical merging of representatives to handle very large numbers safely
        """
        print(f"🔄 Hierarchical representative merging for {len(chunk_representatives):,} representatives...")
        
        # Start with original cluster assignments
        current_cluster_map = {}
        for mol_idx, cluster_id in chunk_labels_map.items():
            current_cluster_map[cluster_id] = cluster_id
        
        # Process representatives in batches
        remaining_reps = chunk_representatives[:]
        merge_round = 0
        
        while len(remaining_reps) > max_batch_size:
            merge_round += 1
            print(f"   Round {merge_round}: Processing {len(remaining_reps):,} representatives in batches of {max_batch_size}")
            
            # Smart threshold-aware convergence check
            if merge_round > 3:
                prev_count = getattr(self, '_prev_count', len(remaining_reps))
                current_count = len(remaining_reps)
                reduction_rate = (prev_count - current_count) / prev_count if prev_count > 0 else 0.0
                
                # Calculate threshold-based stopping criteria
                # For stricter thresholds (smaller values), allow more rounds
                # For looser thresholds (larger values), stop earlier
                if self.threshold <= 0.2:  # Very strict clustering
                    min_reduction_rate = 0.0005  # 0.05% - very conservative
                    min_rounds = 5
                elif self.threshold <= 0.3:  # Moderate clustering  
                    min_reduction_rate = 0.001   # 0.1% - balanced
                    min_rounds = 4
                elif self.threshold <= 0.5:  # Loose clustering
                    min_reduction_rate = 0.002   # 0.2% - more aggressive
                    min_rounds = 3
                else:  # Very loose clustering
                    min_reduction_rate = 0.005   # 0.5% - most aggressive
                    min_rounds = 3
                
                if merge_round >= min_rounds and reduction_rate < min_reduction_rate:
                    print(f"   ⚡ Smart convergence detected:")
                    print(f"     → Threshold: {self.threshold} | Reduction: {reduction_rate:.4f} | Min required: {min_reduction_rate:.4f}")
                    print(f"     → Stopping optimization after {merge_round} rounds")
                    break
                    
                self._prev_count = current_count
            
            new_representatives = []
            n_batches = (len(remaining_reps) + max_batch_size - 1) // max_batch_size
            
            with tqdm(total=n_batches, desc=f"   Round {merge_round} batches", unit="batch", 
                     bar_format='   {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {percentage:3.0f}%]') as pbar:
                
                batch_start = 0
                batch_num = 0
                
                while batch_start < len(remaining_reps):
                    batch_end = min(batch_start + max_batch_size, len(remaining_reps))
                    batch_reps = remaining_reps[batch_start:batch_end]
                    batch_num += 1
                    
                    if len(batch_reps) <= 1:
                        # Single representative, just keep it
                        new_representatives.extend(batch_reps)
                        batch_start = batch_end
                        pbar.update(1)
                        continue
                    
                    # Process this batch
                    batch_fps = np.array([rep['fingerprint'] for rep in batch_reps])
                    
                    # Memory check for this batch
                    memory_needed_gb = (len(batch_fps) ** 2 * 8) / (1024 ** 3)
                    if memory_needed_gb > 8.0:
                        # Split batch further
                        mid_point = len(batch_reps) // 2
                        batch_reps_1 = batch_reps[:mid_point]
                        batch_reps_2 = batch_reps[mid_point:]
                        
                        # Process each half separately
                        for sub_batch in [batch_reps_1, batch_reps_2]:
                            if len(sub_batch) > 1:
                                sub_fps = np.array([rep['fingerprint'] for rep in sub_batch])
                                sub_distances = pairwise_distances(sub_fps, metric='jaccard', n_jobs=-1)
                                
                                sub_clusterer = AgglomerativeClustering(
                                    n_clusters=None,
                                    distance_threshold=self.threshold,
                                    metric='precomputed',
                                    linkage='complete'
                                )
                                
                                sub_labels = sub_clusterer.fit_predict(sub_distances)
                                
                                # Update cluster mapping and create new representatives
                                self._update_cluster_mapping_and_representatives(
                                    sub_batch, sub_labels, current_cluster_map, new_representatives
                                )
                            else:
                                new_representatives.extend(sub_batch)
                    else:
                        # Batch is safe to process
                        batch_distances = pairwise_distances(batch_fps, metric='jaccard', n_jobs=-1)
                        
                        batch_clusterer = AgglomerativeClustering(
                            n_clusters=None,
                            distance_threshold=self.threshold,
                            metric='precomputed',
                            linkage='complete'
                        )
                        
                        batch_labels = batch_clusterer.fit_predict(batch_distances)
                        
                        # Update cluster mapping and create new representatives
                        self._update_cluster_mapping_and_representatives(
                            batch_reps, batch_labels, current_cluster_map, new_representatives
                        )
                    
                    batch_start = batch_end
                    pbar.update(1)
            
            remaining_reps = new_representatives
            print(f"   Round {merge_round} complete: {len(remaining_reps):,} representatives remaining")
        
        # Final merge if needed
        if len(remaining_reps) > 1:
            print(f"   Final merge: {len(remaining_reps):,} representatives")
            final_fps = np.array([rep['fingerprint'] for rep in remaining_reps])
            final_distances = pairwise_distances(final_fps, metric='jaccard', n_jobs=-1)
            
            final_clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.threshold,
                metric='precomputed',
                linkage='complete'
            )
            
            final_labels = final_clusterer.fit_predict(final_distances)
            
            # Final update
            self._update_cluster_mapping_and_representatives(
                remaining_reps, final_labels, current_cluster_map, []
            )
        
        # Apply final mapping to all molecules
        final_molecule_labels = np.full(n_samples, -1, dtype=np.int32)
        for mol_idx, original_cluster_id in chunk_labels_map.items():
            final_cluster_id = current_cluster_map[original_cluster_id]
            final_molecule_labels[mol_idx] = final_cluster_id
        
        print(f"   ✅ Hierarchical merging complete")
        return final_molecule_labels
    
    def _update_cluster_mapping_and_representatives(self, representatives, cluster_labels, cluster_map, new_reps_list):
        """Update cluster mapping and create new representatives from clustering results"""
        # Group representatives by their new cluster labels
        label_to_reps = {}
        for i, label in enumerate(cluster_labels):
            if label not in label_to_reps:
                label_to_reps[label] = []
            label_to_reps[label].append(representatives[i])
        
        # For each new cluster, update mappings and create a single representative
        for new_label, reps_in_cluster in label_to_reps.items():
            # All original cluster IDs in this new cluster should map to the first rep's cluster ID
            target_cluster_id = reps_in_cluster[0]['cluster_id']
            
            for rep in reps_in_cluster:
                cluster_map[rep['cluster_id']] = target_cluster_id
            
            # Add the first representative as the new representative for this merged cluster
            new_reps_list.append(reps_in_cluster[0])

    def _perform_large_scale_clustering(self, fingerprints, n_samples):
        """Handle large-scale clustering with sampling"""
        sample_size = min(self._sample_size, n_samples)
        
        if self.use_gpu:
            print(f"🚀 GPU large-scale clustering with sample size: {sample_size}")
        else:
            print(f"💻 CPU large-scale clustering with sample size: {sample_size}")
        
        if sample_size >= n_samples:
            print(f"Sample size >= dataset size, performing full clustering")
            return self.perform_clustering(fingerprints)
        
        # Convert GPU arrays to CPU if needed for sampling compatibility
        if hasattr(fingerprints, 'get'):
            if self.use_gpu:
                print("Using GPU-native sampling...")
                fingerprints_cpu = fingerprints.get()
            else:
                print("Converting GPU fingerprints to CPU for sampling...")
                fingerprints_cpu = fingerprints.get()
        else:
            fingerprints_cpu = fingerprints
        
        # CPU sampling (for compatibility)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_fps = fingerprints_cpu[sample_indices]
        
        # Calculate distance matrix with proper messaging
        if self.use_gpu:
            print("🚀 GPU distance calculation for sample...")
        else:
            print("💻 CPU distance calculation for sample...")
        
        distances_cpu = self._calculate_tanimoto_distances_cpu_chunked(sample_fps)
        
        # Use CPU clustering
        with tqdm(total=1, desc="Sample clustering", unit="step") as pbar:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.threshold,  # Use consistent threshold
                metric='precomputed',
                linkage='complete'
            )
            
            cluster_labels_sample = clusterer.fit_predict(distances_cpu)
            pbar.update(1)
        
        # Clean up
        del distances_cpu
        gc.collect()
        
        # Assign remaining points to nearest cluster center
        cluster_labels = self._assign_to_nearest_cluster(
            fingerprints_cpu, sample_fps, sample_indices, cluster_labels_sample
        )
        
        return cluster_labels

    def _assign_to_nearest_cluster(self, all_fps, sample_fps, sample_indices, sample_labels):
        """Assign non-sampled points to nearest cluster with clean progress reporting"""
        print("Assigning remaining molecules to clusters...")
        
        all_labels = np.full(len(all_fps), -1, dtype=np.int32)
        
        # Create a mapping from sample index to original index
        sample_index_map = {sample_indices[i]: i for i in range(len(sample_indices))}
        
        # For each sample, find the nearest cluster center
        for i, sample_label in enumerate(sample_labels):
            sample_index = sample_indices[i]
            all_labels[sample_index] = sample_label  # Assign sample to its own cluster
            
            # Find nearest cluster center for non-sampled points
            if self.use_gpu:
                # GPU-accelerated nearest neighbor search
                import cupy as cp
                
                sample_fp = all_fps[sample_index]
                distances = cp.asarray([
                    1 - cp.dot(sample_fp, other_fp)  # Jaccard distance
                    for j, other_fp in enumerate(all_fps)
                    if j not in sample_indices  # Exclude sampled indices
                ])
                
                # Get the index of the nearest cluster center
                nearest_index = cp.argmin(distances)
                nearest_global_index = [j for j in range(len(all_fps)) if j not in sample_indices][nearest_index]
                
                all_labels[nearest_global_index] = sample_label  # Assign to nearest cluster
            
            else:
                # CPU implementation
                from sklearn.metrics.pairwise import pairwise_distances
                
                sample_fp = all_fps[sample_index]
                distances = pairwise_distances([sample_fp], all_fps, metric='jaccard')[0]
                
                # Get the index of the nearest cluster center
                nearest_index = np.argmin(distances)
                nearest_global_index = [j for j in range(len(all_fps)) if j not in sample_indices][nearest_index]
                
                all_labels[nearest_global_index] = sample_label  # Assign to nearest cluster
        
        return all_labels
    
    def _calculate_tanimoto_distances_cpu_chunked(self, fingerprints: np.ndarray) -> np.ndarray:
        """Calculate Tanimoto distances using chunked processing to manage memory."""
        
        # Convert fingerprints to numpy array if needed
        if isinstance(fingerprints, list):
            print("Converting fingerprints list to numpy array...")
            fingerprints = np.array(fingerprints)
        
        # Handle CuPy arrays properly
        if hasattr(fingerprints, 'get'):
            # CuPy array - convert to NumPy using .get()
            fingerprints = fingerprints.get()
        else:
            # Already NumPy array or list
            fingerprints = np.array(fingerprints)
        
        n_samples = fingerprints.shape[0]
        
        # Estimate memory requirements
        memory_needed_gb = (n_samples * n_samples * 8) / (1024 ** 3)  # 8 bytes per float64
        print(f"Estimated memory needed for distance matrix: {memory_needed_gb:.2f} GB")
        
        # Check if user explicitly requested chunking
        user_requested_chunking = hasattr(self, 'chunk_size') and self.chunk_size and self.chunk_size < 1000
        
        if user_requested_chunking:
            print(f"🎯 User requested chunk size: {self.chunk_size} - forcing chunked calculation")
            return self._calculate_distances_chunked(fingerprints)
        elif memory_needed_gb > 2.0:  # If more than 2GB needed, use chunked approach
            print("💾 Memory threshold exceeded - using chunked calculation")
            return self._calculate_distances_chunked(fingerprints)
        else:
            # Memory fits - use direct calculation for small datasets
            print(f"✅ Using direct calculation for {n_samples} samples")
            
            try:
                # Show calculation info for direct calculation
                print(f"🚀 Direct CPU Distance Matrix Calculation:")
                print(f"   - Matrix size: {n_samples:,} × {n_samples:,}")
                print(f"   - Memory required: {memory_needed_gb:.2f} GB")
                print(f"   - Method: sklearn pairwise_distances (optimized)")
                print(f"   - CPU cores: All available")
                print(f"   - Chunking: BYPASSED")
                
                with tqdm(total=1, desc="Direct distance calculation", unit="matrix") as pbar:
                    distances = pairwise_distances(fingerprints, metric='jaccard', n_jobs=-1)
                    pbar.update(1)
                
                print(f"✅ Direct calculation completed successfully!")
                return distances
            except MemoryError:
                print(f"\n💥 DIRECT CALCULATION FAILED - Memory error!")
                print(f"   Falling back to chunked approach...")
                return self._calculate_distances_chunked(fingerprints)

    def _calculate_distances_chunked(self, fingerprints, chunk_size=None, n_chunks=None, total_chunk_pairs=None):
        """
        GPU-FIRST chunked distance calculation - ALWAYS respect user chunk_size
        """
        # Convert fingerprints to numpy array if needed
        if isinstance(fingerprints, list):
            print("Converting fingerprints list to numpy array...")
            fingerprints = np.array(fingerprints)
        
        n_samples = fingerprints.shape[0]
        
        # ALWAYS use user-provided chunk_size if available
        if chunk_size is None:
            if hasattr(self, 'chunk_size') and self.chunk_size:
                chunk_size = self.chunk_size
                print(f"🎯 Using user-provided chunk size: {chunk_size}")
            else:
                chunk_size = min(100, max(25, int(np.sqrt(n_samples))))
                print(f"🎯 Calculated default chunk size: {chunk_size}")
        
        # FORCE GPU CHUNKING for ANY dataset size if GPU available
        print(f"🔍 GPU Detection: use_gpu={self.use_gpu}, gpu_backend={getattr(self, 'gpu_backend', None)}")
        if self.use_gpu and GPU_AVAILABLE and GPU_BACKEND == "cupy":
            print(f"🚀 FORCING GPU CHUNKING for {n_samples:,} molecules")
            print(f"🎯 Requested chunk size: {chunk_size}")
            try:
                import cupy as cp
                cp.cuda.Device(0).use()
                
                # Always use user chunk size if provided, otherwise calculate optimal  
                final_chunk_size = chunk_size  # Use user-provided chunk size directly
                
                # Show verbose chunk info for GPU - SAME AS CPU
                n_chunks = (n_samples + final_chunk_size - 1) // final_chunk_size
                print(f"🧩 GPU Chunking info:")
                print(f"   - Molecules per chunk: {final_chunk_size}")
                print(f"   - Number of chunks: {n_chunks}")
                print(f"   - Dataset size: {n_samples:,} molecules")
                print(f"   - Processing mode: GPU-native")
                print(f"   - Estimated processing time: {n_chunks * 0.1:.1f}s")
                
                return self._calculate_distances_chunked_cuda_native(fingerprints, final_chunk_size)
                
            except Exception as cuda_e:
                print(f"💥 GPU chunking failed: {cuda_e}")
                print(f"🔄 Falling back to CPU for dataset...")
                # Continue to CPU processing below
        
        # CPU processing with user chunk size
        print(f"🔄 Using CPU for {n_samples:,} molecules")
        print(f"🎯 Requested chunk size: {chunk_size}")
        
        # Calculate CPU chunk info with user-provided chunk_size
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        total_chunk_pairs = n_chunks * n_chunks
        
        print(f"🧩 CPU Chunking info:")
        print(f"   - Molecules per chunk: {chunk_size}")
        print(f"   - Number of chunks: {n_chunks}")
        print(f"   - Total chunk pairs: {total_chunk_pairs:,}")
        print(f"   - Dataset size: {n_samples:,} molecules")
        print(f"   - Processing mode: CPU-parallel")
        print(f"   - Estimated processing time: {total_chunk_pairs * 0.1:.1f}s")
        
        # Prevent huge CPU calculations unless user explicitly requested small chunks
        if n_samples > 50000 and chunk_size < 100:
            estimated_time_hours = total_chunk_pairs * 0.5 / 3600
            print(f"⚠️  WARNING: Small chunk size on large dataset!")
            print(f"   Estimated time: {estimated_time_hours:.1f} hours")
            print(f"   Consider using larger chunks or GPU acceleration")
        
        return self._calculate_distances_chunked_cpu_fallback(fingerprints, chunk_size, total_chunk_pairs)

    def _calculate_distances_chunked_cuda_native(self, fingerprints, chunk_size):
        """
        GPU-native chunked distance calculation - MEMORY-EFFICIENT approach that doesn't allocate full matrix
        """
        import cupy as cp
        
        n_samples = fingerprints.shape[0]
        print(f"🚀 GPU-NATIVE chunked distance calculation: {n_samples:,} molecules")
        
        # Transfer fingerprints to GPU if not already there
        if hasattr(fingerprints, 'get'):
            print("✅ Fingerprints already on GPU")
            fingerprints_gpu = fingerprints
        else:
            print("📤 Transferring fingerprints to GPU")
            fingerprints_gpu = cp.asarray(fingerprints.astype(np.float32))
        
        # Calculate chunks
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        # CRITICAL: Check if we can fit the full distance matrix in memory
        matrix_size_gb = (n_samples * n_samples * 4) / (1024**3)  # 4 bytes per float32
        available_memory_gb = cp.cuda.Device().mem_info[0] / (1024**3)  # Free memory in GB
        
        print(f"📊 Memory Analysis:")
        print(f"   - Required for full matrix: {matrix_size_gb:.1f} GB")
        print(f"   - Available GPU memory: {available_memory_gb:.1f} GB")
        
        if matrix_size_gb > available_memory_gb * 0.8:  # Use 80% of available memory as safety margin
            print(f"⚠️ Full matrix ({matrix_size_gb:.1f} GB) too large for GPU memory ({available_memory_gb:.1f} GB)")
            print(f"🔄 Using streaming approach - processing chunk pairs on-demand")
            
            # Use streaming approach - don't store full distance matrix
            return self._gpu_streaming_clustering(fingerprints_gpu, chunk_size)
        else:
            print(f"✅ Full matrix fits in GPU memory - using standard approach")
            
            # Original approach - build full distance matrix
            print(f"🚀 Processing {n_chunks:,} chunks with GPU acceleration (FULL MATRIX METHOD)")
            
            # Initialize distance matrix (GPU)
            distances_gpu = cp.zeros((n_samples, n_samples), dtype=cp.float32)
            
            # Process chunks sequentially
            with tqdm(total=n_chunks, desc="🚀 GPU chunks", unit="chunk") as pbar:
                for i in range(n_chunks):
                    # Define chunk indices
                    start_i = i * chunk_size
                    end_i = min((i + 1) * chunk_size, n_samples)
                    
                    # Skip empty chunks
                    if start_i >= n_samples:
                        pbar.update(1)
                        continue
                    
                    # Calculate distances for this chunk against ALL molecules (GPU version)
                    chunk_distances = self._jaccard_distance_cuda_chunk_vs_all(
                        fingerprints_gpu[start_i:end_i],
                        fingerprints_gpu
                    )
                    
                    # Store in distance matrix
                    distances_gpu[start_i:end_i, :] = chunk_distances
                    
                    pbar.update(1)
            
            # Convert to CPU numpy array
            return cp.asnumpy(distances_gpu)

    def _gpu_streaming_clustering(self, fingerprints_gpu, chunk_size):
        """
        GPU streaming clustering - processes data in chunks without storing full distance matrix
        This is the approach for very large datasets that don't fit in GPU memory
        """
        import cupy as cp
        
        n_samples = len(fingerprints_gpu)
        print(f"🌊 GPU Streaming Clustering for {n_samples:,} molecules")
        print(f"💡 This approach processes chunks on-demand to save memory")
        
        # For datasets this large, we need to use the representative clustering approach
        # but with GPU acceleration for individual chunk processing
        
        # Convert back to CPU for the representative clustering approach
        # but use GPU for individual chunk distance calculations
        fingerprints_cpu = fingerprints_gpu.get()
        
        print(f"🔄 Switching to GPU-accelerated chunked representative clustering...")
        return self._gpu_accelerated_chunked_representative_clustering(
            fingerprints_cpu, fingerprints_gpu, chunk_size
        )

    def _gpu_accelerated_chunked_representative_clustering(self, fingerprints_cpu, fingerprints_gpu, chunk_size):
        """
        Chunked representative clustering with GPU acceleration for individual chunks
        """
        import cupy as cp
        
        n_samples = len(fingerprints_cpu)
        
        print(f"📊 GPU-Accelerated Chunked Representative Clustering:")
        print(f"   → Dataset: {n_samples:,} molecules")
        print(f"   → Chunk size: {chunk_size}")
        print(f"   → Threshold: {self.threshold}")
        print(f"   → Method: GPU for chunks, CPU for hierarchical clustering")
        
        # Step 1: Process molecules in chunks using GPU for distance calculations
        print(f"📊 Step 1: Processing molecules in chunks of {chunk_size:,} with GPU acceleration...")
        
        chunk_representatives = []
        chunk_labels_map = {}
        current_cluster_id = 0
        
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        with tqdm(total=n_chunks, desc="🚀 GPU-accelerated chunks", unit="chunk") as pbar:
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, n_samples)
                
                if start_idx >= n_samples:
                    pbar.update(1)
                    continue
                
                # Get chunk fingerprints (GPU)
                chunk_fps_gpu = fingerprints_gpu[start_idx:end_idx]
                chunk_indices = list(range(start_idx, end_idx))
                
                # Cluster this chunk using GPU for distance calculation
                if len(chunk_fps_gpu) > 1:
                    # GPU distance calculation for this chunk
                    chunk_distances_gpu = self._jaccard_distance_cuda_chunk_vs_all(
                        chunk_fps_gpu, chunk_fps_gpu
                    )
                    
                    # Convert to CPU for sklearn clustering
                    chunk_distances_cpu = cp.asnumpy(chunk_distances_gpu)
                    
                    # Use sklearn hierarchical clustering
                    clusterer = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=self.threshold,
                        metric='precomputed',
                        linkage='complete'
                    )
                    
                    chunk_cluster_labels = clusterer.fit_predict(chunk_distances_cpu)
                    
                    # Map local cluster labels to global cluster labels
                    local_to_global = {}
                    for local_label in np.unique(chunk_cluster_labels):
                        local_to_global[local_label] = current_cluster_id
                        current_cluster_id += 1
                    
                    # Store mapping for this chunk
                    for i, local_label in enumerate(chunk_cluster_labels):
                        global_idx = start_idx + i
                        global_label = local_to_global[local_label]
                        chunk_labels_map[global_idx] = global_label
                        
                        # Find representative for each cluster (first molecule in cluster)
                        if local_label == 0 or local_label not in [cl for cl in chunk_cluster_labels[:i]]:
                            chunk_representatives.append({
                                'index': global_idx,
                                'fingerprint': fingerprints_cpu[global_idx],  # Use CPU version for storage
                                'cluster_id': global_label
                            })
                else:
                    # Single molecule chunk
                    chunk_labels_map[start_idx] = current_cluster_id
                    chunk_representatives.append({
                        'index': start_idx,
                        'fingerprint': fingerprints_cpu[start_idx],
                        'cluster_id': current_cluster_id
                    })
                    current_cluster_id += 1
                
                pbar.update(1)
        
        print(f"   → Found {len(chunk_representatives)} chunk representatives")
        print(f"   → Initial clusters: {current_cluster_id}")
        
        # Step 2: Use the same hierarchical merging approach as before
        print(f"📊 Step 2: Merging similar representatives across chunks...")
        
        if len(chunk_representatives) > 1:
            # Use the existing hierarchical merging logic
            max_safe_representatives = 10000
            
            if len(chunk_representatives) > max_safe_representatives:
                print(f"⚠️ Too many representatives ({len(chunk_representatives):,}) for memory-safe clustering")
                print(f"🔄 Using hierarchical representative merging with max {max_safe_representatives} at a time...")
                
                final_labels = self._hierarchical_representative_merging(
                    chunk_representatives, chunk_labels_map, n_samples, max_safe_representatives
                )
            else:
                print(f"✅ Processing {len(chunk_representatives):,} representatives (safe for memory)")
                rep_fps = np.array([rep['fingerprint'] for rep in chunk_representatives])
                
                memory_needed_gb = (len(rep_fps) ** 2 * 8) / (1024 ** 3)
                
                if memory_needed_gb > 8.0:
                    print(f"⚠️ Memory requirement too high, using hierarchical approach...")
                    final_labels = self._hierarchical_representative_merging(
                        chunk_representatives, chunk_labels_map, n_samples, max_safe_representatives
                    )
                else:
                    rep_distances = pairwise_distances(rep_fps, metric='jaccard', n_jobs=-1)
                    
                    rep_clusterer = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=self.threshold,
                        metric='precomputed',
                        linkage='complete'
                    )
                    
                    rep_cluster_labels = rep_clusterer.fit_predict(rep_distances)
                    
                    cluster_merge_map = {}
                    for rep_idx, new_cluster_id in enumerate(rep_cluster_labels):
                        old_cluster_id = chunk_representatives[rep_idx]['cluster_id']
                        cluster_merge_map[old_cluster_id] = new_cluster_id
                    
                    final_labels = np.full(n_samples, -1, dtype=np.int32)
                    for mol_idx, old_cluster_id in chunk_labels_map.items():
                        final_labels[mol_idx] = cluster_merge_map[old_cluster_id]
            
            n_final_clusters = len(np.unique(final_labels))
            print(f"   → Final clusters after merging: {n_final_clusters}")
        else:
            final_labels = np.zeros(n_samples, dtype=np.int32)
            print(f"   → Single cluster (all molecules similar)")
        
        print(f"✅ GPU-accelerated chunked representative clustering completed!")
        return final_labels

    def _jaccard_distance_cuda_chunk_vs_all(self, chunk_gpu, all_fps_gpu):
        """GPU calculation: chunk of molecules vs ALL molecules"""
        import cupy as cp
        
        # Expand dimensions for broadcasting
        chunk_expanded = chunk_gpu[:, None, :].astype(cp.float32)  # (chunk_size, 1, features)
        all_expanded = all_fps_gpu[None, :, :].astype(cp.float32)   # (1, n_samples, features)
        
        # Vectorized Jaccard distance calculation
        intersection = cp.sum(chunk_expanded * all_expanded, axis=2, dtype=cp.float32)
        union = cp.sum(cp.maximum(chunk_expanded, all_expanded), axis=2, dtype=cp.float32)
        
        # Avoid division by zero and calculate distance
        jaccard_sim = cp.zeros_like(intersection)
        non_zero_mask = union != 0
        jaccard_sim[non_zero_mask] = intersection[non_zero_mask] / union[non_zero_mask]
        jaccard_dist = 1.0 - jaccard_sim
        
        return jaccard_dist

    def _calculate_distances_chunked_cpu_fallback(self, fingerprints, chunk_size, total_chunk_pairs):
        """
        CPU fallback for chunked distance calculation with early validation
        """
        n_samples = len(fingerprints)
        
        # Early validation to prevent memory allocation errors
        max_safe_samples = 10000  # Adjust based on available memory
        
        print(f"🔍 CPU Fallback: Processing {n_samples} samples with chunk size {chunk_size}")
        
        if n_samples > max_safe_samples:
            print(f"⚠️ Dataset too large for standard CPU processing: {n_samples:,} > {max_safe_samples:,}")
            print(f"🔄 Switching to matrix-free clustering approach...")
            return self._perform_clustering_without_full_matrix(fingerprints, chunk_size)
        
        # Calculate memory requirement
        memory_gb = (n_samples * n_samples * 4) / (1024**3)  # 4 bytes per float32
        print(f"📊 Memory requirement: {memory_gb:.2f} GB for distance matrix")
        
        if memory_gb > 8:  # Adjust threshold based on available RAM
            print(f"❌ Memory requirement ({memory_gb:.2f} GB) exceeds safe threshold")
            raise MemoryError(f"Required memory ({memory_gb:.2f} GB) exceeds available memory")
        
        try:
            print(f"🔄 Using CPU for {n_samples:,} molecules")
            print(f"🎯 Requested chunk size: {chunk_size}")
            
            # Calculate CPU chunk info with user-provided chunk_size
            n_chunks = (n_samples + chunk_size - 1) // chunk_size
            total_chunk_pairs = n_chunks * n_chunks
            
            print(f"🧩 CPU Chunking info:")
            print(f"   - Molecules per chunk: {chunk_size}")
            print(f"   - Number of chunks: {n_chunks}")
            print(f"   - Total chunk pairs: {total_chunk_pairs:,}")
            print(f"   - Dataset size: {n_samples:,} molecules")
            print(f"   - Processing mode: CPU-parallel")
            print(f"   - Estimated processing time: {total_chunk_pairs * 0.1:.1f}s")
            
            # Prevent huge CPU calculations unless user explicitly requested small chunks
            if n_samples > 50000 and chunk_size < 100:
                estimated_time_hours = total_chunk_pairs * 0.5 / 3600
                print(f"⚠️  WARNING: Small chunk size on large dataset!")
                print(f"   Estimated time: {estimated_time_hours:.1f} hours")
                print(f"   Consider using larger chunks or GPU acceleration")
            
            # Create memory-mapped distance matrix for smaller datasets
            print(f"💾 Creating memory-mapped distance matrix ({n_samples}x{n_samples})...")
            
            try:
                distances_mmap = np.zeros((n_samples, n_samples), dtype=np.float32)
            except MemoryError:
                print(f"❌ Cannot allocate matrix for {n_samples} samples - using matrix-free approach...")
                return self._perform_clustering_without_full_matrix(fingerprints, chunk_size)
            
            # Process chunks sequentially (not pairs)
            with tqdm(total=n_chunks, desc=f"Processing chunks", unit="chunk") as pbar:
                for i in range(n_chunks):
                    # Define chunk indices
                    start_i = i * chunk_size
                    end_i = min((i + 1) * chunk_size, n_samples)
                    
                    # Skip empty chunks
                    if start_i >= n_samples:
                        pbar.update(1)
                        continue
                    
                    # Calculate distances for this chunk against all molecules
                    chunk_distances = pairwise_distances(
                        fingerprints[start_i:end_i],
                        fingerprints,
                        metric='jaccard',
                        n_jobs=-1
                    )
                    
                    # Store in distance matrix
                    distances_mmap[start_i:end_i, :] = chunk_distances
                    
                    pbar.update(1)
            
            return distances_mmap
        except Exception as e:
            print(f"Error during chunked distance calculation: {e}")
            raise

    def calculate_tanimoto_similarities(self, fingerprints, cluster_labels, mol_ids):
        """Calculate Tanimoto similarities within clusters with GPU acceleration"""
        print("Calculating Tanimoto similarities within clusters...")
        
        # Convert CuPy arrays to NumPy if needed for processing
        if hasattr(fingerprints, 'get'):
            fingerprints_cpu = fingerprints.get()
        else:
            fingerprints_cpu = fingerprints
        
        # Use GPU acceleration if available
        if self.use_gpu and GPU_AVAILABLE and GPU_BACKEND == "cupy":
            print("🚀 Using GPU acceleration for similarity calculations...")
            return self._calculate_similarities_gpu_accelerated(fingerprints_cpu, cluster_labels)
        else:
            print("💻 Using CPU for similarity calculations...")
            return self._calculate_similarities_cpu(fingerprints_cpu, cluster_labels)
    
    def _calculate_similarities_gpu_accelerated(self, fingerprints, cluster_labels):
        """GPU-accelerated similarity calculation within clusters"""
        import cupy as cp
        
        # Transfer fingerprints to GPU
        fingerprints_gpu = cp.asarray(fingerprints.astype(np.float32))
        cluster_labels_gpu = cp.asarray(cluster_labels)
        
        similarities = []
        unique_labels = cp.unique(cluster_labels_gpu)
        
        print(f"🚀 Processing {len(unique_labels)} clusters on GPU...")
        
        with tqdm(total=len(unique_labels), desc="🚀 GPU similarity calc", unit="cluster") as pbar:
            for label_gpu in unique_labels:
                label = label_gpu.item()  # Convert to Python scalar
                
                if label == -1:
                    # Skip noise points (if any)
                    pbar.update(1)
                    continue
                
                # Get indices of molecules in this cluster (GPU)
                cluster_mask = cluster_labels_gpu == label
                cluster_indices = cp.where(cluster_mask)[0]
                
                if len(cluster_indices) < 2:
                    # Not enough molecules for similarity calculation
                    similarities.extend([0] * len(cluster_indices))
                    pbar.update(1)
                    continue
                
                # Extract cluster fingerprints (GPU)
                cluster_fps_gpu = fingerprints_gpu[cluster_indices]
                
                # Calculate pairwise similarities within the cluster (GPU)
                cluster_similarities_gpu = self._gpu_jaccard_similarities_matrix(cluster_fps_gpu)
                
                # Fill the diagonal with 1.0 (similarity with self)
                n_cluster = len(cluster_similarities_gpu)
                diag_indices = cp.arange(n_cluster)
                cluster_similarities_gpu[diag_indices, diag_indices] = 1.0
                
                # Average similarity for molecules in this cluster
                avg_similarity_gpu = cp.mean(cluster_similarities_gpu, axis=1)
                
                # Convert back to CPU and store
                similarities.extend(avg_similarity_gpu.get().tolist())
                
                pbar.update(1)
        
        return similarities
    
    def _gpu_jaccard_similarities_matrix(self, cluster_fps_gpu):
        """Calculate Jaccard similarity matrix for a cluster on GPU"""
        import cupy as cp
        
        n_molecules = len(cluster_fps_gpu)
        
        # Expand dimensions for broadcasting
        fps_i = cluster_fps_gpu[:, None, :].astype(cp.float32)  # (n, 1, features)
        fps_j = cluster_fps_gpu[None, :, :].astype(cp.float32)  # (1, n, features)
        
        # Vectorized Jaccard similarity calculation
        intersection = cp.sum(fps_i * fps_j, axis=2, dtype=cp.float32)
        union = cp.sum(cp.maximum(fps_i, fps_j), axis=2, dtype=cp.float32)
        
        # Avoid division by zero and calculate similarity
        jaccard_sim = cp.zeros_like(intersection)
        non_zero_mask = union != 0
        jaccard_sim[non_zero_mask] = intersection[non_zero_mask] / union[non_zero_mask]
        
        return jaccard_sim
    
    def _calculate_similarities_cpu(self, fingerprints, cluster_labels):
        """CPU fallback for similarity calculation"""
        similarities = []
        unique_labels = np.unique(cluster_labels)
        
        with tqdm(total=len(unique_labels), desc="💻 CPU similarity calc", unit="cluster") as pbar:
            for label in unique_labels:
                if label == -1:
                    # Skip noise points (if any)
                    pbar.update(1)
                    continue
                
                # Get indices of molecules in this cluster
                cluster_indices = np.where(cluster_labels == label)[0]
                
                if len(cluster_indices) < 2:
                    # Not enough molecules for similarity calculation
                    similarities.extend([0] * len(cluster_indices))
                    pbar.update(1)
                    continue
                
                # Calculate pairwise similarities within the cluster
                cluster_fps = fingerprints[cluster_indices]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    cluster_similarities = 1 - pairwise_distances(cluster_fps, metric='jaccard', n_jobs=-1)
                
                # Fill the diagonal with 1.0 (similarity with self)
                for i in range(len(cluster_similarities)):
                    cluster_similarities[i, i] = 1.0
                
                # Average similarity for molecules in this cluster
                avg_similarity = np.mean(cluster_similarities, axis=1)
                
                # Store similarities for all molecules in this cluster
                similarities.extend(avg_similarity)
                
                pbar.update(1)
        
        return similarities

    def calculate_detailed_similarity_matrices(self, fingerprints, cluster_labels, mol_ids):
        """Calculate detailed similarity matrices for all clusters with GPU acceleration"""
        print("🧮 Calculating detailed similarity matrices...")
        
        # Convert CuPy arrays to NumPy if needed for processing
        if hasattr(fingerprints, 'get'):
            fingerprints_cpu = fingerprints.get()
        else:
            fingerprints_cpu = fingerprints
        
        # Use GPU acceleration if available
        if self.use_gpu and GPU_AVAILABLE and GPU_BACKEND == "cupy":
            print("🚀 Using GPU acceleration for detailed similarity matrices...")
            return self._calculate_detailed_matrices_gpu_accelerated(fingerprints_cpu, cluster_labels, mol_ids)
        else:
            print("💻 Using CPU for detailed similarity matrices...")
            return self._calculate_detailed_matrices_cpu(fingerprints_cpu, cluster_labels, mol_ids)
    
    def _calculate_detailed_matrices_gpu_accelerated(self, fingerprints, cluster_labels, mol_ids):
        """GPU-accelerated detailed similarity matrix calculation"""
        import cupy as cp
        
        # Transfer data to GPU
        fingerprints_gpu = cp.asarray(fingerprints.astype(np.float32))
        cluster_labels_gpu = cp.asarray(cluster_labels)
        
        detailed_results = {}
        unique_labels = cp.unique(cluster_labels_gpu)
        n_clusters = len(unique_labels)
        n_molecules = len(fingerprints)
        
        print(f"   📊 {n_clusters} clusters • {n_molecules:,} molecules")
        
        with tqdm(total=n_clusters, desc="🚀 GPU detailed matrices", unit="cluster") as pbar:
            for label_gpu in unique_labels:
                label = label_gpu.item()
                
                if label == -1:
                    # Skip noise points
                    pbar.update(1)
                    continue
                
                # Get cluster data (GPU)
                cluster_mask = cluster_labels_gpu == label
                cluster_indices = cp.where(cluster_mask)[0]
                cluster_size = len(cluster_indices)
                
                if cluster_size < 2:
                    # Skip clusters too small for meaningful analysis
                    detailed_results[label] = {
                        'size': cluster_size,
                        'similarity_matrix': None,
                        'avg_similarities': [1.0] * cluster_size if cluster_size > 0 else [],
                        'molecule_ids': [mol_ids[idx] for idx in cluster_indices.get()],
                        'intra_cluster_distances': None
                    }
                    pbar.update(1)
                    continue
                
                # Extract cluster fingerprints (GPU)
                cluster_fps_gpu = fingerprints_gpu[cluster_indices]
                
                # Calculate full similarity matrix for this cluster (GPU)
                similarity_matrix_gpu = self._gpu_jaccard_similarities_matrix(cluster_fps_gpu)
                
                # Calculate comprehensive statistics (GPU)
                # Fill diagonal with 1.0
                n_cluster = len(similarity_matrix_gpu)
                diag_indices = cp.arange(n_cluster)
                similarity_matrix_gpu[diag_indices, diag_indices] = 1.0
                
                # Calculate average similarities per molecule
                avg_similarities_gpu = cp.mean(similarity_matrix_gpu, axis=1)
                
                # Calculate intra-cluster distances (1 - similarities)
                distance_matrix_gpu = 1.0 - similarity_matrix_gpu
                
                # Convert results back to CPU for storage
                detailed_results[label] = {
                    'size': cluster_size,
                    'similarity_matrix': similarity_matrix_gpu.get(),
                    'avg_similarities': avg_similarities_gpu.get().tolist(),
                    'molecule_ids': [mol_ids[idx] for idx in cluster_indices.get()],
                    'intra_cluster_distances': distance_matrix_gpu.get(),
                    'min_similarity': float(cp.min(similarity_matrix_gpu[similarity_matrix_gpu < 1.0]).get()),
                    'max_similarity': float(cp.max(similarity_matrix_gpu[similarity_matrix_gpu < 1.0]).get()),
                    'mean_similarity': float(cp.mean(similarity_matrix_gpu[similarity_matrix_gpu < 1.0]).get())
                }
                
                pbar.update(1)
        
        return detailed_results
    
    def _calculate_detailed_matrices_cpu(self, fingerprints, cluster_labels, mol_ids):
        """CPU fallback for detailed similarity matrix calculation"""
        detailed_results = {}
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)
        n_molecules = len(fingerprints)
        
        print(f"   📊 {n_clusters} clusters • {n_molecules:,} molecules")
        
        with tqdm(total=n_clusters, desc="💻 CPU detailed matrices", unit="cluster") as pbar:
            for label in unique_labels:
                if label == -1:
                    # Skip noise points
                    pbar.update(1)
                    continue
                
                # Get cluster data
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_size = len(cluster_indices)
                
                if cluster_size < 2:
                    # Skip clusters too small for meaningful analysis
                    detailed_results[label] = {
                        'size': cluster_size,
                        'similarity_matrix': None,
                        'avg_similarities': [1.0] * cluster_size if cluster_size > 0 else [],
                        'molecule_ids': [mol_ids[idx] for idx in cluster_indices],
                        'intra_cluster_distances': None
                    }
                    pbar.update(1)
                    continue
                
                # Extract cluster fingerprints
                cluster_fps = fingerprints[cluster_indices]
                
                # Calculate similarity matrix for this cluster
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    similarity_matrix = 1 - pairwise_distances(cluster_fps, metric='jaccard', n_jobs=-1)
                
                # Fill diagonal with 1.0
                np.fill_diagonal(similarity_matrix, 1.0)
                
                # Calculate average similarities per molecule
                avg_similarities = np.mean(similarity_matrix, axis=1)
                
                # Calculate intra-cluster distances
                distance_matrix = 1.0 - similarity_matrix
                
                # Store detailed results
                detailed_results[label] = {
                    'size': cluster_size,
                    'similarity_matrix': similarity_matrix,
                    'avg_similarities': avg_similarities.tolist(),
                    'molecule_ids': [mol_ids[idx] for idx in cluster_indices],
                    'intra_cluster_distances': distance_matrix,
                    'min_similarity': float(np.min(similarity_matrix[similarity_matrix < 1.0])),
                    'max_similarity': float(np.max(similarity_matrix[similarity_matrix < 1.0])),
                    'mean_similarity': float(np.mean(similarity_matrix[similarity_matrix < 1.0]))
                }
                
                pbar.update(1)
        
        return detailed_results

    def _add_cluster_analysis(self, results, fingerprints, cluster_labels, mol_ids):
        """Add comprehensive cluster analysis to results DataFrame"""
        print("Adding cluster analysis...")
        
        # Add cluster size column
        cluster_sizes = {}
        for label in cluster_labels:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        
        results['cluster_size'] = results['cluster_id'].map(cluster_sizes)
        
        # Add cluster members column
        cluster_members = results.groupby('cluster_id')['molecule_id'].apply(list)
        results['cluster_members'] = results['cluster_id'].map(cluster_members)
        
        # Calculate Tanimoto similarities within clusters
        similarities = self.calculate_tanimoto_similarities(
            fingerprints, cluster_labels, mol_ids
        )
        results['tanimoto_similarities'] = similarities
        
        return results

    def print_clustering_stats(self, df):
        """Print clustering statistics"""
        print("\n=== Clustering Statistics ===")
        print(f"Total molecules: {len(df):,}")
        print(f"Number of clusters: {df['cluster_id'].nunique():,}")
        print(f"Average cluster size: {len(df) / df['cluster_id'].nunique():.2f}")
        
        cluster_sizes = df['cluster_id'].value_counts().sort_values(ascending=False)
        print(f"Largest cluster size: {cluster_sizes.iloc[0]:,}")
        print(f"Smallest cluster size: {cluster_sizes.iloc[-1]:,}")
        print(f"Singletons (clusters of size 1): {sum(cluster_sizes == 1):,}")
        
        # Show distribution of cluster sizes
        size_distribution = cluster_sizes.value_counts().sort_index()
        print(f"\nCluster size distribution:")
        for size, count in size_distribution.head(10).items():
            print(f"  Size {size}: {count:,} clusters")
        if len(size_distribution) > 10:
            print(f"  ... and {len(size_distribution) - 10} more size categories")

    def cluster_smiles(self, input_file='chembl_smiles.csv', output_file='clustered_smiles.csv', 
                      use_custom_data=False, smiles_column='smiles', molecule_id_column=None):
        """Cluster SMILES data using hierarchical clustering"""
        print(f"\nClustering SMILES data: {input_file}")
        
        # Load custom data if specified
        if use_custom_data:
            df = self.load_custom_data(input_file, smiles_column, molecule_id_column)
        else:
            # Download ChEMBL data
            df = self.download_chembl_smiles(n_samples=100000, output_file=input_file)
        
        # Calculate fingerprints
        fingerprints, valid_indices = self.calculate_morgan_fingerprints(df['smiles'].tolist())
        
        # Perform clustering
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            cluster_labels = self.perform_clustering(fingerprints)
        
        # Add cluster labels to results
        df['cluster_id'] = -1  # Default to -1 (no cluster)
        df.loc[valid_indices, 'cluster_id'] = cluster_labels
        
        # Save results
        df.to_csv(output_file, index=False)
        print(f"Clustered data saved to: {output_file}")
        
        # Print clustering statistics
        self.print_clustering_stats(df)
        
        return df

    def load_custom_data(self, input_file, smiles_column='smiles', molecule_id_column=None):
        """Load custom SMILES data from a CSV file"""
        print(f"Loading custom data from: {input_file}")
        
        df = pd.read_csv(input_file)
        
        if smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found in the input file")
        
        # Use molecule_id_column if provided, otherwise create default IDs
        if molecule_id_column and molecule_id_column in df.columns:
            df['molecule_id'] = df[molecule_id_column]
        else:
            df['molecule_id'] = [f'CUSTOM_{i+1}' for i in range(len(df))]
        
        # Keep only relevant columns
        df = df[['molecule_id', smiles_column]]
        df.columns = ['molecule_id', 'smiles']
        
        print(f"Loaded {len(df)} molecules from custom data")
        
        return df

def main():
    # Initialize clusterer
    clusterer = SMILESClusterer(
        radius=3,
        n_bits=2048,
        threshold=0.3
    )
    
    # Example usage for custom data
    # Uncomment and modify these lines to use your own CSV file:
    """
    results = clusterer.cluster_smiles(
        input_file='my_compounds.csv',
        output_file='my_clustered_compounds.csv',
        use_custom_data=True,
        smiles_column='SMILES',  # Your SMILES column name
        molecule_id_column='compound_id'  # Your ID column name (optional)
    )
    """
    
    # Default ChEMBL clustering
    results = clusterer.cluster_smiles(
        input_file='chembl_smiles.csv',
        output_file='clustered_smiles.csv'
    )
    
    # Save fingerprints for future use
    print("Saving fingerprints for future analysis...")
    fingerprints, valid_indices = clusterer.calculate_morgan_fingerprints(
        results['smiles'].tolist()
    )
    
    with open('morgan_fingerprints.pkl', 'wb') as f:
        pickle.dump({
            'fingerprints': fingerprints,
            'smiles': results['smiles'].tolist(),
            'cluster_ids': results['cluster_id'].tolist()
        }, f)

if __name__ == "__main__":
    main()
