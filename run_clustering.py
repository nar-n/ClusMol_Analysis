import argparse
import sys
import os
import pandas as pd
import numpy as np
import logging
from smiles_clustering import SMILESClusterer
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_gpu_info():
    """Check and report GPU availability including CUDA"""
    print("\n" + "="*60)
    print("🖥️  HARDWARE DETECTION")
    print("="*60)
    
    # Check CuPy/CUDA with better error handling
    try:
        import cupy as cp
        
        # Test basic GPU functionality
        cp.cuda.Device(0).use()
        
        # Get device name with fallback methods
        try:
            device_name = cp.cuda.Device().name.decode('utf-8')
        except AttributeError:
            try:
                device_props = cp.cuda.runtime.getDeviceProperties(0)
                device_name = device_props['name'].decode('utf-8')
            except:
                device_name = f"CUDA Device {cp.cuda.Device().id}"
        
        # Get memory info
        free_bytes, total_bytes = cp.cuda.Device().mem_info
        total_gb = total_bytes / 1024**3
        free_gb = free_bytes / 1024**3
        
        print("✅ GPU Support: Available (CuPy/CUDA)")
        print(f"📱 Device: {device_name}")
        print(f"💾 Memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total ({free_gb/total_gb*100:.1f}% available)")
        
        # Specific info for RTX A4000
        if "RTX A4000" in device_name:
            print(f"🚀 RTX A4000 Professional GPU Detected")
            print(f"   • CUDA Cores: ~6,144 • Memory Bandwidth: ~448 GB/s")
            print(f"   • Optimal for: 200K-500K molecules • Recommended chunk size: 25-50")
            
        try:
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            print(f"🔧 CUDA Version: {cuda_version}")
        except:
            pass
            
        return True
        
    except ImportError:
        print("❌ GPU Support: CuPy not available")
    except Exception as e:
        print(f"⚠️  GPU Support: CuPy available but error: {str(e)[:50]}...")
        print("   💡 Try: pip install cupy-cuda11x or cupy-cuda12x")
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print("✅ GPU Support: Available (PyTorch CUDA)")
            print(f"📱 Device: {device_name}")
            print(f"💾 Memory: {total_memory:.1f} GB")
            return True
    except ImportError:
        pass
    
    print("🖥️  Using CPU-only processing")
    print("="*60)
    return False

def prompt_cpu_fallback(args):
    """Prompt user to continue with CPU when GPU is not available for large datasets"""
    if not args.cuda:
        return True
    
    print("\n" + "⚠️ "*20)
    print("GPU ACCELERATION NOT AVAILABLE")
    print("⚠️ "*20)
    
    # Estimate dataset size and processing time
    if hasattr(args, 'input') and args.input:
        try:
            df = pd.read_csv(args.input)
            dataset_size = len(df)
            print(f"📊 Dataset size: {dataset_size:,} molecules")
            
            if args.force_full_clustering:
                if dataset_size > 50000:
                    estimated_time = "6-12 hours"
                    warning_level = "🔴 VERY LONG"
                elif dataset_size > 20000:
                    estimated_time = "2-4 hours"
                    warning_level = "🟠 LONG"
                elif dataset_size > 10000:
                    estimated_time = "30-90 minutes"
                    warning_level = "🟡 MODERATE"
                else:
                    estimated_time = "5-30 minutes"
                    warning_level = "🟢 REASONABLE"
            else:
                estimated_time = "15-45 minutes (with sampling)"
                warning_level = "🟢 REASONABLE"
            
            print(f"⏱️  Estimated CPU time: {estimated_time}")
            print(f"📈 Processing complexity: {warning_level}")
            
        except Exception:
            print("❓ Could not estimate processing time")
    
    print("\n📋 Options:")
    print("  [y] Continue with CPU-only processing")
    print("  [n] Cancel and install GPU support")
    print("  [h] Help")
    
    while True:
        try:
            choice = input("\n➤ Continue with CPU? [y/n/h]: ")

            if choice in ['y', 'yes', '1']:
                print("✅ Continuing with CPU processing...")
                return True
            elif choice in ['n', 'no', '2']:
                print("❌ Clustering cancelled.")
                print("\n💡 To enable GPU support:")
                print("   pip install cupy-cuda11x    # For CUDA 11.x")
                print("   pip install cupy-cuda12x    # For CUDA 12.x")
                return False
            elif choice in ['h', 'help']:
                print("\n📖 Help:")
                print("   • GPU provides 5-10x speedup for large datasets")
                print("   • Use sampling (remove --force-full-clustering) for faster processing")
                print("   • 'y' = Continue with CPU (may be slow)")
                print("   • 'n' = Cancel and fix GPU setup")
                continue
            else:
                print("❓ Please enter 'y', 'n', or 'h'")
                
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Cancelled by user.")
            return False

def load_and_validate_data(input_file, smiles_col, id_col):
    """Load and validate input data with flexible column detection"""
    print(f"\n📂 Loading data from {input_file}...")
    
    try:
        df = pd.read_csv(input_file)
        
        with tqdm(total=4, desc="Validating data", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            pbar.set_description("Loading CSV file")
            pbar.update(1)
            
            # Column detection
            pbar.set_description("Detecting columns")
            available_cols = list(df.columns)
            smiles_column_found = None
            
            # Try exact match first
            if smiles_col in df.columns:
                smiles_column_found = smiles_col
            else:
                # Try case-insensitive and common variants
                smiles_variants = [smiles_col.lower(), 'smiles', 'SMILES', 'smi', 'SMI', 'canonical_smiles', 'structure']
                for variant in smiles_variants:
                    matching_cols = [col for col in available_cols if col.lower() == variant.lower()]
                    if matching_cols:
                        smiles_column_found = matching_cols[0]
                        if smiles_column_found != smiles_col:
                            print(f"   📝 Found SMILES column: '{smiles_column_found}'")
                        break
            
            if not smiles_column_found:
                raise ValueError(f"SMILES column '{smiles_col}' not found. Available: {available_cols}")
            
            pbar.update(1)
            
            # ID column detection
            pbar.set_description("Setting up ID column")
            id_column_found = None
            if id_col and id_col in df.columns:
                id_column_found = id_col
            else:
                id_variants = ['molecule_id', 'Molecule_ID', 'mol_id', 'ID', 'id', 'compound_id', 'cid']
                for variant in id_variants:
                    if variant in available_cols:
                        id_column_found = variant
                        if id_column_found != id_col:
                            print(f"   📝 Auto-detected ID column: '{id_column_found}'")
                        break
            
            # Standardize column names
            if id_column_found and id_column_found != 'molecule_id':
                df = df.rename(columns={id_column_found: 'molecule_id'})
            elif not id_column_found:
                df['molecule_id'] = [f'MOL_{i+1}' for i in range(len(df))]
                print(f"   📝 Created molecule_id column")
            
            if smiles_column_found != 'smiles':
                df = df.rename(columns={smiles_column_found: 'smiles'})
            
            pbar.update(1)
            
            # Validation
            pbar.set_description("Validating SMILES")
            initial_count = len(df)
            df = df.dropna(subset=['smiles'])
            df = df[df['smiles'].str.strip() != '']
            
            if 'cluster_id' in df.columns:
                print("   ⚠️  Input appears to be already clustered - will re-cluster")
            
            pbar.update(1)
        
        valid_percent = len(df) / initial_count * 100
        print(f"✅ Data loaded: {len(df):,} valid molecules ({valid_percent:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def create_clusterer(args):
    """Create clusterer with correct parameters including CUDA support"""
    # Determine GPU backend
    use_gpu = args.gpu and not args.cpu_only
    gpu_backend = None
    
    if use_gpu:
        if args.cuda:
            # Force CUDA backend
            try:
                import cupy as cp
                gpu_backend = "cuda"
                print("Using CUDA GPU acceleration")
            except ImportError:
                print("CUDA requested but CuPy not available, falling back to CPU")
                use_gpu = False
        else:
            # Auto-detect best GPU backend
            gpu_backend = "auto"
    
    clusterer = SMILESClusterer(
        threshold=args.threshold,
        radius=args.radius,
        n_bits=args.n_bits,
        use_gpu=use_gpu,
        force_full=args.force_full_clustering,
        sample_size=getattr(args, 'sample_size', 10000),
        chunk_size=getattr(args, 'chunk_size', 1000),
        suppress_rdkit_errors=True
    )
    
    # Set GPU backend manually if CUDA requested
    if use_gpu and args.cuda:
        clusterer.gpu_backend = "cuda"
    
    return clusterer

def run_standard_clustering(clusterer, args):
    """Run standard clustering"""
    try:
        print("\n=== STANDARD CLUSTERING MODE ===")
        
        # Load and validate data
        df = load_and_validate_data(args.input, args.smiles_col, args.id_col)
        logging.info(f"Loaded {len(df)} molecules from {args.input}")
        
        # Generate fingerprints
        logging.info("Generating molecular fingerprints...")
        fingerprints, valid_indices = clusterer.calculate_morgan_fingerprints(df['smiles'].tolist())
        
        if fingerprints is None or len(fingerprints) == 0:
            raise ValueError("No valid fingerprints generated")
        
        logging.info(f"Generated {len(fingerprints)} fingerprints")
        
        # Filter to valid molecules
        df_valid = df.iloc[valid_indices].copy().reset_index(drop=True)
        
        # Perform clustering
        logging.info("Starting clustering...")
        cluster_labels = clusterer.perform_clustering(fingerprints)
        
        # Add cluster information
        df_valid['cluster_id'] = cluster_labels
        
        # Add cluster analysis
        df_valid = clusterer._add_cluster_analysis(df_valid, fingerprints, cluster_labels, df_valid['molecule_id'].tolist())
        
        # Save results
        df_valid.to_csv(args.output, index=False)
        logging.info(f"Saved results to {args.output}")
        
        # Print statistics
        clusterer.print_clustering_stats(df_valid)
        
        return df_valid
        
    except Exception as e:
        logging.error(f"Error in clustering: {e}")
        raise

def run_two_stage_clustering(clusterer, args):
    """Run two-stage clustering"""
    try:
        print("\n=== TWO-STAGE CLUSTERING MODE ===")
        
        # Load reference data
        ref_df = load_and_validate_data(args.reference_file, args.smiles_col, args.id_col)
        logging.info(f"Stage 1: Clustering {len(ref_df)} reference molecules")
        
        # Stage 1: Cluster reference set
        ref_results = clusterer.cluster_reference_set(
            ref_df['smiles'].tolist(),
            ref_df['molecule_id'].tolist(),
            tight_clustering=True
        )
        
        # Save reference results
        ref_output = args.output.replace('.csv', '_reference.csv')
        ref_results.to_csv(ref_output, index=False)
        logging.info(f"Saved reference clustering to {ref_output}")
        
        # Load assignment data
        assign_df = load_and_validate_data(args.input, args.smiles_col, args.id_col)
        logging.info(f"Stage 2: Assigning {len(assign_df)} molecules to reference clusters")
        
        # Stage 2: Assign molecules to reference clusters
        assign_results = clusterer.assign_to_reference_clusters(
            assign_df['smiles'].tolist(),
            assign_df['molecule_id'].tolist()
        )
        
        # Save assignment results
        assign_results.to_csv(args.output, index=False)
        logging.info(f"Saved assignment results to {args.output}")
        
        return assign_results
        
    except Exception as e:
        logging.error(f"Error in two-stage clustering: {e}")
        raise

def calculate_tanimoto_similarities_for_molecule(reference_smiles, cluster_smiles_list, use_cuda=False):
    """Calculate Tanimoto similarities of reference molecule with all molecules in cluster"""
    if len(cluster_smiles_list) <= 1:
        return [1.00]
    
    # Convert reference SMILES to molecule and fingerprint
    ref_mol = Chem.MolFromSmiles(reference_smiles)
    if ref_mol is None:
        return [0.00] * len(cluster_smiles_list)
    
    # Use modern RDKit fingerprint method
    from rdkit.Chem import rdMolDescriptors
    ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol, radius=2, nBits=2048)
    
    # Calculate similarities with all molecules in cluster
    similarities = []
    for smiles in cluster_smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            if smiles == reference_smiles:
                # Self-similarity is always 1.0
                similarities.append(1.00)
            else:
                similarity = DataStructs.TanimotoSimilarity(ref_fp, fp)
                similarities.append(round(similarity, 2))
        else:
            similarities.append(0.00)
    
    return similarities

def create_fingerprints(smiles_list, use_cuda=False):
    """Create molecular fingerprints - consistent between CPU and GPU modes"""
    from rdkit.Chem import rdMolDescriptors
    
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Use modern RDKit method with explicit parameters
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fingerprints.append(fp)
        else:
            # Create zero fingerprint for invalid SMILES
            fp = DataStructs.ExplicitBitVect(2048)
            fingerprints.append(fp)
    return fingerprints

def calculate_similarity_matrix_batch(fingerprints, batch_size=1000, use_cuda=False, verbose=False):
    """Calculate similarity matrix with consistent batching for CPU/GPU"""
    n = len(fingerprints)
    
    # Suppress RDKit warnings for cleaner output
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    if use_cuda:
        try:
            import cupy as cp
            
            print(f"\n🚀 GPU Mode: Processing {n:,} fingerprints with CUDA")
            print(f"   📦 Batch size: {batch_size}")
            print(f"   💾 Expected memory usage: ~{(n*n*4)/(1024**3):.2f} GB")
            
            # Ensure consistent matrix initialization
            similarity_matrix = cp.zeros((n, n), dtype=cp.float32)
            
            # Calculate total batches for progress tracking
            total_i_batches = (n + batch_size - 1) // batch_size
            total_batches = sum((total_i_batches - i) for i in range(total_i_batches))
            
            # Process in batches with consistent ordering and progress bar
            with tqdm(
                total=total_batches,
                desc="GPU similarity calculation",
                unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]"
            ) as pbar:
                
                for i in range(0, n, batch_size):
                    end_i = min(i + batch_size, n)
                    batch_fps_i = fingerprints[i:end_i]
                    i_batch_num = i // batch_size
                    
                    for j in range(i, n, batch_size):
                        end_j = min(j + batch_size, n)
                        batch_fps_j = fingerprints[j:end_j]
                        
                        pbar.set_description(f"GPU batch [{i_batch_num}] processing {len(batch_fps_i)}x{len(batch_fps_j)} comparisons")
                        
                        # Create batch similarities on CPU first (RDKit is CPU-only)
                        batch_similarities_cpu = np.zeros((len(batch_fps_i), len(batch_fps_j)), dtype=np.float32)
                        
                        # Calculate similarities on CPU (RDKit requirement)
                        for bi, fp_i in enumerate(batch_fps_i):
                            for bj, fp_j in enumerate(batch_fps_j):
                                sim = DataStructs.TanimotoSimilarity(fp_i, fp_j)
                                batch_similarities_cpu[bi, bj] = sim
                        
                        # Convert to CuPy array and store results
                        batch_similarities_gpu = cp.asarray(batch_similarities_cpu)
                        similarity_matrix[i:end_i, j:end_j] = batch_similarities_gpu
                        if i != j:  # Fill symmetric part
                            similarity_matrix[j:end_j, i:end_i] = batch_similarities_gpu.T
                        
                        pbar.update(1)
            
            print("✅ GPU similarity matrix calculation complete")
            return similarity_matrix
            
        except ImportError:
            print("❌ CuPy not available, falling back to CPU mode")
            use_cuda = False
        except Exception as e:
            print(f"⚠️  GPU error: {e}, falling back to CPU mode")
            use_cuda = False
    
    # CPU fallback with identical logic and improved progress tracking
    print(f"\n🖥️  CPU Mode: Processing {n:,} fingerprints")
    print(f"   📦 Batch size: {batch_size}")
    print(f"   💾 Expected memory usage: ~{(n*n*4)/(1024**3):.2f} GB")
    
    similarity_matrix = np.zeros((n, n), dtype=np.float32)
    
    # Calculate total batches for progress tracking
    total_i_batches = (n + batch_size - 1) // batch_size
    total_batches = sum((total_i_batches - i) for i in range(total_i_batches))
    
    with tqdm(
        total=total_batches,
        desc="CPU similarity calculation",
        unit="batch",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]"
    ) as pbar:
        
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            batch_fps_i = fingerprints[i:end_i]
            i_batch_num = i // batch_size
            
            for j in range(i, n, batch_size):
                end_j = min(j + batch_size, n)
                batch_fps_j = fingerprints[j:end_j]
                
                pbar.set_description(f"CPU batch [{i_batch_num}] processing {len(batch_fps_i)}x{len(batch_fps_j)} comparisons")
                
                batch_similarities = np.zeros((len(batch_fps_i), len(batch_fps_j)), dtype=np.float32)
                
                for bi, fp_i in enumerate(batch_fps_i):
                    for bj, fp_j in enumerate(batch_fps_j):
                        sim = DataStructs.TanimotoSimilarity(fp_i, fp_j)
                        batch_similarities[bi, bj] = sim
                
                similarity_matrix[i:end_i, j:end_j] = batch_similarities
                if i != j:
                    similarity_matrix[j:end_j, i:end_i] = batch_similarities.T
                
                pbar.update(1)
    
    print("✅ CPU similarity matrix calculation complete")
    
    # Re-enable RDKit logging
    RDLogger.EnableLog('rdApp.*')
    
    return similarity_matrix

def cluster_molecules_cupy(similarity_matrix, threshold, verbose=False):
    """GPU clustering using CuPy - ensure consistent results"""
    import cupy as cp
    
    print(f"\n🚀 GPU Clustering: Applying threshold {threshold}")
    print(f"   📊 Matrix size: {similarity_matrix.shape[0]:,} x {similarity_matrix.shape[1]:,}")
    
    with tqdm(total=3, desc="GPU clustering", bar_format="{l_bar}{bar}| {desc}") as pbar:
        pbar.set_description("Converting to CuPy array")
        # Ensure we're working with CuPy array
        if not isinstance(similarity_matrix, cp.ndarray):
            similarity_matrix_cp = cp.asarray(similarity_matrix)
        else:
            similarity_matrix_cp = similarity_matrix
        pbar.update(1)
        
        pbar.set_description("Applying threshold")
        # Apply threshold to create binary matrix
        binary_matrix = similarity_matrix_cp >= threshold
        
        # Ensure consistent handling of self-loops - use CuPy function
        cp.fill_diagonal(binary_matrix, True)  # Use True instead of 1 for boolean matrix
        pbar.update(1)
        
        pbar.set_description("Finding connected components")
        # Use CuPy's connected components to find clusters
        from cupyx.scipy.sparse import csgraph
        
        # Find connected components (clusters)
        connected_components = csgraph.connected_components(binary_matrix, directed=False, return_labels=True)
        pbar.update(1)
    
    # Extract clusters from connected components - handle CuPy arrays properly
    clusters = []
    n_components, labels = connected_components
    
    for cluster_id in range(n_components):
        # Use CuPy operations and explicitly convert to CPU
        cluster_indices = cp.where(labels == cluster_id)[0]
        clusters.append(cluster_indices.get())  # Convert CuPy array to NumPy
    
    # Ensure deterministic results by sorting clusters consistently
    sorted_clusters = []
    for cluster_indices in clusters:
        # cluster_indices is already a NumPy array from .get()
        sorted_cluster = sorted(cluster_indices.tolist())
        sorted_clusters.append(sorted_cluster)
    
    # Sort clusters by size (largest first), then by first element for consistency
    sorted_clusters.sort(key=lambda x: (-len(x), x[0]))
    
    print(f"✅ GPU clustering complete: {len(sorted_clusters)} clusters found")
    return sorted_clusters

def cluster_molecules_networkx(similarity_matrix, threshold, verbose=False):
    """CPU clustering using NetworkX - ensure consistent results"""
    import networkx as nx
    
    print(f"\n🖥️  CPU Clustering: Applying threshold {threshold}")
    print(f"   📊 Matrix size: {similarity_matrix.shape[0]:,} x {similarity_matrix.shape[1]:,}")
    
    with tqdm(total=3, desc="CPU clustering", bar_format="{l_bar}{bar}| {desc}") as pbar:
        pbar.set_description("Creating graph from matrix")
        # Create a graph from the similarity matrix
        G = nx.from_numpy_array(similarity_matrix)
        pbar.update(1)
        
        pbar.set_description("Applying threshold to edges")
        # Apply a threshold to the edges
        edges_to_remove = []
        for i, j, data in G.edges(data=True):
            if data['weight'] < threshold:
                edges_to_remove.append((i, j))
        G.remove_edges_from(edges_to_remove)
        pbar.update(1)
        
        pbar.set_description("Finding connected components")
        # Find connected components (clusters)
        connected_components = list(nx.connected_components(G))
        pbar.update(1)
    
    # Ensure deterministic results by sorting clusters consistently
    clusters = []
    for cluster_indices in connected_components:
        sorted_cluster = sorted(list(cluster_indices))
        clusters.append(sorted_cluster)
    
    # Sort clusters by size (largest first), then by first element for consistency
    clusters.sort(key=lambda x: (-len(x), x[0]))
    
    print(f"✅ CPU clustering complete: {len(clusters)} clusters found")
    return clusters

def update_tanimoto_similarities(df, use_cuda=False):
    """Update the DataFrame with detailed Tanimoto similarity arrays"""
    
    # Suppress RDKit warnings for cleaner output
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    total_clusters = df['cluster_id'].nunique()
    total_molecules = len(df)
    cluster_groups = list(df.groupby('cluster_id').groups.items())
    
    print(f"\n🧮 Calculating detailed similarity matrices...")
    print(f"   📊 {total_clusters:,} clusters • {total_molecules:,} molecules")
    
    # Progress bar for clusters
    with tqdm(
        cluster_groups, 
        desc="Computing similarities", 
        unit="cluster",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} clusters [{elapsed}<{remaining}, {rate_fmt}]"
    ) as pbar:
        
        for cluster_id, cluster_indices in pbar:
            cluster_size = len(cluster_indices)
            pbar.set_description(f"Cluster {cluster_id} (size: {cluster_size})")
            
            cluster_data = df.loc[cluster_indices]
            cluster_smiles_list = cluster_data.sort_index()['smiles'].tolist()
            
            # For each molecule in the cluster
            for row_idx in sorted(cluster_indices):
                reference_smiles = df.loc[row_idx, 'smiles']
                
                similarities = calculate_tanimoto_similarities_for_molecule(
                    reference_smiles, cluster_smiles_list, use_cuda=use_cuda
                )
                
                if len(similarities) > 1:
                    similarity_str = '[' + ', '.join([f'{sim:.2f}' for sim in similarities]) + ']'
                else:
                    similarity_str = '[1.00]'
                
                df.loc[row_idx, 'tanimoto_similarities'] = similarity_str
    
    # Re-enable RDKit logging
    RDLogger.EnableLog('rdApp.*')
    
    print("✅ Similarity calculations complete")
    return df

def print_clustering_summary(results, args):
    """Print a nice summary of clustering results"""
    print("\n" + "="*60)
    print("🎯 CLUSTERING SUMMARY")
    print("="*60)
    
    total_molecules = len(results)
    total_clusters = results['cluster_id'].nunique()
    
    print(f"📊 Dataset Statistics:")
    print(f"   • Total molecules: {total_molecules:,}")
    print(f"   • Total clusters: {total_clusters:,}")
    print(f"   • Average cluster size: {total_molecules/total_clusters:.1f}")
    print(f"   • Clustering threshold: {args.threshold} (similarity ≥ {1-args.threshold:.2f})")
    
    # Cluster size distribution
    cluster_sizes = results.groupby('cluster_id').size()
    print(f"\n📈 Cluster Size Distribution:")
    
    size_ranges = [
        (1, 1, "Singletons"),
        (2, 5, "Small (2-5)"),
        (6, 20, "Medium (6-20)"),
        (21, 50, "Large (21-50)"),
        (51, float('inf'), "Very Large (50+)")
    ]
    
    for min_size, max_size, label in size_ranges:
        if max_size == float('inf'):
            count = (cluster_sizes >= min_size).sum()
        else:
            count = ((cluster_sizes >= min_size) & (cluster_sizes <= max_size)).sum()
        
        if count > 0:
            percentage = count / total_clusters * 100
            print(f"   • {label}: {count:,} clusters ({percentage:.1f}%)")
    
    # Show largest clusters
    if total_clusters > 1:
        largest_clusters = cluster_sizes.nlargest(5)
        print(f"\n🏆 Largest Clusters:")
        for cluster_id, size in largest_clusters.items():
            print(f"   • Cluster {cluster_id}: {size} molecules")
    
    print(f"\n✅ Results saved to: {args.output}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='SMILES Molecular Clustering Tool with CUDA Support')
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV file containing SMILES')
    parser.add_argument('--output', '-o', required=True,
                        help='Output CSV file for clustering results')
    parser.add_argument('--custom', action='store_true',
                        help='Use custom data format')
    parser.add_argument('--smiles_col', default='smiles',
                        help='Name of SMILES column (default: smiles)')
    parser.add_argument('--id_col', default='molecule_id',
                        help='Name of molecule ID column (default: molecule_id)')
    
    # Clustering parameters
    parser.add_argument('--threshold', '-t', type=float, default=0.3,
                        help='Distance threshold for clustering (default: 0.3, similarity ≥ 0.7)')
    parser.add_argument('--radius', '-r', type=int, default=2,
                        help='Morgan fingerprint radius (default: 2)')
    parser.add_argument('--n_bits', '-b', type=int, default=2048,
                        help='Fingerprint length in bits (default: 2048)')
    
    # Clustering mode
    parser.add_argument('--force-full-clustering', action='store_true',
                        help='Force full clustering without sampling')
    parser.add_argument('--sample-size', type=int, default=10000,
                        help='Sample size for large datasets (default: 10000)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Chunk size for memory management (default: 1000)')
    
    # Hardware options
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration if available')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU-only processing')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA GPU acceleration (requires CuPy)')
    
    # Two-stage clustering
    parser.add_argument('--two-stage', action='store_true',
                        help='Use two-stage clustering')
    parser.add_argument('--reference-file',
                        help='Reference molecules file for two-stage clustering')
    
    # Performance options
    parser.add_argument('--optimize-large', action='store_true',
                        help='Use optimized settings for large datasets (>50K molecules)')
    
    # Advanced options
    parser.add_argument('--tight-clustering', action='store_true',
                        help='Use tight clustering parameters')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Check for conflicting options
    if args.cuda and args.cpu_only:
        parser.error("Cannot use both --cuda and --cpu-only")
    
    if args.cuda and not args.gpu:
        args.gpu = True
    
    # Setup and check GPU
    gpu_available = setup_gpu_info()
    
    # Handle GPU not available when requested
    if args.cuda and not gpu_available:
        if not prompt_cpu_fallback(args):
            sys.exit(1)
        args.cuda = False
        args.gpu = False
        args.cpu_only = True
        print("🔄 Reconfigured to use CPU-only processing")
    
    # Print configuration
    print(f"\n⚙️  CLUSTERING CONFIGURATION")
    print("="*60)
    print(f"🎯 Threshold: {args.threshold} (similarity ≥ {1-args.threshold:.2f})")
    print(f"🧬 Fingerprint: Morgan radius={args.radius}, bits={args.n_bits}")
    
    mode_str = "Full clustering" if args.force_full_clustering else f"Sample-based ({args.sample_size:,})"
    hardware_str = "CUDA GPU" if args.cuda else "CPU"
    print(f"⚡ Mode: {mode_str} on {hardware_str}")
    
    if args.force_full_clustering and args.cpu_only:
        print("⚠️  Note: Full clustering on CPU may take significant time for large datasets")
    
    print("="*60)
    
    try:
        # Create clusterer
        clusterer = create_clusterer(args)
        
        # Run clustering with progress tracking
        print(f"\n🚀 Starting clustering process...")
        
        if args.two_stage:
            results = run_two_stage_clustering(clusterer, args)
        else:
            results = run_standard_clustering(clusterer, args)
        
        # Add detailed similarities - ensure consistent data types
        results = update_tanimoto_similarities(results, use_cuda=args.cuda)
        
        # Ensure results DataFrame is on CPU before saving
        if hasattr(results, 'values') and hasattr(results.values, 'get'):
            # If somehow we have CuPy arrays in DataFrame, convert to CPU
            results = results.copy()
            for col in results.columns:
                if hasattr(results[col].values, 'get'):
                    results[col] = results[col].values.get()
        
        # Save final results
        with tqdm(total=1, desc="Saving results", bar_format="{l_bar}{bar}| {desc}") as pbar:
            results.to_csv(args.output, index=False, encoding='utf-8')
            pbar.update(1)
        
        # Print summary
        print_clustering_summary(results, args)
        
    except KeyboardInterrupt:
        print("\n❌ Clustering interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
