#!/usr/bin/env python3
"""
Similarity-Based Molecular Clustering Tool

This script performs molecular clustering based on Tanimoto similarity matrix.
Each molecule is clustered with all other molecules that have similarity >= threshold.

Usage:
    python run_simclustering.py --input test.csv --output test_clustered.csv --sim_threshold 0.7
    python run_simclustering.py --verbose --custom --input test.csv --output test_clustered.csv --smiles_col "SMILES" --id_col "Molecule ID" --sim_threshold 0.7 --cuda

Author: GitHub Copilot
Date: 2025
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
import time
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json

# Import clustering dependencies
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
from sklearn.metrics import pairwise_distances
import pickle

# GPU support (optional)
GPU_AVAILABLE = False
GPU_BACKEND = None

try:
    import cupy as cp
    GPU_BACKEND = "cupy"
    GPU_AVAILABLE = True
    print(f"🚀 GPU Backend: CuPy available")
except ImportError:
    try:
        import torch
        if torch.cuda.is_available():
            GPU_BACKEND = "pytorch"
            GPU_AVAILABLE = True
            print(f"🚀 GPU Backend: PyTorch available")
    except ImportError:
        print("💻 No GPU backend available - using CPU only")

warnings.filterwarnings('ignore')

class SimilarityClusterer:
    """
    Similarity-based molecular clustering that creates similarity matrix
    and clusters molecules based on threshold.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, radius: int = 2, 
                 n_bits: int = 2048, use_gpu: bool = False, verbose: bool = False):
        """
        Initialize similarity clusterer.
        
        Args:
            similarity_threshold: Minimum Tanimoto similarity for clustering (default: 0.7)
            radius: Morgan fingerprint radius (default: 2)
            n_bits: Fingerprint bit size (default: 2048)
            use_gpu: Whether to use GPU acceleration (default: False)
            verbose: Whether to print detailed progress (default: False)
        """
        self.similarity_threshold = similarity_threshold
        self.radius = radius
        self.n_bits = n_bits
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.verbose = verbose
        self.gpu_backend = GPU_BACKEND if self.use_gpu else None
        
        # Initialize Morgan fingerprint generator
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits
        )
        
        if self.verbose:
            print(f"🎯 Similarity Clustering Configuration:")
            print(f"   - Similarity threshold: {self.similarity_threshold}")
            print(f"   - Morgan radius: {self.radius}")
            print(f"   - Fingerprint bits: {self.n_bits}")
            print(f"   - GPU acceleration: {'✅ Enabled' if self.use_gpu else '❌ Disabled'}")
            if self.use_gpu:
                print(f"   - GPU backend: {self.gpu_backend}")
    
    def calculate_fingerprints(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[int]]:
        """
        Calculate Morgan fingerprints for SMILES list.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (fingerprints_array, valid_indices)
        """
        if self.verbose:
            print(f"🧬 Calculating Morgan fingerprints for {len(smiles_list)} molecules...")
        
        fingerprints = []
        valid_indices = []
        error_count = 0
        
        with tqdm(total=len(smiles_list), desc="Processing SMILES", 
                 disable=not self.verbose, unit="mol") as pbar:
            
            for i, smiles in enumerate(smiles_list):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        fp = self.morgan_generator.GetFingerprintAsNumPy(mol)
                        fingerprints.append(fp)
                        valid_indices.append(i)
                    else:
                        error_count += 1
                        if self.verbose and error_count <= 5:
                            print(f"   ⚠️ Invalid SMILES at index {i}")
                except Exception as e:
                    error_count += 1
                    if self.verbose and error_count <= 5:
                        print(f"   ❌ Error processing SMILES at index {i}")
                
                pbar.update(1)
        
        if len(fingerprints) == 0:
            raise ValueError("No valid fingerprints could be generated")
        
        fingerprints_array = np.array(fingerprints)
        
        if self.verbose:
            print(f"   ✅ Valid fingerprints: {len(fingerprints)}/{len(smiles_list)} ({len(fingerprints)/len(smiles_list)*100:.1f}%)")
            if error_count > 0:
                print(f"   ❌ Invalid SMILES: {error_count}")
        
        return fingerprints_array, valid_indices
    
    def calculate_similarity_matrix(self, fingerprints: np.ndarray) -> np.ndarray:
        """
        Calculate full Tanimoto similarity matrix.
        
        Args:
            fingerprints: Array of molecular fingerprints
            
        Returns:
            Similarity matrix (n_molecules x n_molecules)
        """
        n_molecules = len(fingerprints)
        
        if self.verbose:
            print(f"🧮 Calculating {n_molecules}x{n_molecules} similarity matrix...")
            memory_gb = (n_molecules * n_molecules * 8) / (1024**3)
            print(f"   📊 Estimated memory: {memory_gb:.2f} GB")
        
        # Safety check for very large datasets
        if n_molecules > 20000 and not hasattr(self, 'force_full_matrix'):
            memory_gb = (n_molecules * n_molecules * 8) / (1024**3)
            print(f"⚠️  WARNING: Large dataset detected ({n_molecules:,} molecules)")
            print(f"   💾 Memory requirement: {memory_gb:.1f} GB")
            print(f"   ⏱️  Estimated processing time: {n_molecules/1000:.1f}+ minutes")
            
            if memory_gb > 20:  # More than 20GB
                print(f"❌ Memory requirement too high for standard processing!")
                print(f"💡 Use streaming mode: --streaming (no user input required)")
                print(f"💡 Or force full matrix: --force_full_matrix")
                
                if hasattr(self, 'streaming_mode') and self.streaming_mode:
                    print(f"✅ Using streaming mode - no full matrix will be created")
                    return self._calculate_similarity_streaming(fingerprints)
                
                response = input("Continue with full matrix anyway? (y/N): ").strip().lower()
                if response != 'y':
                    raise ValueError("Processing cancelled due to memory constraints")
        
        if self.use_gpu and self.gpu_backend == "cupy":
            return self._calculate_similarity_matrix_gpu(fingerprints)
        else:
            return self._calculate_similarity_matrix_cpu(fingerprints)
    
    def _calculate_similarity_matrix_gpu(self, fingerprints: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix using GPU acceleration."""
        try:
            import cupy as cp
            
            if self.verbose:
                print("   🚀 Using GPU (CuPy) for similarity calculation...")
            
            # Transfer to GPU
            fingerprints_gpu = cp.asarray(fingerprints.astype(np.float32))
            n_molecules = len(fingerprints_gpu)
            
            # Calculate similarity matrix on GPU
            similarity_matrix_gpu = cp.zeros((n_molecules, n_molecules), dtype=cp.float32)
            
            # Vectorized Tanimoto similarity calculation
            with tqdm(total=n_molecules, desc="GPU similarity", 
                     disable=not self.verbose, unit="mol") as pbar:
                
                for i in range(n_molecules):
                    # Expand dimensions for broadcasting
                    fp_i = fingerprints_gpu[i:i+1, :]  # (1, n_bits)
                    fp_all = fingerprints_gpu  # (n_molecules, n_bits)
                    
                    # Tanimoto similarity calculation
                    intersection = cp.sum(fp_i * fp_all, axis=1)
                    union = cp.sum(cp.maximum(fp_i, fp_all), axis=1)
                    
                    # Avoid division by zero
                    similarities = cp.where(union > 0, intersection / union, 0.0)
                    similarity_matrix_gpu[i, :] = similarities
                    
                    pbar.update(1)
            
            # Transfer back to CPU
            similarity_matrix = cp.asnumpy(similarity_matrix_gpu)
            
            if self.verbose:
                print("   ✅ GPU similarity calculation completed")
            
            return similarity_matrix
            
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ GPU calculation failed: {e}")
                print("   🔄 Falling back to CPU...")
            return self._calculate_similarity_matrix_cpu(fingerprints)
    
    def _calculate_similarity_matrix_cpu(self, fingerprints: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix using CPU."""
        if self.verbose:
            print("   💻 Using CPU for similarity calculation...")
        
        # Use sklearn for efficient Jaccard similarity calculation
        # Note: Jaccard distance = 1 - Tanimoto similarity
        with tqdm(total=1, desc="CPU similarity", disable=not self.verbose) as pbar:
            distance_matrix = pairwise_distances(fingerprints, metric='jaccard', n_jobs=-1)
            similarity_matrix = 1.0 - distance_matrix
            pbar.update(1)
        
        if self.verbose:
            print("   ✅ CPU similarity calculation completed")
        
        return similarity_matrix
    
    def create_similarity_clusters(self, similarity_matrix: np.ndarray, 
                                 molecule_ids: List[str]) -> pd.DataFrame:
        """
        Create similarity-based clusters where each molecule is associated
        with all other molecules that have similarity >= threshold.
        
        Args:
            similarity_matrix: Precomputed similarity matrix
            molecule_ids: List of molecule identifiers
            
        Returns:
            DataFrame with clustering results
        """
        n_molecules = len(similarity_matrix)
        
        if self.verbose:
            print(f"🎯 Creating similarity clusters with threshold {self.similarity_threshold}...")
        
        # First pass: determine cluster assignments and collect all similarities
        cluster_assignments = {}
        cluster_members = {}  # cluster_id -> set of all member molecules
        molecule_similarities = {}  # molecule_id -> dict of {neighbor_id: similarity}
        next_cluster_id = 1
        
        with tqdm(total=n_molecules, desc="Assigning clusters", 
                 disable=not self.verbose, unit="mol") as pbar:
            
            for i in range(n_molecules):
                current_mol_id = molecule_ids[i]
                
                # Find all molecules with similarity >= threshold
                similar_indices = np.where(similarity_matrix[i] >= self.similarity_threshold)[0]
                cluster_ids = [molecule_ids[idx] for idx in similar_indices]
                
                # Store similarities for this molecule
                mol_sims = {}
                for idx in similar_indices:
                    mol_sims[molecule_ids[idx]] = round(float(similarity_matrix[i, idx]), 2)
                molecule_similarities[current_mol_id] = mol_sims
                
                # Check if any similar molecules already have cluster IDs
                existing_cluster_ids = set()
                for mol_id in cluster_ids:
                    if mol_id in cluster_assignments:
                        existing_cluster_ids.add(cluster_assignments[mol_id])
                
                if existing_cluster_ids:
                    # Use the lowest existing cluster ID and merge clusters if needed
                    cluster_id = min(existing_cluster_ids)
                    
                    # Merge all existing clusters into the lowest ID
                    all_members = set(cluster_ids)
                    for old_cluster_id in existing_cluster_ids:
                        if old_cluster_id in cluster_members:
                            all_members.update(cluster_members[old_cluster_id])
                            if old_cluster_id != cluster_id:
                                del cluster_members[old_cluster_id]
                    
                    cluster_members[cluster_id] = all_members
                    
                    # Update assignments for all members
                    for mol_id in all_members:
                        cluster_assignments[mol_id] = cluster_id
                else:
                    # Create new cluster
                    cluster_id = next_cluster_id
                    next_cluster_id += 1
                    cluster_members[cluster_id] = set(cluster_ids)
                    
                    # Assign cluster ID to all molecules in this cluster
                    for mol_id in cluster_ids:
                        cluster_assignments[mol_id] = cluster_id
                
                pbar.update(1)
        
        # Second pass: create final results with consistent cluster members
        results = []
        
        with tqdm(total=n_molecules, desc="Building results", 
                 disable=not self.verbose, unit="mol") as pbar:
            
            for i, mol_id in enumerate(molecule_ids):
                cluster_id = cluster_assignments[mol_id]
                all_cluster_members = list(cluster_members[cluster_id])
                
                # Get similarities for this molecule with all cluster members
                mol_sims = molecule_similarities[mol_id]
                
                # Create lists of members and their similarities (only for molecules this one is similar to)
                member_sim_pairs = []
                for member in all_cluster_members:
                    if member in mol_sims:
                        member_sim_pairs.append((member, mol_sims[member]))
                
                # Sort by similarity (descending)
                member_sim_pairs.sort(key=lambda x: x[1], reverse=True)
                sorted_members, sorted_sims = zip(*member_sim_pairs) if member_sim_pairs else ([], [])
                
                # Calculate consecutive pairwise similarities in cluster_members order
                consecutive_sims = []
                if len(all_cluster_members) > 1:
                    for j in range(len(all_cluster_members) - 1):
                        mol_a = all_cluster_members[j]
                        mol_b = all_cluster_members[j + 1]
                        
                        # Check if we have the similarity stored (from either direction)
                        sim_score = None
                        if mol_a in molecule_similarities and mol_b in molecule_similarities[mol_a]:
                            sim_score = molecule_similarities[mol_a][mol_b]
                        elif mol_b in molecule_similarities and mol_a in molecule_similarities[mol_b]:
                            sim_score = molecule_similarities[mol_b][mol_a]
                        
                        if sim_score is not None:
                            consecutive_sims.append(f"{mol_a}→{mol_b}:{sim_score}")
                        else:
                            consecutive_sims.append(f"{mol_a}→{mol_b}:N/A")
                
                results.append({
                    'molecule_id': mol_id,
                    'cluster_members': all_cluster_members,
                    'similar_members': list(sorted_members),
                    'similarity_scores': list(sorted_sims),
                    'consecutive_similarities': consecutive_sims
                })
                
                pbar.update(1)
        
        if self.verbose:
            unique_clusters = len(set([frozenset(r['cluster_members']) for r in results]))
            print(f"   🆔 Unique clusters found: {unique_clusters}")
            print(f"   📊 Total molecule assignments: {len(results)}")
        
        # Convert results to DataFrame
        return pd.DataFrame(results)
    
    def save_results(self, results: List[Dict], output_file: str, similarity_threshold: float, append_mode: bool = False):
        """
        Save clustering results to CSV file with option to append
        
        Args:
            results: List of result dictionaries
            output_file: Output CSV file path
            similarity_threshold: Threshold used for this run
            append_mode: If True, append to existing file with threshold info
        """
        df_results = pd.DataFrame(results)
        
        # Add threshold information to distinguish different runs
        df_results['similarity_threshold'] = similarity_threshold
        
        if append_mode and os.path.exists(output_file):
            # Read existing data and append new results
            try:
                existing_df = pd.read_csv(output_file)
                combined_df = pd.concat([existing_df, df_results], ignore_index=True)
                combined_df.to_csv(output_file, index=False)
                if self.verbose:
                    print(f"   ✅ Appended {len(df_results)} results (threshold={similarity_threshold}) to {output_file}")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Could not append to existing file: {e}")
                    print(f"   � Creating new file instead")
                df_results.to_csv(output_file, index=False)
        else:
            # Save as new file
            df_results.to_csv(output_file, index=False)
            if self.verbose:
                print(f"   ✅ Saved {len(df_results)} results (threshold={similarity_threshold}) to {output_file}")
        
        return df_results
    
    def _calculate_similarity_streaming(self, fingerprints: np.ndarray, 
                                       molecule_ids: List[str]) -> pd.DataFrame:
        """
        Memory-efficient streaming similarity calculation - no full matrix stored.
        Processes similarities row by row and directly builds clusters.
        """
        n_molecules = len(fingerprints)
        
        if self.verbose:
            print(f"🌊 Streaming similarity calculation for {n_molecules:,} molecules...")
            print(f"   💾 Memory efficient: No full matrix will be stored")
            print(f"   🎯 Threshold: {self.similarity_threshold}")
        
        # First pass: collect all similarities and cluster assignments
        molecule_similarities = {}  # molecule_id -> dict of {neighbor_id: similarity}
        cluster_assignments = {}  
        cluster_members = {}  
        next_cluster_id = 1
        
        # Move fingerprints to GPU if available
        if self.use_gpu and self.gpu_backend == "cupy":
            try:
                import cupy as cp
                fingerprints_gpu = cp.asarray(fingerprints.astype(np.float32))
                use_gpu = True
                if self.verbose:
                    print(f"   🚀 Using GPU (CuPy) for streaming calculation...")
            except:
                fingerprints_gpu = fingerprints
                use_gpu = False
                if self.verbose:
                    print(f"   💻 Using CPU for streaming calculation...")
        else:
            fingerprints_gpu = fingerprints
            use_gpu = False
            if self.verbose:
                print(f"   💻 Using CPU for streaming calculation...")
        
        # Process each molecule
        with tqdm(total=n_molecules, desc="Streaming clusters", 
                 disable=not self.verbose, unit="mol") as pbar:
            
            for i in range(n_molecules):
                if use_gpu:
                    # GPU calculation
                    fp_i = fingerprints_gpu[i:i+1, :]  # (1, n_bits)
                    
                    # Tanimoto similarity calculation
                    intersection = cp.sum(fp_i * fingerprints_gpu, axis=1)
                    union = cp.sum(cp.maximum(fp_i, fingerprints_gpu), axis=1)
                    similarities = cp.where(union > 0, intersection / union, 0.0)
                    
                    # Find similar molecules
                    similar_mask = similarities >= self.similarity_threshold
                    similar_indices = cp.where(similar_mask)[0]
                    cluster_similarities = similarities[similar_mask]
                    
                    # Convert back to CPU
                    similar_indices = cp.asnumpy(similar_indices)
                    cluster_similarities = cp.asnumpy(cluster_similarities)
                    
                else:
                    # CPU calculation
                    fp_i = fingerprints[i:i+1, :]
                    
                    # Vectorized Tanimoto similarity
                    intersection = np.sum(fp_i * fingerprints, axis=1)
                    union = np.sum(np.maximum(fp_i, fingerprints), axis=1)
                    similarities = np.where(union > 0, intersection / union, 0.0)
                    
                    # Find similar molecules
                    similar_indices = np.where(similarities >= self.similarity_threshold)[0]
                    cluster_similarities = similarities[similar_indices]
                
                # Get molecule IDs and similarities (rounded to 2 decimal places)
                cluster_ids = [molecule_ids[idx] for idx in similar_indices]
                cluster_sims = [round(float(sim), 2) for sim in cluster_similarities]
                
                # Store similarities for this molecule
                current_mol_id = molecule_ids[i]
                mol_sims = {}
                for j, mol_id in enumerate(cluster_ids):
                    mol_sims[mol_id] = cluster_sims[j]
                molecule_similarities[current_mol_id] = mol_sims
                
                # Determine cluster ID
                # Check if any similar molecules already have cluster IDs
                existing_cluster_ids = set()
                for mol_id in cluster_ids:
                    if mol_id in cluster_assignments:
                        existing_cluster_ids.add(cluster_assignments[mol_id])
                
                if existing_cluster_ids:
                    # Use the lowest existing cluster ID and merge clusters if needed
                    cluster_id = min(existing_cluster_ids)
                    
                    # Merge all existing clusters into the lowest ID
                    all_members = set(cluster_ids)
                    for old_cluster_id in existing_cluster_ids:
                        if old_cluster_id in cluster_members:
                            all_members.update(cluster_members[old_cluster_id])
                            if old_cluster_id != cluster_id:
                                del cluster_members[old_cluster_id]
                    
                    cluster_members[cluster_id] = all_members
                    
                    # Update assignments for all members
                    for mol_id in all_members:
                        cluster_assignments[mol_id] = cluster_id
                else:
                    # Create new cluster ID
                    cluster_id = next_cluster_id
                    next_cluster_id += 1
                    cluster_members[cluster_id] = set(cluster_ids)
                    
                    # Assign cluster ID to all molecules in this cluster
                    for mol_id in cluster_ids:
                        cluster_assignments[mol_id] = cluster_id
                
                pbar.update(1)
        
        # Second pass: create final results with consistent cluster members
        results = []
        
        with tqdm(total=n_molecules, desc="Building results", 
                 disable=not self.verbose, unit="mol") as pbar:
            
            for i, mol_id in enumerate(molecule_ids):
                cluster_id = cluster_assignments[mol_id]
                all_cluster_members = list(cluster_members[cluster_id])
                
                # Get similarities for this molecule with all cluster members
                mol_sims = molecule_similarities[mol_id]
                
                # Create lists of members and their similarities (only for molecules this one is similar to)
                member_sim_pairs = []
                for member in all_cluster_members:
                    if member in mol_sims:
                        member_sim_pairs.append((member, mol_sims[member]))
                
                # Sort by similarity (descending)
                member_sim_pairs.sort(key=lambda x: x[1], reverse=True)
                sorted_members, sorted_sims = zip(*member_sim_pairs) if member_sim_pairs else ([], [])
                
                # Calculate consecutive pairwise similarities in cluster_members order
                consecutive_sims = []
                if len(all_cluster_members) > 1:
                    for j in range(len(all_cluster_members) - 1):
                        mol_a = all_cluster_members[j]
                        mol_b = all_cluster_members[j + 1]
                        
                        # Check if we have the similarity stored (from either direction)
                        sim_score = None
                        if mol_a in molecule_similarities and mol_b in molecule_similarities[mol_a]:
                            sim_score = molecule_similarities[mol_a][mol_b]
                        elif mol_b in molecule_similarities and mol_a in molecule_similarities[mol_b]:
                            sim_score = molecule_similarities[mol_b][mol_a]
                        
                        if sim_score is not None:
                            consecutive_sims.append(f"{mol_a}→{mol_b}:{sim_score}")
                        else:
                            consecutive_sims.append(f"{mol_a}→{mol_b}:N/A")
                
                results.append({
                    'molecule_id': mol_id,
                    'cluster_id': cluster_id,
                    'cluster_size': len(all_cluster_members),
                    'cluster_members': all_cluster_members,  # All members in cluster
                    'similar_members': list(sorted_members),  # Only members this molecule is similar to
                    'similarity_scores': list(sorted_sims),
                    'consecutive_similarities': consecutive_sims,  # A→B, B→C, C→D chain similarities
                    'max_similarity': round(max(sorted_sims), 2) if sorted_sims else 0.0,
                    'min_similarity': round(min(sorted_sims), 2) if sorted_sims else 0.0,
                    'avg_similarity': round(np.mean(sorted_sims), 2) if sorted_sims else 0.0
                })
                
                pbar.update(1)
        
        df_results = pd.DataFrame(results)
        
        if self.verbose:
            print(f"   ✅ Streaming clustering completed!")
            print(f"   🆔 Total unique clusters: {len(cluster_members)}")
            print(f"   📊 Average cluster size: {df_results['cluster_size'].mean():.2f}")
            print(f"   📊 Max cluster size: {df_results['cluster_size'].max()}")
            print(f"   📊 Min cluster size: {df_results['cluster_size'].min()}")
        
        return df_results

    def run_similarity_clustering(self, df: pd.DataFrame, smiles_col: str,
                                id_col: str, streaming_mode: bool = False) -> pd.DataFrame:
        """
        Run complete similarity-based clustering pipeline.
        
        Args:
            df: Input DataFrame with SMILES and IDs
            smiles_col: Name of SMILES column
            id_col: Name of molecule ID column
            streaming_mode: Use memory-efficient streaming mode
            
        Returns:
            DataFrame with clustering results
        """
        if self.verbose:
            print(f"🚀 Starting similarity-based clustering...")
            print(f"   📊 Input: {len(df)} molecules")
            print(f"   🧬 SMILES column: '{smiles_col}'")
            print(f"   🏷️ ID column: '{id_col}'")
            print(f"   🌊 Streaming mode: {'✅ Enabled' if streaming_mode else '❌ Disabled'}")
        
        # Step 1: Calculate fingerprints
        fingerprints, valid_indices = self.calculate_fingerprints(df[smiles_col].tolist())
        
        # Get valid molecule IDs
        valid_molecule_ids = [df.iloc[i][id_col] for i in valid_indices]
        
        # Step 2: Calculate similarities and create clusters
        if streaming_mode:
            # Streaming mode - no full matrix
            results_df = self._calculate_similarity_streaming(fingerprints, valid_molecule_ids)
        else:
            # Traditional mode - full matrix
            similarity_matrix = self.calculate_similarity_matrix(fingerprints)
            results_df = self.create_similarity_clusters(similarity_matrix, valid_molecule_ids)
        
        # Step 3: Add original data
        valid_df = df.iloc[valid_indices].reset_index(drop=True)
        results_df = results_df.merge(valid_df, left_on='molecule_id', right_on=id_col, how='left')
        
        if self.verbose:
            print(f"✅ Similarity clustering completed!")
            print(f"   📊 Output: {len(results_df)} clustered molecules")
        
        return results_df

    def cluster_molecules_multi_threshold(self, df, thresholds, verbose=False):
        """
        Cluster molecules with multiple thresholds and create separate columns for each threshold
        
        Args:
            df: DataFrame with molecules
            thresholds: List of similarity thresholds
            verbose: Enable verbose output
            
        Returns:
            DataFrame with threshold-specific columns
        """
        # Initialize results dictionary with molecule IDs
        molecules_dict = {}
        for _, row in df.iterrows():
            mol_id = row.iloc[1] if len(row) > 1 else row.iloc[0]  # Get molecule ID
            molecules_dict[mol_id] = {'molecule_id': mol_id}
        
        # Run clustering for each threshold
        for threshold in thresholds:
            if verbose:
                print(f"\n{'='*50}")
                print(f"🎯 CLUSTERING WITH THRESHOLD: {threshold}")
                print(f"{'='*50}")
            
            # Update threshold for this run
            original_threshold = self.similarity_threshold
            self.similarity_threshold = threshold
            
            # Perform clustering
            results = self.cluster_molecules(df)
            
            # Create threshold-specific column names
            threshold_str = str(threshold).replace('.', '_')
            col_cluster_members = f'cluster_members_{threshold_str}'
            col_similar_members = f'similar_members_{threshold_str}'
            col_similarity_scores = f'similarity_scores_{threshold_str}'
            col_consecutive_similarities = f'consecutive_similarities_{threshold_str}'
            
            # Add results for this threshold to molecules_dict
            for result in results:
                mol_id = result['molecule_id']
                if mol_id in molecules_dict:
                    molecules_dict[mol_id][col_cluster_members] = result['cluster_members']
                    molecules_dict[mol_id][col_similar_members] = result['similar_members']
                    molecules_dict[mol_id][col_similarity_scores] = result['similarity_scores']
                    molecules_dict[mol_id][col_consecutive_similarities] = result['consecutive_similarities']
            
            if verbose:
                unique_clusters = len(set([frozenset(r['cluster_members']) for r in results]))
                print(f"   🆔 Unique clusters found: {unique_clusters}")
                print(f"   📊 Total molecule assignments: {len(results)}")
            
            # Restore original threshold
            self.similarity_threshold = original_threshold
        
        # Convert to DataFrame
        result_list = list(molecules_dict.values())
        return result_list

    # ...existing code...
def save_results(results_df: pd.DataFrame, output_file: str, verbose: bool = False):
    """Save clustering results to file."""
    if verbose:
        print(f"💾 Saving results to: {output_file}")
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save based on file extension
    if output_file.endswith('.csv'):
        results_df.to_csv(output_file, index=False)
    elif output_file.endswith(('.xlsx', '.xls')):
        results_df.to_excel(output_file, index=False)
    else:
        # Default to CSV
        results_df.to_csv(output_file, index=False)
    
    if verbose:
        print(f"   ✅ Saved {len(results_df)} clustered molecules")
        print(f"   📊 Output columns: {list(results_df.columns)}")


def print_summary_stats(results_df: pd.DataFrame, verbose: bool = False):
    """Print summary statistics of clustering results."""
    if not verbose:
        return
    
    # Handle column name conflicts from merge operation
    cluster_size_col = None
    if 'cluster_size' in results_df.columns:
        cluster_size_col = 'cluster_size'
    elif 'cluster_size_x' in results_df.columns:
        cluster_size_col = 'cluster_size_x'
    elif 'cluster_size_y' in results_df.columns:
        cluster_size_col = 'cluster_size_y'
    
    print(f"\n📈 Clustering Summary Statistics:")
    print(f"   Total molecules: {len(results_df):,}")
    
    if cluster_size_col:
        print(f"   Average cluster size: {results_df[cluster_size_col].mean():.2f}")
        print(f"   Largest cluster: {results_df[cluster_size_col].max()}")
        print(f"   Smallest cluster: {results_df[cluster_size_col].min()}")
        print(f"   Singletons (cluster size = 1): {sum(results_df[cluster_size_col] == 1):,}")
        
        # Cluster size distribution
        size_dist = results_df[cluster_size_col].value_counts().sort_index()
        print(f"\n📊 Cluster Size Distribution (top 10):")
        for size, count in size_dist.head(10).items():
            print(f"   Size {size}: {count:,} molecules")
    
    # Similarity statistics
    if 'max_similarity' in results_df.columns:
        print(f"\n🎯 Similarity Statistics:")
        print(f"   Average max similarity: {results_df['max_similarity'].mean():.3f}")
        print(f"   Average min similarity: {results_df['min_similarity'].mean():.3f}")
        print(f"   Average avg similarity: {results_df['avg_similarity'].mean():.3f}")


def find_smiles_column(df):
    """Find SMILES column with case-insensitive search"""
    possible_names = ['smiles', 'SMILES', 'Smiles', 'smi', 'SMI']
    for col in df.columns:
        if col.lower() in [name.lower() for name in possible_names]:
            return col
    return None


def load_molecules(file_path, smiles_col=None, id_col=None):
    """Load molecules from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"📊 Loaded {len(df)} molecules from {file_path}")
        
        # Auto-detect SMILES column if not specified
        if smiles_col is None:
            # Look for common SMILES column names (case-insensitive)
            smiles_candidates = ['smiles', 'SMILES', 'Smiles', 'smi', 'SMI', 'molecule', 'structure']
            for candidate in smiles_candidates:
                if candidate in df.columns:
                    smiles_col = candidate
                    break
            
            if smiles_col is None:
                raise ValueError(f"SMILES column not found. Available columns: {list(df.columns)}")
        
        # Auto-detect ID column if not specified
        if id_col is None:
            # Look for common ID column names (case-insensitive)
            id_candidates = ['id', 'ID', 'Id', 'molecule_id', 'Molecule ID', 'mol_id', 'compound_id', 'name']
            for candidate in id_candidates:
                if candidate in df.columns:
                    id_col = candidate
                    break
            
            if id_col is None:
                # Use index if no ID column found
                df['molecule_id'] = df.index
                id_col = 'molecule_id'
        
        print(f"🔍 Using SMILES column: '{smiles_col}'")
        print(f"🏷️  Using ID column: '{id_col}'")
        
        # Validate columns exist
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found. Available columns: {list(df.columns)}")
        if id_col not in df.columns:
            raise ValueError(f"ID column '{id_col}' not found. Available columns: {list(df.columns)}")
        
        # Remove rows with missing SMILES
        initial_count = len(df)
        df = df.dropna(subset=[smiles_col])
        if len(df) < initial_count:
            print(f"⚠️  Removed {initial_count - len(df)} rows with missing SMILES")
        
        molecules = df[[id_col, smiles_col]].rename(columns={id_col: 'id', smiles_col: 'smiles'})
        return molecules
        
    except Exception as e:
        print(f"❌ Error loading molecules: {e}")
        raise


def run_multiple_thresholds_clustering():
    """
    Script to run clustering with multiple thresholds and append results
    """
    print("🧬 Multi-Threshold Molecular Clustering")
    print("="*50)
    
    # Example usage with your requirements
    input_file = input("📁 Enter input CSV file path: ").strip()
    output_file = input("📁 Enter output CSV file path: ").strip()
    
    # Default thresholds: 0.7 (default) and 0.6 (as requested)
    thresholds = [0.7, 0.6]
    
    print(f"🎯 Running with thresholds: {thresholds}")
    print(f"📊 Results will be appended to the same file with threshold info")
    
    try:
        # Load data once
        df = load_and_validate_data(input_file, 'SMILES', 'Molecule ID')
        
        all_results = []
        
        for i, threshold in enumerate(thresholds):
            print(f"\n{'='*40}")
            print(f"🎯 THRESHOLD: {threshold}")
            print(f"{'='*40}")
            
            # Create clusterer
            clusterer = SimilarityClusterer(
                similarity_threshold=threshold,
                radius=2,
                n_bits=2048,
                use_gpu=True,  # Try GPU
                streaming_mode=False,
                force_full_matrix=False,
                verbose=True
            )
            
            # Perform clustering
            results = clusterer.cluster_molecules(df)
            
            # Add threshold info to each result
            for result in results:
                result['similarity_threshold'] = threshold
            
            all_results.extend(results)
            
            print(f"✅ Completed threshold {threshold}")
        
        # Save all results to single file
        df_combined = pd.DataFrame(all_results)
        df_combined.to_csv(output_file, index=False)
        
        print(f"\n🎉 SUCCESS!")
        print(f"📁 Combined results saved to: {output_file}")
        print(f"📊 Total records: {len(all_results)}")
        print(f"🔗 Thresholds included: {thresholds}")
        
        # Show summary
        summary = df_combined.groupby('similarity_threshold').agg({
            'molecule_id': 'count',
            'cluster_members': lambda x: len(set([frozenset(eval(members)) for members in x]))
        }).round(2)
        print(f"\n📈 Summary by threshold:")
        print(summary)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='SMILES Similarity-Based Clustering Tool')
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV file containing SMILES')
    parser.add_argument('--output', '-o', required=True,
                        help='Output CSV file for clustering results')
    parser.add_argument('--smiles_col', default='SMILES',
                        help='Name of SMILES column (default: SMILES)')
    parser.add_argument('--id_col', default='Molecule ID',
                        help='Name of molecule ID column (default: Molecule ID)')
    
    # Clustering parameters - FIXED the typo here
    parser.add_argument('--sim_threshold', '-t', type=float, nargs='+', default=[0.6, 0.7],
                        help='Similarity threshold(s) for clustering (default: [0.6, 0.7]). Can specify multiple values like --sim_threshold 0.6 0.7')
    parser.add_argument('--radius', '-r', type=int, default=2,
                        help='Morgan fingerprint radius (default: 2)')
    parser.add_argument('--n_bits', '-b', type=int, default=2048,
                        help='Fingerprint length in bits (default: 2048)')
    
    # Performance options
    parser.add_argument('--cuda', '--gpu', action='store_true',
                        help='Use GPU acceleration if available')
    parser.add_argument('--streaming', action='store_true',
                        help='Use streaming mode for large datasets')
    parser.add_argument('--force_full_matrix', action='store_true',
                        help='Force full matrix calculation')
    parser.add_argument('--max_molecules', '-m', type=int,
                        help='Limit to first N molecules for testing')
    parser.add_argument('--random_sample', action='store_true',
                        help='Use random sampling instead of first N molecules')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Determine which thresholds to run
    thresholds = args.sim_threshold
    if args.verbose:
        print(f"🎯 Running clustering with thresholds: {thresholds}")
    
    try:
        # Load and validate data once
        df = pd.read_csv(args.input)
        
        # Validate required columns
        if args.smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{args.smiles_col}' not found in input file")
        if args.id_col not in df.columns:
            raise ValueError(f"ID column '{args.id_col}' not found in input file")
        
        if args.verbose:
            print(f"📊 Loaded {len(df)} molecules from {args.input}")
        
        # Apply molecule limit if specified
        if args.max_molecules:
            if args.random_sample:
                df = df.sample(n=min(args.max_molecules, len(df))).reset_index(drop=True)
                if args.verbose:
                    print(f"🎲 Using random sample of {len(df)} molecules")
            else:
                df = df.head(args.max_molecules).reset_index(drop=True)
                if args.verbose:
                    print(f"📊 Using first {len(df)} molecules")
        
        # Initialize results dictionary with molecule IDs
        molecules_dict = {}
        for _, row in df.iterrows():
            mol_id = row[args.id_col]
            molecules_dict[mol_id] = {'molecule_id': mol_id}
        
        # Run clustering for each threshold
        for i, threshold in enumerate(thresholds):
            if args.verbose:
                print(f"\n{'='*60}")
                print(f"🎯 CLUSTERING WITH SIMILARITY THRESHOLD: {threshold}")
                print(f"{'='*60}")
            
            # Create clusterer for this threshold
            clusterer = SimilarityClusterer(
                similarity_threshold=threshold,
                radius=args.radius,
                n_bits=args.n_bits,
                use_gpu=args.cuda,
                verbose=args.verbose
            )
            
            # Perform clustering
            results_df = clusterer.run_similarity_clustering(
                df, args.smiles_col, args.id_col, 
                streaming_mode=args.streaming
            )
            
            # Create threshold-specific column names
            threshold_str = str(threshold).replace('.', '_')
            col_cluster_members = f'cluster_members_{threshold_str}'
            col_similar_members = f'similar_members_{threshold_str}'
            col_similarity_scores = f'similarity_scores_{threshold_str}'
            col_consecutive_similarities = f'consecutive_similarities_{threshold_str}'
            
            # Add results for this threshold to molecules_dict
            for _, row in results_df.iterrows():
                mol_id = row['molecule_id']
                if mol_id in molecules_dict:
                    molecules_dict[mol_id][col_cluster_members] = row.get('cluster_members', [])
                    molecules_dict[mol_id][col_similar_members] = row.get('similar_members', [])
                    molecules_dict[mol_id][col_similarity_scores] = row.get('similarity_scores', [])
                    molecules_dict[mol_id][col_consecutive_similarities] = row.get('consecutive_similarities', [])
            
            if args.verbose:
                unique_clusters = len(set([frozenset(row.get('cluster_members', [])) for _, row in results_df.iterrows()]))
                print(f"   🆔 Unique clusters found: {unique_clusters}")
                print(f"   📊 Total molecule assignments: {len(results_df)}")
        
        # Convert to DataFrame and save results
        result_list = list(molecules_dict.values())
        df_results = pd.DataFrame(result_list)
        df_results.to_csv(args.output, index=False)
        
        if args.verbose:
            print(f"\n✅ All clustering completed!")
            print(f"📁 Results saved to: {args.output}")
            print(f"🔗 Multiple thresholds ({thresholds}) combined with separate columns")
            print(f"📊 Total records: {len(df_results)}")
            
            # Show column summary
            threshold_columns = [col for col in df_results.columns if any(str(t).replace('.', '_') in col for t in thresholds)]
            print(f"\n📈 Threshold-specific columns created:")
            for threshold in thresholds:
                threshold_str = str(threshold).replace('.', '_')
                threshold_cols = [col for col in threshold_columns if threshold_str in col]
                print(f"   Threshold {threshold}: {len(threshold_cols)} columns")
        
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
    # Check if this is being run as multi-threshold script
    if len(sys.argv) > 1 and sys.argv[1] == "--multi-threshold":
        run_multiple_thresholds_clustering()
    else:
        main()