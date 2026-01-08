"""
Evaluation Metrics for Clustering
Implements: Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI, Cluster Purity
"""

import numpy as np
import torch
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from collections import Counter


class ClusteringEvaluator:
    """
    Comprehensive evaluation suite for clustering algorithms
    """
    
    def __init__(self):
        self.metrics = {}
    
    def silhouette_coefficient(self, latent_vectors, cluster_labels):
        """
        Silhouette Score: s(i) = (b(i) - a(i)) / max(a(i), b(i))
        
        Measures how similar an object is to its own cluster compared to other clusters.
        Range: [-1, 1], higher is better.
        
        Args:
            latent_vectors: Feature vectors (n_samples, n_features)
            cluster_labels: Cluster assignments
        
        Returns:
            silhouette_score: float
        """
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.cpu().detach().numpy()
        if isinstance(cluster_labels, torch.Tensor):
            cluster_labels = cluster_labels.cpu().detach().numpy()
        
        # Filter out noise points (label = -1) if DBSCAN
        mask = cluster_labels != -1
        if mask.sum() < 2:
            return 0.0
        
        latent_filtered = latent_vectors[mask]
        labels_filtered = cluster_labels[mask]
        
        # Need at least 2 unique labels
        if len(np.unique(labels_filtered)) < 2:
            return 0.0
        
        score = silhouette_score(latent_filtered, labels_filtered)
        return score
    
    def calinski_harabasz_index(self, latent_vectors, cluster_labels):
        """
        Calinski-Harabasz Index (Variance Ratio Criterion)
        
        Ratio of between-cluster variance to within-cluster variance.
        Higher values indicate better clustering.
        
        CH = [trace(B_k) / (k-1)] / [trace(W_k) / (n-k)]
        where B_k is between-group dispersion matrix
        and W_k is within-group dispersion matrix
        
        Args:
            latent_vectors: Feature vectors
            cluster_labels: Cluster assignments
        
        Returns:
            ch_score: float
        """
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.cpu().detach().numpy()
        if isinstance(cluster_labels, torch.Tensor):
            cluster_labels = cluster_labels.cpu().detach().numpy()
        
        # Filter out noise points
        mask = cluster_labels != -1
        if mask.sum() < 2:
            return 0.0
        
        latent_filtered = latent_vectors[mask]
        labels_filtered = cluster_labels[mask]
        
        # Need at least 2 unique labels
        if len(np.unique(labels_filtered)) < 2:
            return 0.0
        
        score = calinski_harabasz_score(latent_filtered, labels_filtered)
        return score
    
    def davies_bouldin_index(self, latent_vectors, cluster_labels):
        """
        Davies-Bouldin Index
        
        Average similarity of each cluster with its most similar cluster.
        Lower values indicate better clustering.
        
        DB = (1/k) * Σ max_{i≠j} (σ_i + σ_j) / d(c_i, c_j)
        where σ_i is average distance within cluster i
        and d(c_i, c_j) is distance between cluster centroids
        
        Args:
            latent_vectors: Feature vectors
            cluster_labels: Cluster assignments
        
        Returns:
            db_score: float (lower is better)
        """
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.cpu().detach().numpy()
        if isinstance(cluster_labels, torch.Tensor):
            cluster_labels = cluster_labels.cpu().detach().numpy()
        
        # Filter out noise points
        mask = cluster_labels != -1
        if mask.sum() < 2:
            return np.inf
        
        latent_filtered = latent_vectors[mask]
        labels_filtered = cluster_labels[mask]
        
        # Need at least 2 unique labels
        if len(np.unique(labels_filtered)) < 2:
            return np.inf
        
        score = davies_bouldin_score(latent_filtered, labels_filtered)
        return score
    
    def adjusted_rand_index(self, true_labels, cluster_labels):
        """
        Adjusted Rand Index (ARI)
        
        Measures agreement between true labels and cluster assignments,
        adjusted for chance.
        
        Range: [-1, 1], 1 = perfect agreement, 0 = random, <0 = worse than random
        
        Args:
            true_labels: Ground truth labels
            cluster_labels: Predicted cluster assignments
        
        Returns:
            ari_score: float
        """
        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().detach().numpy()
        if isinstance(cluster_labels, torch.Tensor):
            cluster_labels = cluster_labels.cpu().detach().numpy()
        
        score = adjusted_rand_score(true_labels, cluster_labels)
        return score
    
    def normalized_mutual_information(self, true_labels, cluster_labels):
        """
        Normalized Mutual Information (NMI)
        
        Measures mutual information between true labels and clusters,
        normalized by entropy.
        
        NMI(U, V) = 2 * I(U; V) / [H(U) + H(V)]
        where I is mutual information and H is entropy
        
        Range: [0, 1], 1 = perfect agreement, 0 = independent
        
        Args:
            true_labels: Ground truth labels
            cluster_labels: Predicted cluster assignments
        
        Returns:
            nmi_score: float
        """
        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().detach().numpy()
        if isinstance(cluster_labels, torch.Tensor):
            cluster_labels = cluster_labels.cpu().detach().numpy()
        
        score = normalized_mutual_info_score(true_labels, cluster_labels)
        return score
    
    def cluster_purity(self, true_labels, cluster_labels):
        """
        Cluster Purity
        
        Fraction of samples in each cluster that belong to the dominant class.
        
        Purity = (1/N) * Σ max_j |cluster_i ∩ class_j|
        
        Range: [0, 1], higher is better
        
        Args:
            true_labels: Ground truth labels
            cluster_labels: Predicted cluster assignments
        
        Returns:
            purity_score: float
        """
        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().detach().numpy()
        if isinstance(cluster_labels, torch.Tensor):
            cluster_labels = cluster_labels.cpu().detach().numpy()
        
        # Filter out noise points
        mask = cluster_labels != -1
        true_labels_filtered = true_labels[mask]
        cluster_labels_filtered = cluster_labels[mask]
        
        if len(cluster_labels_filtered) == 0:
            return 0.0
        
        # Calculate purity
        total_correct = 0
        for cluster_id in np.unique(cluster_labels_filtered):
            # Get all samples in this cluster
            cluster_mask = cluster_labels_filtered == cluster_id
            cluster_true_labels = true_labels_filtered[cluster_mask]
            
            # Find most common true label in this cluster
            if len(cluster_true_labels) > 0:
                most_common = Counter(cluster_true_labels).most_common(1)[0][1]
                total_correct += most_common
        
        purity = total_correct / len(cluster_labels_filtered)
        return purity
    
    def evaluate_all(self, latent_vectors, cluster_labels, true_labels=None):
        """
        Compute all evaluation metrics
        
        Args:
            latent_vectors: Feature vectors
            cluster_labels: Predicted cluster assignments
            true_labels: Ground truth labels (optional, for supervised metrics)
        
        Returns:
            metrics_dict: Dictionary of all metrics
        """
        metrics = {}
        
        # Unsupervised metrics
        try:
            metrics['silhouette'] = self.silhouette_coefficient(latent_vectors, cluster_labels)
        except Exception as e:
            print(f"⚠️ Silhouette calculation failed: {e}")
            metrics['silhouette'] = 0.0
        
        try:
            metrics['calinski_harabasz'] = self.calinski_harabasz_index(latent_vectors, cluster_labels)
        except Exception as e:
            print(f"⚠️ Calinski-Harabasz calculation failed: {e}")
            metrics['calinski_harabasz'] = 0.0
        
        try:
            metrics['davies_bouldin'] = self.davies_bouldin_index(latent_vectors, cluster_labels)
        except Exception as e:
            print(f"⚠️ Davies-Bouldin calculation failed: {e}")
            metrics['davies_bouldin'] = np.inf
        
        # Supervised metrics (if true labels provided)
        if true_labels is not None:
            try:
                metrics['ari'] = self.adjusted_rand_index(true_labels, cluster_labels)
            except Exception as e:
                print(f"⚠️ ARI calculation failed: {e}")
                metrics['ari'] = 0.0
            
            try:
                metrics['nmi'] = self.normalized_mutual_information(true_labels, cluster_labels)
            except Exception as e:
                print(f"⚠️ NMI calculation failed: {e}")
                metrics['nmi'] = 0.0
            
            try:
                metrics['purity'] = self.cluster_purity(true_labels, cluster_labels)
            except Exception as e:
                print(f"⚠️ Purity calculation failed: {e}")
                metrics['purity'] = 0.0
        
        self.metrics = metrics
        return metrics
    
    def print_metrics(self, metrics=None):
        """
        Print metrics in a formatted table
        
        Args:
            metrics: Dictionary of metrics (uses self.metrics if None)
        """
        if metrics is None:
            metrics = self.metrics
        
        if not metrics:
            print("⚠️ No metrics to display")
            return
        
        print("\n" + "="*60)
        print("CLUSTERING EVALUATION METRICS")
        print("="*60)
        
        # Unsupervised metrics
        print("\n📊 Unsupervised Metrics:")
        print(f"  Silhouette Score:       {metrics.get('silhouette', 0.0):>8.4f}  (higher is better)")
        print(f"  Calinski-Harabasz:      {metrics.get('calinski_harabasz', 0.0):>8.2f}  (higher is better)")
        db_score = metrics.get('davies_bouldin', np.inf)
        db_str = f"{db_score:>8.4f}" if db_score != np.inf else "     inf"
        print(f"  Davies-Bouldin Index:   {db_str}  (lower is better)")
        
        # Supervised metrics (if available)
        if 'ari' in metrics:
            print("\n🎯 Supervised Metrics:")
            print(f"  Adjusted Rand Index:    {metrics.get('ari', 0.0):>8.4f}  (higher is better)")
            print(f"  Normalized Mutual Info: {metrics.get('nmi', 0.0):>8.4f}  (higher is better)")
            print(f"  Cluster Purity:         {metrics.get('purity', 0.0):>8.4f}  (higher is better)")
        
        print("="*60 + "\n")


def compare_models(results_dict):
    """
    Compare multiple models/methods side-by-side
    
    Args:
        results_dict: Dictionary of {model_name: metrics_dict}
    
    Example:
        results = {
            'PCA + K-Means': {'silhouette': 0.45, 'ari': 0.62, ...},
            'VAE + K-Means': {'silhouette': 0.58, 'ari': 0.71, ...}
        }
    """
    if not results_dict:
        print("⚠️ No results to compare")
        return
    
    # Get all metric names
    all_metrics = set()
    for metrics in results_dict.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    
    # Header
    print(f"{'Metric':<25}", end='')
    for model_name in results_dict.keys():
        print(f"{model_name[:20]:>20}", end='')
    print()
    print("-"*80)
    
    # Rows
    for metric in all_metrics:
        print(f"{metric:<25}", end='')
        for model_name, metrics in results_dict.items():
            value = metrics.get(metric, 0.0)
            if value == np.inf:
                print(f"{'inf':>20}", end='')
            elif isinstance(value, float):
                print(f"{value:>20.4f}", end='')
            else:
                print(f"{value:>20}", end='')
        print()
    
    print("="*80 + "\n")
