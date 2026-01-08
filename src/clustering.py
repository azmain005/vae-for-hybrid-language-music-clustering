"""
Clustering Manager - K-Means, Agglomerative, DBSCAN, Spectral
Includes dimensionality reduction (t-SNE, UMAP) for visualization
"""

import numpy as np
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ UMAP not installed. Install with: pip install umap-learn")


class ClusterManager:
    """
    Manages clustering algorithms and dimensionality reduction
    """
    
    def __init__(self, n_clusters=4, random_state=42):
        """
        Args:
            n_clusters: Number of clusters (default 4 for Q1-Q4)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_labels = None
        self.algorithm_name = None
    
    def kmeans_clustering(self, latent_vectors, n_clusters=None):
        """
        K-Means Clustering (Easy Task)
        
        Args:
            latent_vectors: numpy array or torch tensor of shape (n_samples, latent_dim)
            n_clusters: Number of clusters (uses self.n_clusters if None)
        
        Returns:
            cluster_labels: numpy array of cluster assignments
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        # Convert to numpy if tensor
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.cpu().detach().numpy()
        
        print(f"\n🔵 Running K-Means (k={n_clusters})...")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.cluster_labels = kmeans.fit_predict(latent_vectors)
        self.algorithm_name = f"K-Means (k={n_clusters})"
        
        print(f"✅ K-Means completed. Cluster distribution:")
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"   Cluster {cluster_id}: {count} samples")
        
        return self.cluster_labels
    
    def agglomerative_clustering(self, latent_vectors, n_clusters=None, linkage='ward'):
        """
        Agglomerative Hierarchical Clustering (Medium Task)
        
        Args:
            latent_vectors: numpy array or torch tensor
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        
        Returns:
            cluster_labels: numpy array of cluster assignments
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.cpu().detach().numpy()
        
        print(f"\n🟣 Running Agglomerative Clustering (k={n_clusters}, linkage={linkage})...")
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        
        self.cluster_labels = agg.fit_predict(latent_vectors)
        self.algorithm_name = f"Agglomerative (k={n_clusters}, {linkage})"
        
        print(f"✅ Agglomerative completed. Cluster distribution:")
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"   Cluster {cluster_id}: {count} samples")
        
        return self.cluster_labels
    
    def dbscan_clustering(self, latent_vectors, eps=0.5, min_samples=5):
        """
        DBSCAN Density-Based Clustering (Medium Task)
        
        Args:
            latent_vectors: numpy array or torch tensor
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
        
        Returns:
            cluster_labels: numpy array of cluster assignments (-1 for noise)
        """
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.cpu().detach().numpy()
        
        print(f"\n🟢 Running DBSCAN (eps={eps}, min_samples={min_samples})...")
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        self.cluster_labels = dbscan.fit_predict(latent_vectors)
        self.algorithm_name = f"DBSCAN (eps={eps})"
        
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        print(f"✅ DBSCAN completed.")
        print(f"   Found {n_clusters} clusters")
        print(f"   Noise points: {n_noise}")
        
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            if cluster_id == -1:
                print(f"   Noise: {count} samples")
            else:
                print(f"   Cluster {cluster_id}: {count} samples")
        
        return self.cluster_labels
    
    def spectral_clustering(self, latent_vectors, n_clusters=None, affinity='rbf'):
        """
        Spectral Clustering (Hard Task)
        
        Args:
            latent_vectors: numpy array or torch tensor
            n_clusters: Number of clusters
            affinity: Affinity metric ('rbf', 'nearest_neighbors')
        
        Returns:
            cluster_labels: numpy array of cluster assignments
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.cpu().detach().numpy()
        
        print(f"\n🟡 Running Spectral Clustering (k={n_clusters}, affinity={affinity})...")
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            random_state=self.random_state,
            assign_labels='kmeans'
        )
        
        self.cluster_labels = spectral.fit_predict(latent_vectors)
        self.algorithm_name = f"Spectral (k={n_clusters}, {affinity})"
        
        print(f"✅ Spectral Clustering completed. Cluster distribution:")
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"   Cluster {cluster_id}: {count} samples")
        
        return self.cluster_labels
    
    def pca_reduction(self, latent_vectors, n_components=2):
        """
        PCA dimensionality reduction
        
        Args:
            latent_vectors: High-dimensional vectors
            n_components: Target dimensionality
        
        Returns:
            reduced_vectors: numpy array of shape (n_samples, n_components)
        """
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.cpu().detach().numpy()
        
        print(f"\n📊 Running PCA (n_components={n_components})...")
        pca = PCA(n_components=n_components, random_state=self.random_state)
        reduced = pca.fit_transform(latent_vectors)
        
        explained_var = pca.explained_variance_ratio_
        print(f"✅ PCA completed. Explained variance: {explained_var.sum():.2%}")
        
        return reduced
    
    def tsne_reduction(self, latent_vectors, n_components=2, perplexity=30):
        """
        t-SNE dimensionality reduction for visualization
        
        Args:
            latent_vectors: High-dimensional vectors
            n_components: Target dimensionality (typically 2 for visualization)
            perplexity: t-SNE perplexity parameter
        
        Returns:
            reduced_vectors: numpy array of shape (n_samples, n_components)
        """
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.cpu().detach().numpy()
        
        print(f"\n📊 Running t-SNE (n_components={n_components}, perplexity={perplexity})...")
        
        # Adjust perplexity if dataset is small
        n_samples = latent_vectors.shape[0]
        if n_samples < perplexity * 3:
            perplexity = max(5, n_samples // 3)
            print(f"   ⚠️ Adjusted perplexity to {perplexity} (small dataset)")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=self.random_state,
            max_iter=1000
        )
        
        reduced = tsne.fit_transform(latent_vectors)
        print(f"✅ t-SNE completed.")
        
        return reduced
    
    def umap_reduction(self, latent_vectors, n_components=2, n_neighbors=15):
        """
        UMAP dimensionality reduction for visualization
        
        Args:
            latent_vectors: High-dimensional vectors
            n_components: Target dimensionality (typically 2 for visualization)
            n_neighbors: UMAP n_neighbors parameter
        
        Returns:
            reduced_vectors: numpy array of shape (n_samples, n_components)
        """
        if not UMAP_AVAILABLE:
            print("⚠️ UMAP not available. Using t-SNE instead.")
            return self.tsne_reduction(latent_vectors, n_components)
        
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.cpu().detach().numpy()
        
        print(f"\n📊 Running UMAP (n_components={n_components}, n_neighbors={n_neighbors})...")
        
        # Adjust n_neighbors if dataset is small
        n_samples = latent_vectors.shape[0]
        if n_samples < n_neighbors + 1:
            n_neighbors = max(2, n_samples // 2)
            print(f"   ⚠️ Adjusted n_neighbors to {n_neighbors} (small dataset)")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=self.random_state
        )
        
        reduced = reducer.fit_transform(latent_vectors)
        print(f"✅ UMAP completed.")
        
        return reduced
    
    def get_cluster_labels(self):
        """Return the most recent cluster labels"""
        return self.cluster_labels
    
    def get_algorithm_name(self):
        """Return the name of the most recent algorithm"""
        return self.algorithm_name


def multimodal_fusion(audio_latent, lyrics_latent, method='concat'):
    """
    Fuse multi-modal latent representations (Hard Task)
    
    Args:
        audio_latent: Audio latent vectors (n_samples, latent_dim)
        lyrics_latent: Lyrics latent vectors (n_samples, latent_dim)
        method: Fusion method ('concat', 'sum', 'product', 'mean')
    
    Returns:
        fused_latent: Fused latent vectors
    """
    if isinstance(audio_latent, torch.Tensor):
        audio_latent = audio_latent.cpu().detach().numpy()
    if isinstance(lyrics_latent, torch.Tensor):
        lyrics_latent = lyrics_latent.cpu().detach().numpy()
    
    if method == 'concat':
        # Concatenate along feature dimension
        fused = np.concatenate([audio_latent, lyrics_latent], axis=1)
    elif method == 'sum':
        # Element-wise sum
        fused = audio_latent + lyrics_latent
    elif method == 'product':
        # Element-wise product
        fused = audio_latent * lyrics_latent
    elif method == 'mean':
        # Element-wise mean
        fused = (audio_latent + lyrics_latent) / 2
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    
    print(f"🔗 Multi-modal fusion ({method}): {audio_latent.shape} + {lyrics_latent.shape} -> {fused.shape}")
    
    return fused
