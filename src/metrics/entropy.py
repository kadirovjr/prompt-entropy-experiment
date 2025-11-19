"""
Entropy calculation methods for LLM outputs
"""

import numpy as np
from typing import List, Dict
from collections import Counter
from sklearn.cluster import KMeans
import numpy.typing as npt


def calculate_token_entropy(responses: List[str]) -> float:
    """
    Calculate Shannon entropy over token distribution
    
    Args:
        responses: List of response strings
        
    Returns:
        Token entropy in bits
    """
    # Tokenize (simple whitespace splitting for now)
    all_tokens = []
    for response in responses:
        tokens = response.lower().split()
        all_tokens.extend(tokens)
    
    # Calculate probability distribution
    token_counts = Counter(all_tokens)
    total = len(all_tokens)
    
    # Calculate entropy
    entropy = 0.0
    for count in token_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def calculate_semantic_entropy(embeddings: npt.NDArray[np.float64], 
                               max_clusters: int = 10) -> float:
    """
    Calculate entropy in embedding space via clustering
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        max_clusters: Maximum number of clusters
        
    Returns:
        Semantic entropy in bits
    """
    n_samples = len(embeddings)
    k = min(n_samples // 2, max_clusters)
    
    if k < 2:
        return 0.0
    
    # Cluster embeddings
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Calculate cluster distribution
    cluster_counts = Counter(labels)
    
    # Calculate entropy
    entropy = 0.0
    for count in cluster_counts.values():
        p = count / n_samples
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def calculate_structural_entropy(responses: List[str]) -> float:
    """
    Calculate entropy over structural features
    
    Args:
        responses: List of response strings
        
    Returns:
        Structural entropy in bits
    """
    features = []
    
    for response in responses:
        # Extract structural features
        feature_dict = {
            'length_bin': len(response) // 100,  # Binned length
            'num_paragraphs': response.count('\n\n') + 1,
            'has_code': '```' in response,
            'has_list': any(line.strip().startswith(('-', '*', '1.', '2.')) 
                           for line in response.split('\n')),
        }
        
        # Create feature tuple for counting
        feature_tuple = tuple(sorted(feature_dict.items()))
        features.append(feature_tuple)
    
    # Calculate distribution
    feature_counts = Counter(features)
    total = len(features)
    
    # Calculate entropy
    entropy = 0.0
    for count in feature_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def calculate_all_entropies(responses: List[str], 
                            embeddings: npt.NDArray[np.float64] = None) -> Dict[str, float]:
    """
    Calculate all entropy metrics
    
    Args:
        responses: List of response strings
        embeddings: Optional embeddings array
        
    Returns:
        Dictionary with all entropy metrics
    """
    results = {
        'token_entropy': calculate_token_entropy(responses),
        'structural_entropy': calculate_structural_entropy(responses),
    }
    
    if embeddings is not None:
        results['semantic_entropy'] = calculate_semantic_entropy(embeddings)
    
    return results
