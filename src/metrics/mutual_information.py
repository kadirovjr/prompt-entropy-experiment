"""
Mutual information estimation between prompts and tasks
"""

import numpy as np
from typing import Dict
import re


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def estimate_mi_semantic(prompt_embedding: np.ndarray, 
                         task_embedding: np.ndarray) -> float:
    """
    Estimate MI via semantic overlap
    
    Args:
        prompt_embedding: Embedding vector for prompt
        task_embedding: Embedding vector for task
        
    Returns:
        Estimated mutual information (bits)
    """
    similarity = cosine_similarity(prompt_embedding, task_embedding)
    return similarity * 10  # Scale to reasonable bit range


def estimate_mi_content(prompt: str) -> float:
    """
    Estimate MI via information content indicators
    
    Args:
        prompt: Prompt text
        
    Returns:
        Estimated mutual information (bits)
    """
    weights = {
        'numbers': 0.5,
        'constraints': 1.0,
        'examples': 2.0,
        'format_specs': 1.5,
        'technical_terms': 0.8,
    }
    
    counts = {
        'numbers': len(re.findall(r'\d+', prompt)),
        'constraints': len(re.findall(r'\b(must|should|required|constraint|limit)\b', 
                                     prompt.lower())),
        'examples': len(re.findall(r'\b(example|instance|such as|e\.g\.)\b', 
                                   prompt.lower())),
        'format_specs': len(re.findall(r'\b(format|structure|output|return)\b', 
                                       prompt.lower())),
        'technical_terms': len(re.findall(r'\b[A-Z]{2,}|\b\w+\(\)\b', prompt)),
    }
    
    score = sum(weights[k] * counts[k] for k in weights)
    return np.log2(1 + score)


def estimate_mi_coverage(prompt: str, task: str) -> float:
    """
    Estimate MI via task concept coverage
    
    Args:
        prompt: Prompt text
        task: Task description
        
    Returns:
        Estimated mutual information (bits)
    """
    # Extract key terms from task (simple approach)
    task_terms = set(re.findall(r'\b\w{4,}\b', task.lower()))
    prompt_terms = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
    
    if not task_terms:
        return 0.0
    
    coverage = len(task_terms & prompt_terms) / len(task_terms)
    return -np.log2(1 - coverage + 0.01)  # Avoid log(0)


def estimate_mutual_information(prompt: str, 
                                task: str,
                                prompt_embedding: np.ndarray = None,
                                task_embedding: np.ndarray = None) -> Dict[str, float]:
    """
    Estimate mutual information using multiple methods
    
    Args:
        prompt: Prompt text
        task: Task description
        prompt_embedding: Optional prompt embedding
        task_embedding: Optional task embedding
        
    Returns:
        Dictionary with MI estimates
    """
    results = {
        'mi_content': estimate_mi_content(prompt),
        'mi_coverage': estimate_mi_coverage(prompt, task),
    }
    
    if prompt_embedding is not None and task_embedding is not None:
        results['mi_semantic'] = estimate_mi_semantic(prompt_embedding, task_embedding)
        
        # Combined estimate
        results['mi_combined'] = (
            0.4 * results['mi_semantic'] +
            0.3 * results['mi_content'] +
            0.3 * results['mi_coverage']
        )
    else:
        # Combined without semantic
        results['mi_combined'] = (
            0.5 * results['mi_content'] +
            0.5 * results['mi_coverage']
        )
    
    return results
