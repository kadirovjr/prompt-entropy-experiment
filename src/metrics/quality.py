"""
Quality assessment metrics for LLM responses
"""

from typing import Dict, List
import numpy as np


def assess_correctness(response: str, task: str, expected_elements: List[str] = None) -> float:
    """
    Assess task-specific correctness (placeholder for task-specific logic)
    
    Args:
        response: Response text
        task: Task description
        expected_elements: Optional list of expected elements
        
    Returns:
        Correctness score [0, 1]
    """
    if expected_elements is None:
        return 0.5  # Neutral score if no expectations
    
    # Check for presence of expected elements
    found = sum(1 for elem in expected_elements if elem.lower() in response.lower())
    return found / len(expected_elements)


def assess_completeness(response: str, required_components: List[str]) -> float:
    """
    Assess presence of required components
    
    Args:
        response: Response text
        required_components: List of required component identifiers
        
    Returns:
        Completeness score [0, 1]
    """
    if not required_components:
        return 1.0
    
    found = sum(1 for comp in required_components if comp.lower() in response.lower())
    return found / len(required_components)


def assess_relevance(response_embedding: np.ndarray, 
                     task_embedding: np.ndarray) -> float:
    """
    Assess relevance via embedding similarity
    
    Args:
        response_embedding: Response embedding vector
        task_embedding: Task embedding vector
        
    Returns:
        Relevance score [0, 1]
    """
    similarity = np.dot(response_embedding, task_embedding) / (
        np.linalg.norm(response_embedding) * np.linalg.norm(task_embedding)
    )
    # Convert from [-1, 1] to [0, 1]
    return (similarity + 1) / 2


def assess_coherence(response: str) -> float:
    """
    Assess sentence-to-sentence coherence (simple heuristic)
    
    Args:
        response: Response text
        
    Returns:
        Coherence score [0, 1]
    """
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    
    if len(sentences) < 2:
        return 1.0
    
    # Simple heuristic: check for reasonable sentence lengths
    lengths = [len(s.split()) for s in sentences]
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)
    
    # Penalize extreme variation
    if avg_length < 3:
        return 0.5
    
    variation_score = 1.0 - min(std_length / avg_length, 1.0)
    return max(variation_score, 0.0)


def assess_format_compliance(response: str, format_requirements: Dict[str, any]) -> float:
    """
    Assess adherence to format requirements
    
    Args:
        response: Response text
        format_requirements: Dict of format requirements
        
    Returns:
        Format compliance score [0, 1]
    """
    if not format_requirements:
        return 1.0
    
    score = 0.0
    total = len(format_requirements)
    
    for req_type, req_value in format_requirements.items():
        if req_type == 'has_code' and req_value:
            score += 1 if '```' in response else 0
        elif req_type == 'has_list' and req_value:
            score += 1 if any(line.strip().startswith(('-', '*', '1.')) 
                            for line in response.split('\n')) else 0
        elif req_type == 'min_length' and isinstance(req_value, int):
            score += 1 if len(response.split()) >= req_value else 0
        elif req_type == 'max_length' and isinstance(req_value, int):
            score += 1 if len(response.split()) <= req_value else 0
    
    return score / total if total > 0 else 1.0


def calculate_overall_quality(response: str,
                              task: str,
                              response_embedding: np.ndarray = None,
                              task_embedding: np.ndarray = None,
                              expected_elements: List[str] = None,
                              required_components: List[str] = None,
                              format_requirements: Dict[str, any] = None) -> Dict[str, float]:
    """
    Calculate overall quality score across all dimensions
    
    Args:
        response: Response text
        task: Task description
        response_embedding: Optional response embedding
        task_embedding: Optional task embedding
        expected_elements: Optional expected elements for correctness
        required_components: Optional required components
        format_requirements: Optional format requirements
        
    Returns:
        Dictionary with quality scores
    """
    scores = {
        'correctness': assess_correctness(response, task, expected_elements),
        'completeness': assess_completeness(response, required_components or []),
        'coherence': assess_coherence(response),
        'format_compliance': assess_format_compliance(response, format_requirements or {}),
    }
    
    if response_embedding is not None and task_embedding is not None:
        scores['relevance'] = assess_relevance(response_embedding, task_embedding)
    else:
        scores['relevance'] = 0.5  # Neutral if embeddings unavailable
    
    # Weighted average
    weights = {
        'correctness': 0.35,
        'completeness': 0.25,
        'relevance': 0.20,
        'coherence': 0.10,
        'format_compliance': 0.10,
    }
    
    scores['overall'] = sum(scores[k] * weights[k] for k in weights)
    
    return scores
