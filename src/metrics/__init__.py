"""
Metrics module for entropy, mutual information, and quality assessment
"""

from .entropy import (
    calculate_token_entropy,
    calculate_semantic_entropy,
    calculate_structural_entropy,
    calculate_all_entropies,
)

from .mutual_information import (
    estimate_mi_semantic,
    estimate_mi_content,
    estimate_mi_coverage,
    estimate_mutual_information,
)

from .quality import (
    assess_correctness,
    assess_completeness,
    assess_relevance,
    assess_coherence,
    assess_format_compliance,
    calculate_overall_quality,
)

__all__ = [
    # Entropy
    'calculate_token_entropy',
    'calculate_semantic_entropy',
    'calculate_structural_entropy',
    'calculate_all_entropies',
    # Mutual Information
    'estimate_mi_semantic',
    'estimate_mi_content',
    'estimate_mi_coverage',
    'estimate_mutual_information',
    # Quality
    'assess_correctness',
    'assess_completeness',
    'assess_relevance',
    'assess_coherence',
    'assess_format_compliance',
    'calculate_overall_quality',
]
