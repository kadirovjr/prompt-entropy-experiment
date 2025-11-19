"""
Statistical analysis utilities for entropy experiments
"""

from .statistical_analysis import (
    TTestResult,
    CorrelationResult,
    paired_t_test,
    independent_t_test,
    pearson_correlation,
    spearman_correlation,
    descriptive_statistics,
    compare_groups,
    anova_one_way,
    bootstrap_confidence_interval,
    effect_size_interpretation,
)

__all__ = [
    'TTestResult',
    'CorrelationResult',
    'paired_t_test',
    'independent_t_test',
    'pearson_correlation',
    'spearman_correlation',
    'descriptive_statistics',
    'compare_groups',
    'anova_one_way',
    'bootstrap_confidence_interval',
    'effect_size_interpretation',
]
