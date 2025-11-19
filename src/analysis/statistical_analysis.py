"""
Statistical analysis utilities for entropy experiments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass


@dataclass
class TTestResult:
    """Results from a t-test"""
    t_statistic: float
    p_value: float
    degrees_of_freedom: int
    cohens_d: float
    mean_diff: float
    ci_lower: float
    ci_upper: float

    def __str__(self) -> str:
        return (
            f"t({self.degrees_of_freedom})={self.t_statistic:.2f}, "
            f"p={self.p_value:.4f}, d={self.cohens_d:.2f}"
        )


@dataclass
class CorrelationResult:
    """Results from a correlation analysis"""
    r: float
    p_value: float
    n: int
    method: str

    def __str__(self) -> str:
        return f"r={self.r:.3f}, p={self.p_value:.4f}, n={self.n} ({self.method})"


def paired_t_test(
    group1: List[float],
    group2: List[float],
    confidence_level: float = 0.95,
) -> TTestResult:
    """
    Perform paired samples t-test

    Args:
        group1: First group of measurements
        group2: Second group of measurements
        confidence_level: Confidence level for interval (default 0.95)

    Returns:
        TTestResult with statistics
    """
    arr1 = np.array(group1)
    arr2 = np.array(group2)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(arr1, arr2)

    # Calculate effect size (Cohen's d)
    diff = arr1 - arr2
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    # Calculate confidence interval
    mean_diff = np.mean(diff)
    se_diff = stats.sem(diff)
    df = len(diff) - 1

    ci = stats.t.interval(
        confidence_level,
        df,
        loc=mean_diff,
        scale=se_diff,
    )

    return TTestResult(
        t_statistic=t_stat,
        p_value=p_value,
        degrees_of_freedom=df,
        cohens_d=cohens_d,
        mean_diff=mean_diff,
        ci_lower=ci[0],
        ci_upper=ci[1],
    )


def independent_t_test(
    group1: List[float],
    group2: List[float],
    equal_var: bool = True,
    confidence_level: float = 0.95,
) -> TTestResult:
    """
    Perform independent samples t-test

    Args:
        group1: First group of measurements
        group2: Second group of measurements
        equal_var: Assume equal variances (default True)
        confidence_level: Confidence level for interval

    Returns:
        TTestResult with statistics
    """
    arr1 = np.array(group1)
    arr2 = np.array(group2)

    # Independent t-test
    t_stat, p_value = stats.ttest_ind(arr1, arr2, equal_var=equal_var)

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(arr1) - 1) * np.var(arr1, ddof=1) +
         (len(arr2) - 1) * np.var(arr2, ddof=1)) /
        (len(arr1) + len(arr2) - 2)
    )
    cohens_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std

    # Calculate confidence interval
    mean_diff = np.mean(arr1) - np.mean(arr2)
    se_diff = pooled_std * np.sqrt(1/len(arr1) + 1/len(arr2))
    df = len(arr1) + len(arr2) - 2

    ci = stats.t.interval(
        confidence_level,
        df,
        loc=mean_diff,
        scale=se_diff,
    )

    return TTestResult(
        t_statistic=t_stat,
        p_value=p_value,
        degrees_of_freedom=df,
        cohens_d=cohens_d,
        mean_diff=mean_diff,
        ci_lower=ci[0],
        ci_upper=ci[1],
    )


def pearson_correlation(
    x: List[float],
    y: List[float],
) -> CorrelationResult:
    """
    Calculate Pearson correlation coefficient

    Args:
        x: First variable
        y: Second variable

    Returns:
        CorrelationResult with statistics
    """
    arr_x = np.array(x)
    arr_y = np.array(y)

    r, p_value = stats.pearsonr(arr_x, arr_y)

    return CorrelationResult(
        r=r,
        p_value=p_value,
        n=len(x),
        method="Pearson",
    )


def spearman_correlation(
    x: List[float],
    y: List[float],
) -> CorrelationResult:
    """
    Calculate Spearman rank correlation coefficient

    Args:
        x: First variable
        y: Second variable

    Returns:
        CorrelationResult with statistics
    """
    arr_x = np.array(x)
    arr_y = np.array(y)

    r, p_value = stats.spearmanr(arr_x, arr_y)

    return CorrelationResult(
        r=r,
        p_value=p_value,
        n=len(x),
        method="Spearman",
    )


def descriptive_statistics(
    data: List[float],
) -> Dict[str, float]:
    """
    Calculate descriptive statistics

    Args:
        data: List of values

    Returns:
        Dictionary with statistics
    """
    arr = np.array(data)

    return {
        'mean': np.mean(arr),
        'median': np.median(arr),
        'std': np.std(arr, ddof=1),
        'var': np.var(arr, ddof=1),
        'min': np.min(arr),
        'max': np.max(arr),
        'q25': np.percentile(arr, 25),
        'q75': np.percentile(arr, 75),
        'n': len(arr),
        'sem': stats.sem(arr),
    }


def compare_groups(
    data: Dict[str, List[float]],
    test_type: str = "paired",
) -> pd.DataFrame:
    """
    Compare multiple groups and return results table

    Args:
        data: Dictionary mapping group names to values
        test_type: Type of test ('paired' or 'independent')

    Returns:
        DataFrame with comparison results
    """
    results = []
    group_names = list(data.keys())

    for i, group1_name in enumerate(group_names):
        for group2_name in group_names[i+1:]:
            if test_type == "paired":
                result = paired_t_test(
                    data[group1_name],
                    data[group2_name],
                )
            else:
                result = independent_t_test(
                    data[group1_name],
                    data[group2_name],
                )

            results.append({
                'group1': group1_name,
                'group2': group2_name,
                't_statistic': result.t_statistic,
                'p_value': result.p_value,
                'cohens_d': result.cohens_d,
                'mean_diff': result.mean_diff,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper,
                'significant': result.p_value < 0.05,
            })

    return pd.DataFrame(results)


def anova_one_way(
    groups: Dict[str, List[float]],
) -> Tuple[float, float]:
    """
    Perform one-way ANOVA

    Args:
        groups: Dictionary mapping group names to values

    Returns:
        Tuple of (F-statistic, p-value)
    """
    group_arrays = [np.array(values) for values in groups.values()]
    f_stat, p_value = stats.f_oneway(*group_arrays)

    return f_stat, p_value


def bootstrap_confidence_interval(
    data: List[float],
    statistic: callable = np.mean,
    n_iterations: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval

    Args:
        data: Data to bootstrap
        statistic: Function to calculate statistic (default: mean)
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (lower bound, upper bound)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    arr = np.array(data)
    bootstrap_stats = []

    for _ in range(n_iterations):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        bootstrap_stats.append(statistic(sample))

    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return lower, upper


def effect_size_interpretation(cohens_d: float) -> str:
    """
    Interpret Cohen's d effect size

    Args:
        cohens_d: Cohen's d value

    Returns:
        Interpretation string
    """
    abs_d = abs(cohens_d)

    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
