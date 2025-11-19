"""
Unit tests for statistical analysis
"""

import pytest
import numpy as np
from src.analysis.statistical_analysis import (
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


class TestPairedTTest:
    def test_identical_groups(self):
        """Test with identical groups"""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = paired_t_test(group1, group2)

        assert abs(result.t_statistic) < 1e-6
        assert result.p_value > 0.05
        assert abs(result.cohens_d) < 1e-6
        assert abs(result.mean_diff) < 1e-6

    def test_different_groups(self):
        """Test with different groups"""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [2.0, 3.0, 4.0, 5.0, 6.0]

        result = paired_t_test(group1, group2)

        assert result.t_statistic != 0
        assert result.mean_diff == -1.0
        assert result.ci_lower < result.ci_upper

    def test_large_effect(self):
        """Test with large effect size"""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [10.0, 11.0, 12.0, 13.0, 14.0]

        result = paired_t_test(group1, group2)

        assert abs(result.cohens_d) > 1.0
        assert result.p_value < 0.05


class TestIndependentTTest:
    def test_identical_groups(self):
        """Test with identical groups"""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = independent_t_test(group1, group2)

        assert abs(result.t_statistic) < 1e-6
        assert result.p_value > 0.05
        assert abs(result.cohens_d) < 1e-6

    def test_different_groups(self):
        """Test with different groups"""
        group1 = [1.0, 2.0, 3.0]
        group2 = [4.0, 5.0, 6.0]

        result = independent_t_test(group1, group2)

        assert result.t_statistic < 0  # group1 < group2
        assert result.p_value < 0.05


class TestCorrelation:
    def test_perfect_positive_correlation(self):
        """Test perfect positive correlation"""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        result = pearson_correlation(x, y)

        assert abs(result.r - 1.0) < 1e-6
        assert result.p_value < 0.05

    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation"""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]

        result = pearson_correlation(x, y)

        assert abs(result.r - (-1.0)) < 1e-6
        assert result.p_value < 0.05

    def test_no_correlation(self):
        """Test no correlation"""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 3.0, 3.0, 3.0, 3.0]

        result = pearson_correlation(x, y)

        # Should have NaN or undefined correlation
        assert np.isnan(result.r) or abs(result.r) < 0.1

    def test_spearman_vs_pearson(self):
        """Test Spearman vs Pearson for monotonic but nonlinear data"""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 4.0, 9.0, 16.0, 25.0]  # Quadratic

        pearson_r = pearson_correlation(x, y)
        spearman_r = spearman_correlation(x, y)

        # Spearman should be 1.0 (perfect monotonic)
        assert abs(spearman_r.r - 1.0) < 1e-6
        # Pearson should be less than 1.0 (not perfectly linear)
        assert pearson_r.r < 1.0


class TestDescriptiveStatistics:
    def test_basic_stats(self):
        """Test basic descriptive statistics"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        stats = descriptive_statistics(data)

        assert stats['mean'] == 3.0
        assert stats['median'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['n'] == 5

    def test_with_variance(self):
        """Test variance and std"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        stats = descriptive_statistics(data)

        assert stats['std'] > 0
        assert stats['var'] > 0
        assert abs(stats['var'] - stats['std']**2) < 1e-6

    def test_single_value(self):
        """Test with single value"""
        data = [5.0]

        stats = descriptive_statistics(data)

        assert stats['mean'] == 5.0
        assert stats['std'] == 0.0


class TestCompareGroups:
    def test_multiple_groups(self):
        """Test comparing multiple groups"""
        data = {
            'group1': [1.0, 2.0, 3.0],
            'group2': [2.0, 3.0, 4.0],
            'group3': [3.0, 4.0, 5.0],
        }

        results = compare_groups(data, test_type='independent')

        # Should have 3 comparisons (3 choose 2)
        assert len(results) == 3

        # Check columns
        assert 'group1' in results.columns
        assert 't_statistic' in results.columns
        assert 'p_value' in results.columns
        assert 'cohens_d' in results.columns


class TestANOVA:
    def test_three_groups(self):
        """Test one-way ANOVA with three groups"""
        groups = {
            'group1': [1.0, 2.0, 3.0],
            'group2': [4.0, 5.0, 6.0],
            'group3': [7.0, 8.0, 9.0],
        }

        f_stat, p_value = anova_one_way(groups)

        assert f_stat > 0
        assert p_value < 0.05  # Groups are clearly different

    def test_identical_groups(self):
        """Test ANOVA with identical groups"""
        groups = {
            'group1': [3.0, 3.0, 3.0],
            'group2': [3.0, 3.0, 3.0],
            'group3': [3.0, 3.0, 3.0],
        }

        f_stat, p_value = anova_one_way(groups)

        assert f_stat < 1e-6
        assert p_value > 0.05


class TestBootstrap:
    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0] * 10

        lower, upper = bootstrap_confidence_interval(
            data,
            statistic=np.mean,
            n_iterations=1000,
            random_seed=42,
        )

        # Mean should be around 3.0
        assert lower < 3.0 < upper
        assert lower < upper

    def test_bootstrap_with_different_statistic(self):
        """Test bootstrap with different statistic"""
        data = list(range(1, 101))

        lower, upper = bootstrap_confidence_interval(
            data,
            statistic=np.median,
            n_iterations=1000,
            random_seed=42,
        )

        # Median should be around 50.5
        assert lower < 50.5 < upper


class TestEffectSizeInterpretation:
    def test_negligible(self):
        """Test negligible effect size"""
        assert effect_size_interpretation(0.1) == "negligible"

    def test_small(self):
        """Test small effect size"""
        assert effect_size_interpretation(0.3) == "small"

    def test_medium(self):
        """Test medium effect size"""
        assert effect_size_interpretation(0.6) == "medium"

    def test_large(self):
        """Test large effect size"""
        assert effect_size_interpretation(1.0) == "large"

    def test_negative_values(self):
        """Test negative effect sizes"""
        assert effect_size_interpretation(-0.6) == "medium"
        assert effect_size_interpretation(-1.2) == "large"
