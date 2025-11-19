"""
Integration tests for full experimental workflow
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from src.metrics.entropy import calculate_all_entropies
from src.metrics.mutual_information import estimate_mutual_information
from src.metrics.quality import calculate_overall_quality
from src.analysis.statistical_analysis import (
    paired_t_test,
    descriptive_statistics,
    pearson_correlation,
)
from src.utils.io_utils import save_json, load_json, save_dataframe, load_dataframe
from src.utils.data_utils import batch_data, aggregate_by_key
import pandas as pd


class TestEntropyWorkflow:
    """Test complete entropy calculation workflow"""

    def test_entropy_analysis_pipeline(self):
        """Test full entropy analysis pipeline"""
        # Mock responses from two prompt types
        spec_responses = [
            "def add(a, b): return a + b",
            "def add(x, y): return x + y",
            "def add(num1, num2): return num1 + num2",
        ]

        vague_responses = [
            "Here's a function to add numbers",
            "You can add them like this",
            "Adding is straightforward",
        ]

        # Calculate entropies
        spec_entropy = calculate_all_entropies(spec_responses)
        vague_entropy = calculate_all_entropies(vague_responses)

        # Both should have positive entropy
        assert spec_entropy['token_entropy'] > 0
        assert vague_entropy['token_entropy'] > 0

        # Token entropy should be different
        assert spec_entropy['token_entropy'] != vague_entropy['token_entropy']

    def test_entropy_with_embeddings(self):
        """Test entropy calculation with embeddings"""
        responses = ["hello world", "goodbye world", "hello universe"]

        # Mock embeddings
        embeddings = np.random.randn(3, 10)

        results = calculate_all_entropies(responses, embeddings)

        assert 'token_entropy' in results
        assert 'semantic_entropy' in results
        assert 'structural_entropy' in results

        # All should be non-negative
        assert all(v >= 0 for v in results.values())


class TestMutualInformationWorkflow:
    """Test MI estimation workflow"""

    def test_mi_analysis_pipeline(self):
        """Test full MI estimation pipeline"""
        # Specification-driven prompt
        spec_prompt = (
            "Write a function named 'add' that takes two parameters "
            "and returns their sum. Must handle integers."
        )

        # Vague prompt
        vague_prompt = "Write a function"

        task = "Create an addition function"

        # Estimate MI
        spec_mi = estimate_mutual_information(spec_prompt, task)
        vague_mi = estimate_mutual_information(vague_prompt, task)

        # Spec prompt should have higher MI
        assert spec_mi['mi_content'] > vague_mi['mi_content']

    def test_mi_with_embeddings(self):
        """Test MI with embeddings"""
        prompt = "Write a parser for JSON"
        task = "Create a JSON parser"

        # Mock embeddings
        prompt_emb = np.random.randn(10)
        task_emb = np.random.randn(10)

        results = estimate_mutual_information(
            prompt, task, prompt_emb, task_emb
        )

        assert 'mi_semantic' in results
        assert 'mi_content' in results
        assert 'mi_coverage' in results
        assert 'mi_combined' in results


class TestQualityWorkflow:
    """Test quality assessment workflow"""

    def test_quality_assessment_pipeline(self):
        """Test full quality assessment"""
        response = "def add(a, b):\n    return a + b"
        task = "Write an addition function"

        # Mock embeddings
        response_emb = np.random.randn(10)
        task_emb = np.random.randn(10)

        scores = calculate_overall_quality(
            response,
            task,
            response_emb,
            task_emb,
            expected_elements=['def', 'add', 'return'],
            required_components=['function'],
            format_requirements={'has_code': True},
        )

        assert 'overall' in scores
        assert 'correctness' in scores
        assert 'completeness' in scores
        assert 'relevance' in scores

        # Scores should be in [0, 1]
        assert all(0 <= v <= 1 for v in scores.values())


class TestStatisticalWorkflow:
    """Test statistical analysis workflow"""

    def test_hypothesis_testing_pipeline(self):
        """Test hypothesis testing workflow"""
        # Simulate entropy values for two conditions
        spec_entropies = [3.5, 3.6, 3.4, 3.7, 3.5]
        vague_entropies = [4.2, 4.3, 4.1, 4.4, 4.2]

        # Perform t-test
        result = paired_t_test(spec_entropies, vague_entropies)

        assert result.t_statistic < 0  # spec < vague
        assert result.p_value < 0.05  # Significant difference
        assert abs(result.cohens_d) > 0.5  # Medium to large effect

    def test_correlation_analysis(self):
        """Test correlation analysis workflow"""
        # MI and entropy should be negatively correlated
        mi_values = [5.0, 6.0, 7.0, 8.0, 9.0]
        entropy_values = [4.0, 3.5, 3.0, 2.5, 2.0]

        result = pearson_correlation(mi_values, entropy_values)

        assert result.r < 0  # Negative correlation
        assert result.p_value < 0.05


class TestDataManagementWorkflow:
    """Test data management and I/O workflow"""

    def test_save_and_load_results(self):
        """Test saving and loading experimental results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create experimental results
            results = {
                'experiment_id': 'test_001',
                'prompt_type': 'specification',
                'entropy': 3.5,
                'mi': 7.2,
                'quality': 0.85,
            }

            # Save results
            filepath = os.path.join(tmpdir, 'results.json')
            save_json(results, filepath)

            # Load results
            loaded = load_json(filepath)

            assert loaded == results

    def test_batch_processing(self):
        """Test batch processing of responses"""
        responses = [f"response_{i}" for i in range(100)]

        # Process in batches
        batches = batch_data(responses, batch_size=10)

        assert len(batches) == 10
        assert all(len(batch) == 10 for batch in batches)

    def test_aggregate_results(self):
        """Test aggregating experimental results"""
        results = [
            {'prompt_type': 'spec', 'entropy': 3.5},
            {'prompt_type': 'spec', 'entropy': 3.6},
            {'prompt_type': 'vague', 'entropy': 4.2},
            {'prompt_type': 'vague', 'entropy': 4.3},
        ]

        aggregated = aggregate_by_key(
            results,
            key='prompt_type',
            value_key='entropy',
            agg_func='mean',
        )

        assert 'spec' in aggregated
        assert 'vague' in aggregated
        assert aggregated['vague'] > aggregated['spec']


class TestEndToEndExperiment:
    """Test end-to-end experimental workflow"""

    def test_complete_experiment(self):
        """Test complete experimental workflow"""
        # 1. Define experimental conditions
        conditions = {
            'specification': "Write a function named 'add' with parameters a, b",
            'vague': "Write a function",
        }

        task = "Create addition function"

        # 2. Mock generating responses (in real case, use sample_responses)
        responses = {
            'specification': [
                "def add(a, b): return a + b",
                "def add(a, b): return a + b",
            ],
            'vague': [
                "Here's how to add",
                "You can do addition",
            ],
        }

        # 3. Calculate metrics for each condition
        metrics = {}
        for condition, response_list in responses.items():
            entropy = calculate_all_entropies(response_list)
            mi = estimate_mutual_information(conditions[condition], task)

            metrics[condition] = {
                'entropy': entropy['token_entropy'],
                'mi': mi['mi_combined'],
            }

        # 4. Statistical comparison
        spec_entropy = metrics['specification']['entropy']
        vague_entropy = metrics['vague']['entropy']

        # Spec should have lower entropy
        assert spec_entropy < vague_entropy

        # Spec should have higher MI
        assert metrics['specification']['mi'] > metrics['vague']['mi']

    def test_dataframe_workflow(self):
        """Test workflow with pandas DataFrames"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create results DataFrame
            results = pd.DataFrame({
                'condition': ['spec', 'spec', 'vague', 'vague'],
                'entropy': [3.5, 3.6, 4.2, 4.3],
                'mi': [7.2, 7.1, 5.5, 5.4],
                'quality': [0.85, 0.87, 0.72, 0.70],
            })

            # Save DataFrame
            filepath = os.path.join(tmpdir, 'results.csv')
            save_dataframe(results, filepath, format='csv')

            # Load DataFrame
            loaded = load_dataframe(filepath, format='csv')

            assert len(loaded) == 4
            assert list(loaded.columns) == ['condition', 'entropy', 'mi', 'quality']

            # Compute statistics by condition
            stats = loaded.groupby('condition')['entropy'].mean()

            assert stats['vague'] > stats['spec']
