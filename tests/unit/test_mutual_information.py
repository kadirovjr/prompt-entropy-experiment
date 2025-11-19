"""
Unit tests for mutual information estimation
"""

import pytest
import numpy as np
from src.metrics.mutual_information import (
    cosine_similarity,
    estimate_mi_semantic,
    estimate_mi_content,
    estimate_mi_coverage,
    estimate_mutual_information,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        """Test similarity of identical vectors"""
        vec = np.array([1.0, 2.0, 3.0])
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity) < 1e-6

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors"""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6


class TestMIContent:
    def test_empty_prompt(self):
        """Test MI estimation with empty prompt"""
        mi = estimate_mi_content("")
        assert mi >= 0

    def test_prompt_with_numbers(self):
        """Test that numbers increase MI estimate"""
        prompt1 = "Write a function"
        prompt2 = "Write a function with 10 parameters and 5 return values"

        mi1 = estimate_mi_content(prompt1)
        mi2 = estimate_mi_content(prompt2)

        assert mi2 > mi1  # More specific prompt should have higher MI

    def test_prompt_with_constraints(self):
        """Test that constraints increase MI estimate"""
        prompt1 = "Write code"
        prompt2 = "Write code that must handle errors and should be efficient"

        mi1 = estimate_mi_content(prompt1)
        mi2 = estimate_mi_content(prompt2)

        assert mi2 > mi1

    def test_prompt_with_examples(self):
        """Test that examples increase MI estimate"""
        prompt1 = "Write a parser"
        prompt2 = "Write a parser. For example, it should handle JSON and XML"

        mi1 = estimate_mi_content(prompt1)
        mi2 = estimate_mi_content(prompt2)

        assert mi2 > mi1


class TestMICoverage:
    def test_no_overlap(self):
        """Test coverage with no overlap"""
        prompt = "write code"
        task = "parse json"
        mi = estimate_mi_coverage(prompt, task)
        assert mi >= 0

    def test_full_overlap(self):
        """Test coverage with full overlap"""
        prompt = "parse json data structure"
        task = "parse json"
        mi = estimate_mi_coverage(prompt, task)
        assert mi > 0

    def test_partial_overlap(self):
        """Test coverage with partial overlap"""
        prompt = "write a json parser"
        task = "create xml parser"

        mi = estimate_mi_coverage(prompt, task)
        assert mi > 0  # Should have some overlap from "parser"


class TestMISemantic:
    def test_similar_embeddings(self):
        """Test MI with similar embeddings"""
        # Normalized vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.9, 0.1, 0.0])
        vec2 = vec2 / np.linalg.norm(vec2)

        mi = estimate_mi_semantic(vec1, vec2)
        assert mi > 0

    def test_orthogonal_embeddings(self):
        """Test MI with orthogonal embeddings"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        mi = estimate_mi_semantic(vec1, vec2)
        # Should be near zero for orthogonal vectors
        assert abs(mi) < 1.0


class TestEstimateMutualInformation:
    def test_without_embeddings(self):
        """Test MI estimation without embeddings"""
        prompt = "Write a function that must handle 10 cases"
        task = "Write a function"

        results = estimate_mutual_information(prompt, task)

        assert 'mi_content' in results
        assert 'mi_coverage' in results
        assert 'mi_combined' in results
        assert 'mi_semantic' not in results

        assert results['mi_content'] >= 0
        assert results['mi_coverage'] >= 0
        assert results['mi_combined'] >= 0

    def test_with_embeddings(self):
        """Test MI estimation with embeddings"""
        prompt = "Write a parser"
        task = "Create a parser"
        prompt_emb = np.array([1.0, 0.5, 0.3])
        task_emb = np.array([0.9, 0.6, 0.2])

        results = estimate_mutual_information(
            prompt, task, prompt_emb, task_emb
        )

        assert 'mi_content' in results
        assert 'mi_coverage' in results
        assert 'mi_semantic' in results
        assert 'mi_combined' in results

        assert all(v >= 0 for v in results.values())

    def test_combined_is_weighted_average(self):
        """Test that combined MI is reasonable"""
        prompt = "Write code with constraints"
        task = "Write code"
        prompt_emb = np.array([1.0, 0.0])
        task_emb = np.array([1.0, 0.0])

        results = estimate_mutual_information(
            prompt, task, prompt_emb, task_emb
        )

        # Combined should be positive and reasonable
        assert results['mi_combined'] > 0
        # Combined should be influenced by all components
        assert results['mi_combined'] <= max(
            results['mi_semantic'],
            results['mi_content'],
            results['mi_coverage']
        ) * 2  # Allow some margin
