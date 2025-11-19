"""
Unit tests for entropy calculations
"""

import pytest
import numpy as np
from src.metrics.entropy import (
    calculate_token_entropy,
    calculate_semantic_entropy,
    calculate_structural_entropy,
    calculate_all_entropies,
)


class TestTokenEntropy:
    def test_identical_responses(self):
        """Test that identical responses have low entropy"""
        responses = ["hello world"] * 10
        entropy = calculate_token_entropy(responses)
        # With identical responses, entropy should be low (each word appears equally)
        assert entropy > 0  # Not zero because there are 2 different words
        assert entropy < 2  # But should be low

    def test_diverse_responses(self):
        """Test that diverse responses have higher entropy"""
        responses = [
            "hello world",
            "goodbye universe",
            "programming language",
            "data science",
            "machine learning",
        ]
        entropy = calculate_token_entropy(responses)
        assert entropy > 0

    def test_empty_list(self):
        """Test handling of empty list"""
        with pytest.raises((ValueError, ZeroDivisionError)):
            calculate_token_entropy([])

    def test_single_response(self):
        """Test single response"""
        responses = ["hello world"]
        entropy = calculate_token_entropy(responses)
        assert entropy >= 0


class TestSemanticEntropy:
    def test_simple_clustering(self):
        """Test semantic entropy with simple embeddings"""
        # Create simple embeddings that should form distinct clusters
        embeddings = np.array([
            [1.0, 0.0],
            [1.1, 0.1],  # Similar to first
            [0.0, 1.0],
            [0.1, 1.1],  # Similar to third
        ])
        entropy = calculate_semantic_entropy(embeddings, max_clusters=2)
        assert entropy >= 0

    def test_identical_embeddings(self):
        """Test with identical embeddings"""
        embeddings = np.array([[1.0, 0.0]] * 5)
        entropy = calculate_semantic_entropy(embeddings, max_clusters=3)
        # Should have low entropy since all embeddings are identical
        assert entropy >= 0
        assert entropy < 1

    def test_single_embedding(self):
        """Test with single embedding"""
        embeddings = np.array([[1.0, 0.0]])
        entropy = calculate_semantic_entropy(embeddings)
        assert entropy == 0.0


class TestStructuralEntropy:
    def test_varied_structures(self):
        """Test with varied structural features"""
        responses = [
            "Short response.",
            "A much longer response with multiple sentences. It has more content.",
            "```python\ncode_block()\n```",
            "- Item 1\n- Item 2\n- Item 3",
        ]
        entropy = calculate_structural_entropy(responses)
        assert entropy > 0

    def test_identical_structures(self):
        """Test with identical structures"""
        responses = ["Same length text." for _ in range(5)]
        entropy = calculate_structural_entropy(responses)
        # Should be 0 or very low since structures are identical
        assert entropy >= 0
        assert entropy < 0.1

    def test_empty_list(self):
        """Test with empty list"""
        with pytest.raises((ValueError, ZeroDivisionError)):
            calculate_structural_entropy([])


class TestCalculateAllEntropies:
    def test_without_embeddings(self):
        """Test calculating all entropies without embeddings"""
        responses = [
            "hello world",
            "goodbye world",
            "hello universe",
        ]
        results = calculate_all_entropies(responses)

        assert 'token_entropy' in results
        assert 'structural_entropy' in results
        assert 'semantic_entropy' not in results

        assert results['token_entropy'] >= 0
        assert results['structural_entropy'] >= 0

    def test_with_embeddings(self):
        """Test calculating all entropies with embeddings"""
        responses = ["hello world", "goodbye world"]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

        results = calculate_all_entropies(responses, embeddings)

        assert 'token_entropy' in results
        assert 'structural_entropy' in results
        assert 'semantic_entropy' in results

        assert results['token_entropy'] >= 0
        assert results['structural_entropy'] >= 0
        assert results['semantic_entropy'] >= 0
