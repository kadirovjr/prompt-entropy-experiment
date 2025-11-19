"""
Unit tests for utility functions
"""

import pytest
import tempfile
import os
from pathlib import Path
import pandas as pd

from src.utils.data_utils import (
    normalize_text,
    tokenize_simple,
    remove_stopwords,
    compute_hash,
    batch_data,
    deduplicate,
    flatten_dict,
    calculate_percentiles,
    z_score_normalize,
    min_max_normalize,
    sliding_window,
    aggregate_by_key,
)

from src.utils.io_utils import (
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    save_csv,
    load_csv,
    ensure_dir,
    list_files,
)


class TestNormalizeText:
    def test_lowercase(self):
        """Test lowercase conversion"""
        assert normalize_text("HELLO WORLD") == "hello world"

    def test_extra_whitespace(self):
        """Test whitespace normalization"""
        assert normalize_text("hello    world  ") == "hello world"

    def test_combined(self):
        """Test combined normalization"""
        assert normalize_text("  HELLO   WORLD  ") == "hello world"


class TestTokenize:
    def test_simple_tokenization(self):
        """Test simple whitespace tokenization"""
        tokens = tokenize_simple("hello world foo bar")
        assert tokens == ["hello", "world", "foo", "bar"]

    def test_empty_string(self):
        """Test empty string"""
        tokens = tokenize_simple("")
        assert tokens == [""]


class TestRemoveStopwords:
    def test_basic_stopwords(self):
        """Test removing basic stopwords"""
        tokens = ["the", "quick", "brown", "fox"]
        filtered = remove_stopwords(tokens)
        assert "the" not in filtered
        assert "quick" in filtered
        assert "brown" in filtered

    def test_custom_stopwords(self):
        """Test with custom stopwords"""
        tokens = ["apple", "banana", "orange"]
        filtered = remove_stopwords(tokens, stopwords=["banana"])
        assert filtered == ["apple", "orange"]


class TestComputeHash:
    def test_sha256(self):
        """Test SHA256 hashing"""
        hash1 = compute_hash("hello", algorithm="sha256")
        hash2 = compute_hash("hello", algorithm="sha256")
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex characters

    def test_md5(self):
        """Test MD5 hashing"""
        hash1 = compute_hash("hello", algorithm="md5")
        assert len(hash1) == 32  # MD5 produces 32 hex characters

    def test_different_inputs(self):
        """Test that different inputs produce different hashes"""
        hash1 = compute_hash("hello")
        hash2 = compute_hash("world")
        assert hash1 != hash2


class TestBatchData:
    def test_even_batches(self):
        """Test with data that divides evenly"""
        data = list(range(10))
        batches = batch_data(data, batch_size=2)
        assert len(batches) == 5
        assert batches[0] == [0, 1]
        assert batches[-1] == [8, 9]

    def test_uneven_batches(self):
        """Test with data that doesn't divide evenly"""
        data = list(range(10))
        batches = batch_data(data, batch_size=3)
        assert len(batches) == 4
        assert batches[-1] == [9]  # Last batch is smaller


class TestDeduplicate:
    def test_simple_deduplication(self):
        """Test simple list deduplication"""
        items = [1, 2, 2, 3, 3, 3, 4]
        result = deduplicate(items)
        assert result == [1, 2, 3, 4]

    def test_preserves_order(self):
        """Test that order is preserved"""
        items = [3, 1, 2, 1, 3]
        result = deduplicate(items)
        assert result == [3, 1, 2]

    def test_with_key_function(self):
        """Test deduplication with key function"""
        items = ["apple", "Apple", "banana", "BANANA"]
        result = deduplicate(items, key=str.lower)
        assert len(result) == 2


class TestFlattenDict:
    def test_nested_dict(self):
        """Test flattening nested dictionary"""
        d = {
            'a': 1,
            'b': {
                'c': 2,
                'd': 3,
            }
        }
        flat = flatten_dict(d)
        assert flat == {'a': 1, 'b.c': 2, 'b.d': 3}

    def test_deeply_nested(self):
        """Test deeply nested dictionary"""
        d = {'a': {'b': {'c': 1}}}
        flat = flatten_dict(d)
        assert flat == {'a.b.c': 1}

    def test_custom_separator(self):
        """Test custom separator"""
        d = {'a': {'b': 1}}
        flat = flatten_dict(d, sep='_')
        assert flat == {'a_b': 1}


class TestCalculatePercentiles:
    def test_basic_percentiles(self):
        """Test basic percentile calculation"""
        data = list(range(101))  # 0 to 100
        percentiles = calculate_percentiles(data)

        assert abs(percentiles['p25'] - 25.0) < 1.0
        assert abs(percentiles['p50'] - 50.0) < 1.0
        assert abs(percentiles['p75'] - 75.0) < 1.0

    def test_custom_percentiles(self):
        """Test custom percentile values"""
        data = list(range(101))
        percentiles = calculate_percentiles(data, percentiles=[10, 90])

        assert 'p10' in percentiles
        assert 'p90' in percentiles


class TestZScoreNormalize:
    def test_normalization(self):
        """Test z-score normalization"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = z_score_normalize(data)

        # Mean should be ~0
        assert abs(sum(normalized) / len(normalized)) < 1e-6

        # Std should be ~1
        import numpy as np
        assert abs(np.std(normalized) - 1.0) < 1e-6

    def test_constant_data(self):
        """Test with constant data"""
        data = [5.0, 5.0, 5.0]
        normalized = z_score_normalize(data)

        # Should return all zeros
        assert all(x == 0.0 for x in normalized)


class TestMinMaxNormalize:
    def test_default_range(self):
        """Test min-max normalization to [0, 1]"""
        data = [0.0, 5.0, 10.0]
        normalized = min_max_normalize(data)

        assert normalized[0] == 0.0
        assert normalized[-1] == 1.0
        assert normalized[1] == 0.5

    def test_custom_range(self):
        """Test min-max normalization to custom range"""
        data = [0.0, 10.0]
        normalized = min_max_normalize(data, new_min=-1.0, new_max=1.0)

        assert normalized[0] == -1.0
        assert normalized[1] == 1.0


class TestSlidingWindow:
    def test_basic_windows(self):
        """Test basic sliding window"""
        data = [1, 2, 3, 4, 5]
        windows = sliding_window(data, window_size=3, step=1)

        assert len(windows) == 3
        assert windows[0] == [1, 2, 3]
        assert windows[1] == [2, 3, 4]
        assert windows[2] == [3, 4, 5]

    def test_with_step(self):
        """Test sliding window with step > 1"""
        data = [1, 2, 3, 4, 5, 6]
        windows = sliding_window(data, window_size=2, step=2)

        assert len(windows) == 3
        assert windows[0] == [1, 2]
        assert windows[1] == [3, 4]
        assert windows[2] == [5, 6]


class TestAggregateByKey:
    def test_mean_aggregation(self):
        """Test mean aggregation"""
        data = [
            {'category': 'A', 'value': 10},
            {'category': 'A', 'value': 20},
            {'category': 'B', 'value': 30},
        ]
        result = aggregate_by_key(data, 'category', 'value', 'mean')

        assert result['A'] == 15.0
        assert result['B'] == 30.0

    def test_sum_aggregation(self):
        """Test sum aggregation"""
        data = [
            {'category': 'A', 'value': 10},
            {'category': 'A', 'value': 20},
        ]
        result = aggregate_by_key(data, 'category', 'value', 'sum')

        assert result['A'] == 30.0

    def test_count_aggregation(self):
        """Test count aggregation"""
        data = [
            {'category': 'A', 'value': 10},
            {'category': 'A', 'value': 20},
            {'category': 'B', 'value': 30},
        ]
        result = aggregate_by_key(data, 'category', 'value', 'count')

        assert result['A'] == 2
        assert result['B'] == 1


class TestIOUtils:
    def test_json_roundtrip(self):
        """Test JSON save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.json')
            data = {'key': 'value', 'number': 42}

            save_json(data, filepath)
            loaded = load_json(filepath)

            assert loaded == data

    def test_pickle_roundtrip(self):
        """Test pickle save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.pkl')
            data = {'key': 'value', 'list': [1, 2, 3]}

            save_pickle(data, filepath)
            loaded = load_pickle(filepath)

            assert loaded == data

    def test_csv_roundtrip(self):
        """Test CSV save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.csv')
            data = [
                {'name': 'Alice', 'age': '30'},
                {'name': 'Bob', 'age': '25'},
            ]

            save_csv(data, filepath)
            loaded = load_csv(filepath)

            assert len(loaded) == 2
            assert loaded[0]['name'] == 'Alice'

    def test_ensure_dir(self):
        """Test directory creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, 'subdir', 'nested')
            path = ensure_dir(new_dir)

            assert path.exists()
            assert path.is_dir()

    def test_list_files(self):
        """Test file listing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test files
            Path(tmpdir, 'test1.txt').touch()
            Path(tmpdir, 'test2.txt').touch()
            Path(tmpdir, 'other.json').touch()

            files = list_files(tmpdir, pattern='*.txt')

            assert len(files) == 2
