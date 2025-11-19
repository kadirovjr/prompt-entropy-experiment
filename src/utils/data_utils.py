"""
Data processing utilities
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import re


def normalize_text(text: str) -> str:
    """
    Normalize text for analysis

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def tokenize_simple(text: str) -> List[str]:
    """
    Simple whitespace tokenization

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    return text.split()


def remove_stopwords(tokens: List[str], stopwords: Optional[List[str]] = None) -> List[str]:
    """
    Remove stopwords from token list

    Args:
        tokens: List of tokens
        stopwords: Optional list of stopwords (uses basic English set if None)

    Returns:
        Filtered token list
    """
    if stopwords is None:
        # Basic English stopwords
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with'
        }

    return [token for token in tokens if token.lower() not in stopwords]


def compute_hash(text: str, algorithm: str = 'sha256') -> str:
    """
    Compute hash of text

    Args:
        text: Input text
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

    Returns:
        Hex digest of hash
    """
    if algorithm == 'md5':
        h = hashlib.md5()
    elif algorithm == 'sha1':
        h = hashlib.sha1()
    elif algorithm == 'sha256':
        h = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    h.update(text.encode('utf-8'))
    return h.hexdigest()


def batch_data(data: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split data into batches

    Args:
        data: List of items
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def deduplicate(items: List[Any], key: Optional[callable] = None) -> List[Any]:
    """
    Remove duplicates from list while preserving order

    Args:
        items: List of items
        key: Optional function to extract comparison key

    Returns:
        Deduplicated list
    """
    seen = set()
    result = []

    for item in items:
        k = key(item) if key else item

        # Handle unhashable types
        try:
            if k not in seen:
                seen.add(k)
                result.append(item)
        except TypeError:
            # Unhashable type, use slower list search
            if k not in [key(x) if key else x for x in result]:
                result.append(item)

    return result


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    sep: str = '.',
) -> Dict[str, Any]:
    """
    Flatten nested dictionary

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator between keys

    Returns:
        Flattened dictionary
    """
    items = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def calculate_percentiles(
    data: List[float],
    percentiles: List[float] = [25, 50, 75],
) -> Dict[str, float]:
    """
    Calculate percentiles of data

    Args:
        data: List of values
        percentiles: List of percentile values (0-100)

    Returns:
        Dictionary mapping percentile to value
    """
    arr = np.array(data)
    return {
        f"p{p}": np.percentile(arr, p)
        for p in percentiles
    }


def z_score_normalize(data: List[float]) -> List[float]:
    """
    Z-score normalization (standardization)

    Args:
        data: List of values

    Returns:
        Normalized values
    """
    arr = np.array(data)
    mean = np.mean(arr)
    std = np.std(arr)

    if std == 0:
        return [0.0] * len(data)

    return ((arr - mean) / std).tolist()


def min_max_normalize(
    data: List[float],
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> List[float]:
    """
    Min-max normalization

    Args:
        data: List of values
        new_min: New minimum value
        new_max: New maximum value

    Returns:
        Normalized values
    """
    arr = np.array(data)
    old_min = np.min(arr)
    old_max = np.max(arr)

    if old_max == old_min:
        return [new_min] * len(data)

    normalized = (arr - old_min) / (old_max - old_min)
    scaled = normalized * (new_max - new_min) + new_min

    return scaled.tolist()


def sliding_window(
    data: List[Any],
    window_size: int,
    step: int = 1,
) -> List[List[Any]]:
    """
    Create sliding windows over data

    Args:
        data: Input data
        window_size: Size of each window
        step: Step size between windows

    Returns:
        List of windows
    """
    windows = []

    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])

    return windows


def aggregate_by_key(
    data: List[Dict[str, Any]],
    key: str,
    value_key: str,
    agg_func: str = 'mean',
) -> Dict[Any, float]:
    """
    Aggregate values by key

    Args:
        data: List of dictionaries
        key: Key to group by
        value_key: Key of values to aggregate
        agg_func: Aggregation function ('mean', 'sum', 'min', 'max', 'count')

    Returns:
        Dictionary mapping key to aggregated value
    """
    from collections import defaultdict

    groups = defaultdict(list)

    for item in data:
        groups[item[key]].append(item[value_key])

    result = {}

    for k, values in groups.items():
        arr = np.array(values)

        if agg_func == 'mean':
            result[k] = float(np.mean(arr))
        elif agg_func == 'sum':
            result[k] = float(np.sum(arr))
        elif agg_func == 'min':
            result[k] = float(np.min(arr))
        elif agg_func == 'max':
            result[k] = float(np.max(arr))
        elif agg_func == 'count':
            result[k] = len(values)
        else:
            raise ValueError(f"Unsupported aggregation: {agg_func}")

    return result
