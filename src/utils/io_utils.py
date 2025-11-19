"""
Input/Output utilities for data management
"""

import json
import pickle
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to JSON file

    Args:
        data: Data to save
        filepath: Path to output file
        indent: JSON indentation level
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file

    Args:
        filepath: Path to input file

    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data to pickle file

    Args:
        data: Data to save
        filepath: Path to output file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file

    Args:
        filepath: Path to input file

    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_csv(
    data: List[Dict[str, Any]],
    filepath: str,
    fieldnames: Optional[List[str]] = None,
) -> None:
    """
    Save list of dictionaries to CSV file

    Args:
        data: List of dictionaries
        filepath: Path to output file
        fieldnames: Optional list of field names (uses first dict keys if None)
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not data:
        return

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def load_csv(filepath: str) -> List[Dict[str, Any]]:
    """
    Load CSV file as list of dictionaries

    Args:
        filepath: Path to input file

    Returns:
        List of dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_dataframe(df: pd.DataFrame, filepath: str, format: str = 'csv') -> None:
    """
    Save pandas DataFrame to file

    Args:
        df: DataFrame to save
        filepath: Path to output file
        format: File format ('csv', 'json', 'parquet', 'pickle')
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'csv':
        df.to_csv(path, index=False)
    elif format == 'json':
        df.to_json(path, orient='records', indent=2)
    elif format == 'parquet':
        df.to_parquet(path, index=False)
    elif format == 'pickle':
        df.to_pickle(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(filepath: str, format: Optional[str] = None) -> pd.DataFrame:
    """
    Load pandas DataFrame from file

    Args:
        filepath: Path to input file
        format: File format (auto-detected from extension if None)

    Returns:
        Loaded DataFrame
    """
    path = Path(filepath)

    if format is None:
        format = path.suffix[1:]  # Remove leading dot

    if format == 'csv':
        return pd.read_csv(path)
    elif format == 'json':
        return pd.read_json(path)
    elif format == 'parquet':
        return pd.read_parquet(path)
    elif format in ('pickle', 'pkl'):
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def ensure_dir(dirpath: str) -> Path:
    """
    Ensure directory exists, create if needed

    Args:
        dirpath: Directory path

    Returns:
        Path object
    """
    path = Path(dirpath)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(
    directory: str,
    pattern: str = "*",
    recursive: bool = False,
) -> List[Path]:
    """
    List files in directory matching pattern

    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., "*.json")
        recursive: Search recursively

    Returns:
        List of matching file paths
    """
    path = Path(directory)

    if recursive:
        return sorted(path.rglob(pattern))
    else:
        return sorted(path.glob(pattern))
