"""
EMD (Empirical Mode Decomposition) Feature Extraction for Battery Signals

This module extracts features from voltage, current, and temperature signals
using Empirical Mode Decomposition (EMD) to capture multi-scale temporal patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    from PyEMD import EMD
except ImportError:
    print("Warning: PyEMD not installed. Install with: pip install PyEMD")
    EMD = None


def extract_imf_statistics(imfs: np.ndarray, signal_name: str = "") -> Dict[str, float]:
    """
    Extract statistical features from IMFs (Intrinsic Mode Functions).
    
    Parameters:
    -----------
    imfs : np.ndarray
        Array of shape (n_imfs, n_samples) containing the IMFs
    signal_name : str
        Prefix for feature names (e.g., "voltage", "current")
        
    Returns:
    --------
    features : Dict[str, float]
        Dictionary of extracted features
    """
    features = {}
    n_imfs = imfs.shape[0]
    
    for i in range(n_imfs):
        imf = imfs[i, :]
        prefix = f"{signal_name}_imf{i+1}" if signal_name else f"imf{i+1}"
        
        # Basic statistics
        features[f"{prefix}_mean"] = float(np.mean(imf))
        features[f"{prefix}_std"] = float(np.std(imf))
        features[f"{prefix}_energy"] = float(np.sum(imf ** 2))
        features[f"{prefix}_max"] = float(np.max(imf))
        features[f"{prefix}_min"] = float(np.min(imf))
        
        # Higher-order statistics
        if len(imf) > 3:
            from scipy import stats
            features[f"{prefix}_skewness"] = float(stats.skew(imf))
            features[f"{prefix}_kurtosis"] = float(stats.kurtosis(imf))
        else:
            features[f"{prefix}_skewness"] = 0.0
            features[f"{prefix}_kurtosis"] = 0.0
    
    # Cross-IMF features (correlations between adjacent IMFs)
    if n_imfs > 1:
        for i in range(n_imfs - 1):
            corr = np.corrcoef(imfs[i], imfs[i+1])[0, 1]
            prefix = f"{signal_name}_imf_corr" if signal_name else "imf_corr"
            features[f"{prefix}_{i+1}_{i+2}"] = float(corr)
    
    # Total energy distribution
    total_energy = np.sum([features[f"{prefix}_energy"] 
                           for prefix in [f"{signal_name}_imf{j+1}" if signal_name else f"imf{j+1}" 
                                         for j in range(n_imfs)]])
    if total_energy > 0:
        for i in range(n_imfs):
            prefix = f"{signal_name}_imf{i+1}" if signal_name else f"imf{i+1}"
            energy = features.get(f"{prefix}_energy", 0)
            features[f"{prefix}_energy_ratio"] = float(energy / total_energy)
    
    return features


def extract_emd_features_from_cycle(
    cycle_data: pd.DataFrame,
    max_imfs: int = 5,
    signals: List[str] = None
) -> Dict[str, float]:
    """
    Extract EMD features from a single battery cycle.
    
    Parameters:
    -----------
    cycle_data : pd.DataFrame
        Cycle waveform data with columns: Voltage_measured, Current_measured, 
        Temperature_measured, Time
    max_imfs : int
        Maximum number of IMFs to extract
    signals : List[str]
        List of signal names to process. Default: ['Voltage_measured', 'Current_measured', 'Temperature_measured']
        
    Returns:
    --------
    features : Dict[str, float]
        Dictionary of EMD-based features
    """
    if EMD is None:
        return {}
    
    if signals is None:
        signals = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
    
    features = {}
    emd = EMD()
    
    for signal_name in signals:
        if signal_name not in cycle_data.columns:
            continue
            
        signal = cycle_data[signal_name].values.astype(float)
        
        # Remove NaN and handle edge cases
        signal = signal[~np.isnan(signal)]
        if len(signal) < 10:  # Need minimum samples for EMD
            continue
        
        # Normalize signal to improve EMD stability
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        if signal_std > 0:
            signal_normalized = (signal - signal_mean) / signal_std
        else:
            signal_normalized = signal - signal_mean
        
        try:
            # Perform EMD decomposition
            imfs = emd(signal_normalized, max_imf=max_imfs)
            
            # Handle case where EMD returns fewer IMFs than requested
            if len(imfs.shape) == 1:
                imfs = imfs.reshape(1, -1)
            
            # Extract features from IMFs
            signal_prefix = signal_name.replace("_measured", "").lower()
            imf_features = extract_imf_statistics(imfs, signal_prefix)
            features.update(imf_features)
            
        except Exception as e:
            # If EMD fails, return empty features for this signal
            print(f"Warning: EMD failed for {signal_name}: {e}")
            continue
    
    return features


def extract_emd_features_from_file(
    file_path: Path,
    max_imfs: int = 5
) -> Dict[str, float]:
    """
    Extract EMD features from a cycle CSV file.
    
    Parameters:
    -----------
    file_path : Path
        Path to the cycle CSV file
    max_imfs : int
        Maximum number of IMFs to extract
        
    Returns:
    --------
    features : Dict[str, float]
        Dictionary of EMD-based features
    """
    try:
        cycle_data = pd.read_csv(file_path)
        return extract_emd_features_from_cycle(cycle_data, max_imfs=max_imfs)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}


def batch_extract_emd_features(
    file_paths: List[Path],
    max_imfs: int = 5,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Extract EMD features from multiple cycle files.
    
    Parameters:
    -----------
    file_paths : List[Path]
        List of paths to cycle CSV files
    max_imfs : int
        Maximum number of IMFs to extract
    show_progress : bool
        Whether to show progress bar
        
    Returns:
    --------
    features_df : pd.DataFrame
        DataFrame with EMD features, indexed by filename
    """
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(file_paths, desc="Extracting EMD features")
    else:
        iterator = file_paths
    
    records = []
    for fp in iterator:
        feats = extract_emd_features_from_file(fp, max_imfs=max_imfs)
        feats['filename'] = fp.name
        records.append(feats)
    
    return pd.DataFrame.from_records(records)

