"""
Complete Feature Extraction Pipeline

Combines basic statistical features with EMD-based features for battery RUL prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from .emd_extractor import extract_emd_features_from_cycle, extract_emd_features_from_file


def extract_basic_statistics(cycle_data: pd.DataFrame) -> Dict[str, float]:
    """
    Extract basic statistical features from cycle waveform data.
    
    Parameters:
    -----------
    cycle_data : pd.DataFrame
        Cycle waveform data with columns: Voltage_measured, Current_measured, 
        Temperature_measured, Time
        
    Returns:
    --------
    features : Dict[str, float]
        Dictionary of basic statistical features
    """
    features = {}
    
    # Ensure required columns exist
    required = {"Voltage_measured", "Current_measured", "Temperature_measured", "Time"}
    if not required.issubset(cycle_data.columns):
        return features
    
    # Convert to float arrays
    try:
        v = cycle_data["Voltage_measured"].astype(float)
        i = cycle_data["Current_measured"].astype(float)
        t = cycle_data["Time"].astype(float)
        temp = cycle_data["Temperature_measured"].astype(float)
    except Exception:
        return features
    
    if len(t) < 2:
        return features
    
    # Duration
    features["duration_s"] = float(t.iloc[-1] - t.iloc[0])
    
    # Voltage statistics
    features["voltage_mean"] = float(v.mean())
    features["voltage_min"] = float(v.min())
    features["voltage_max"] = float(v.max())
    features["voltage_std"] = float(v.std())
    
    # Current statistics
    features["current_mean_abs"] = float(np.abs(i).mean())
    features["current_max_abs"] = float(np.abs(i).max())
    features["current_std"] = float(i.std())
    
    # Temperature statistics
    features["temp_max"] = float(temp.max())
    features["temp_min"] = float(temp.min())
    features["temp_mean"] = float(temp.mean())
    features["temp_std"] = float(temp.std())
    
    # Coulomb count (Ah)
    dt = np.diff(t.to_numpy(), prepend=t.iloc[0])
    coulomb_Asec = np.sum(np.abs(i.to_numpy()) * dt)
    features["coulomb_Ah"] = float(coulomb_Asec / 3600.0)
    
    # IR drop proxy
    nz = np.where(np.abs(i.to_numpy()) > 1e-3)[0]
    if len(nz) >= 2:
        features["ir_drop_proxy"] = float(v.iloc[nz[0]] - v.iloc[nz[1]])
    else:
        features["ir_drop_proxy"] = np.nan
    
    # Voltage drop rate (dV/dt)
    if len(v) > 1:
        dv_dt = np.diff(v.to_numpy()) / np.diff(t.to_numpy() + 1e-6)  # Avoid division by zero
        features["voltage_drop_rate_mean"] = float(np.mean(dv_dt))
        features["voltage_drop_rate_max"] = float(np.max(np.abs(dv_dt)))
    
    return features


def extract_all_features(
    cycle_data: pd.DataFrame,
    include_emd: bool = True,
    max_imfs: int = 5
) -> Dict[str, float]:
    """
    Extract all features (statistical + EMD) from a cycle.
    
    Parameters:
    -----------
    cycle_data : pd.DataFrame
        Cycle waveform data
    include_emd : bool
        Whether to include EMD features
    max_imfs : int
        Maximum number of IMFs for EMD
        
    Returns:
    --------
    features : Dict[str, float]
        Dictionary of all extracted features
    """
    # Basic statistical features
    features = extract_basic_statistics(cycle_data)
    
    # EMD features (if enabled and PyEMD is available)
    if include_emd:
        try:
            emd_features = extract_emd_features_from_cycle(cycle_data, max_imfs=max_imfs)
            features.update(emd_features)
        except Exception as e:
            print(f"Warning: EMD feature extraction failed: {e}")
    
    return features


def extract_features_from_file(
    file_path: Path,
    include_emd: bool = True,
    max_imfs: int = 5
) -> Dict[str, float]:
    """
    Extract all features from a cycle CSV file.
    
    Parameters:
    -----------
    file_path : Path
        Path to cycle CSV file
    include_emd : bool
        Whether to include EMD features
    max_imfs : int
        Maximum number of IMFs for EMD
        
    Returns:
    --------
    features : Dict[str, float]
        Dictionary of extracted features
    """
    try:
        cycle_data = pd.read_csv(file_path)
        return extract_all_features(cycle_data, include_emd=include_emd, max_imfs=max_imfs)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}

