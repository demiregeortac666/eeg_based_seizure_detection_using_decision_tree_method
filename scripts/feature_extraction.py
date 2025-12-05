# scripts/feature_extraction.py

import numpy as np
import pandas as pd
from scipy import signal, stats

def extract_features(segments, labels, ch_names, fs):
    """
    Extract features from EEG segments

    Parameters
    -----------
    segments : np.ndarray
        Array of shape (N_windows, N_channels, N_samples)
    labels : np.ndarray
        Array of shape (N_windows,) with 0/1 labels
    ch_names : list
        Channel names
    fs : float
        Sampling frequency

    Returns
    --------
    pd.DataFrame
        All extracted features and labels
    """
    # Features to extract
    features = []

    for i, (segment, label) in enumerate(zip(segments, labels)):
        # segment: (N_channels, N_samples)
        feat_dict = {"label": label}

        # Features for each channel
        for c, ch in enumerate(ch_names):
            # Time domain features
            channel_data = segment[c]
            feat_dict[f"{ch}_mean"] = np.mean(channel_data)
            feat_dict[f"{ch}_std"] = np.std(channel_data)
            feat_dict[f"{ch}_kurt"] = stats.kurtosis(channel_data)
            feat_dict[f"{ch}_skew"] = stats.skew(channel_data)

            # Frequency domain features
            freqs, psd = signal.welch(channel_data, fs, nperseg=min(256, len(channel_data)))

            # Power in specific frequency bands
            delta_mask = (freqs >= 0.5) & (freqs < 4)
            theta_mask = (freqs >= 4) & (freqs < 8)
            alpha_mask = (freqs >= 8) & (freqs < 13)
            beta_mask = (freqs >= 13) & (freqs < 30)
            gamma_mask = (freqs >= 30) & (freqs <= 100)

            feat_dict[f"{ch}_delta"] = np.sum(psd[delta_mask])
            feat_dict[f"{ch}_theta"] = np.sum(psd[theta_mask])
            feat_dict[f"{ch}_alpha"] = np.sum(psd[alpha_mask])
            feat_dict[f"{ch}_beta"] = np.sum(psd[beta_mask])
            feat_dict[f"{ch}_gamma"] = np.sum(psd[gamma_mask])

        features.append(feat_dict)

    return pd.DataFrame(features)
