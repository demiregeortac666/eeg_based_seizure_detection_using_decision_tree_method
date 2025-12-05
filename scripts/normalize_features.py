#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
normalize_features.py

This script:
 1. Reads the feature matrix
 2. Applies different normalization/standardization methods
 3. Saves the normalized dataset
"""

import os
import pandas as pd
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

def standardize_features(df, method='standard', target_col='label',
                        exclude_cols=None, save_scaler=True, plot=False):
    """
    Normalizes features in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    method : str
        Normalization method: 'standard', 'minmax', 'robust', 'yeo-johnson', 'quantile'
    target_col : str
        Target/label column name
    exclude_cols : list or None
        Columns to exclude from transformation
    save_scaler : bool
        Whether to save the scaler object
    plot : bool
        Whether to visualize distribution before/after normalization

    Returns
    -------
    pd.DataFrame
        Normalized features
    """
    if exclude_cols is None:
        exclude_cols = []

    # Keep the target variable out of transformation
    if target_col not in exclude_cols:
        exclude_cols.append(target_col)

    # Identify categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    exclude_cols.extend(cat_cols)

    # Determine columns to be transformed
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    if not feature_cols:
        print("Warning: No numerical columns found to transform!")
        return df

    # Display distribution of first few columns (before normalization)
    if plot:
        plot_sample = min(5, len(feature_cols))
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(feature_cols[:plot_sample]):
            plt.subplot(1, plot_sample, i+1)
            plt.hist(df[col], bins=30, alpha=0.7)
            plt.title(f"{col} (Original)")
        plt.tight_layout()
        plt.savefig("output/pre_normalization.png", dpi=150)
        print("Distribution before normalization saved: output/pre_normalization.png")

    # Select transformation method
    if method == 'minmax':
        scaler = MinMaxScaler()
        print("Applying MinMax scaling (values between 0 and 1)")
    elif method == 'robust':
        scaler = RobustScaler()
        print("Applying Robust scaling (median=0, IQR=1)")
    elif method == 'yeo-johnson':
        scaler = PowerTransformer(method='yeo-johnson')
        print("Applying Yeo-Johnson transformation (approximating normal distribution)")
    elif method == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal')
        print("Applying Quantile transformation (uniform->normal)")
    else:  # Default: standard
        scaler = StandardScaler()
        print("Applying Standard scaling (mean=0, std=1)")

    # Apply transformation
    normalized = df.copy()
    normalized[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Display distribution of first few columns (after normalization)
    if plot:
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(feature_cols[:plot_sample]):
            plt.subplot(1, plot_sample, i+1)
            plt.hist(normalized[col], bins=30, alpha=0.7)
            plt.title(f"{col} ({method})")
        plt.tight_layout()
        plt.savefig(f"output/post_normalization_{method}.png", dpi=150)
        print(f"Distribution after normalization saved: output/post_normalization_{method}.png")

    # Save the scaler
    if save_scaler:
        scaler_path = f"output/scaler_{method}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump((scaler, feature_cols), f)
        print(f"Scaler saved: {scaler_path}")

    print(f"{len(feature_cols)} numerical features have been normalized.")

    return normalized

def main():
    parser = argparse.ArgumentParser(
        description="Normalizes EEG feature matrix")
    parser.add_argument("--input", type=str,
                        default="output/balanced_dataset.csv",
                        help="Input CSV file (default: output/balanced_dataset.csv)")
    parser.add_argument("--output", type=str,
                        default="output/normalized_features.csv",
                        help="Output CSV file (default: output/normalized_features.csv)")
    parser.add_argument("--method", type=str,
                        choices=['standard', 'minmax', 'robust', 'yeo-johnson', 'quantile'],
                        default='standard',
                        help="Normalization method (default: standard)")
    parser.add_argument("--exclude", type=str, nargs='+', default=[],
                        help="Column names to exclude from transformation")
    parser.add_argument("--no-save-scaler", action="store_false", dest="save_scaler",
                        help="Do not save the scaler object")
    parser.add_argument("--plot", action="store_true",
                        help="Visualize distributions before/after normalization")

    args = parser.parse_args()

    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1

    # Check output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input)

    print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Normalization
    normalized_df = standardize_features(
        df, method=args.method,
        exclude_cols=args.exclude,
        save_scaler=args.save_scaler,
        plot=args.plot
    )

    # Save results
    normalized_df.to_csv(args.output, index=False)
    print(f"Normalized dataset saved: {args.output}")

    return 0

if __name__ == "__main__":
    main()
