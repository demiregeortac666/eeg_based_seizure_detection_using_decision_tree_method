#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_data.py

This script:
 1. Reads the feature matrix
 2. Cleans NaN (missing) values
 3. Saves the cleaned dataset
"""

import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

def clean_data(df, method='fill_zero', target_col='label', threshold=0.5, fill_value=0):
    """
    Cleans NaN values from the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    method : str
        Cleaning method: 'fill_zero', 'fill_mean', 'fill_median', 'drop_rows', 'drop_cols'
    target_col : str
        Target/label column name (always protected)
    threshold : float
        For drop_cols: if the ratio of NaN values in a column exceeds this threshold,
        the column is removed (between 0-1)
    fill_value : int or float
        Value to be used for the fill_value method

    Returns
    -------
    pd.DataFrame
        Cleaned dataset
    """
    # Initial information
    rows_before = df.shape[0]
    cols_before = df.shape[1]

    # Determine all features and NaN statistics
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()

    print(f"Found {total_nans} NaN values ({total_nans/(rows_before*cols_before)*100:.2f}%)")

    # Categorical and target columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    protected_cols = [target_col] if target_col in df.columns else []

    # Identify numerical feature columns (excluding categorical and target)
    feature_cols = [col for col in df.columns if col not in cat_cols + protected_cols]

    if method == 'fill_zero':
        print(f"Filling NaN values with zero...")
        df[feature_cols] = df[feature_cols].fillna(0)

    elif method == 'fill_mean':
        print(f"Filling NaN values with column mean...")
        for col in feature_cols:
            df[col] = df[col].fillna(df[col].mean())

    elif method == 'fill_median':
        print(f"Filling NaN values with column median...")
        for col in feature_cols:
            df[col] = df[col].fillna(df[col].median())

    elif method == 'fill_value':
        print(f"Filling NaN values with {fill_value}...")
        df[feature_cols] = df[feature_cols].fillna(fill_value)

    elif method == 'drop_rows':
        print(f"Removing rows containing NaN values...")
        df = df.dropna()
        print(f"  Number of rows removed: {rows_before - df.shape[0]}")

    elif method == 'drop_cols':
        # Identify columns where NaN ratio is greater than threshold
        nan_ratio = nan_counts / rows_before
        drop_cols = [col for col in feature_cols if nan_ratio[col] > threshold
                    and col not in protected_cols]

        if drop_cols:
            print(f"Removing {len(drop_cols)} columns with NaN ratio exceeding {threshold*100}%...")
            df = df.drop(columns=drop_cols)
            print(f"  Removed columns: {drop_cols[:5]}{'...' if len(drop_cols) > 5 else ''}")
        else:
            print(f"No columns found with NaN ratio exceeding {threshold*100}%.")

    # Result information
    remaining_nans = df.isna().sum().sum()

    if remaining_nans > 0:
        print(f"{remaining_nans} NaN values remain after cleaning.")
    else:
        print(f"All NaN values have been cleaned!")

    rows_after = df.shape[0]
    cols_after = df.shape[1]
    print(f"Dataset: {rows_before}x{cols_before} -> {rows_after}x{cols_after}")

    return df

def plot_nan_distribution(df, output_path="output/nan_distribution.png"):
    """Visualizes the distribution of NaN values"""
    # Calculate NaN distribution
    nan_counts = df.isna().sum().sort_values(ascending=False)
    nan_ratio = (nan_counts / len(df) * 100).round(2)

    # Show first 20 columns with NaN values
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) == 0:
        print("No NaN values found; plot was not generated.")
        return

    plot_cols = min(20, len(nan_cols))
    plt.figure(figsize=(12, 6))
    plt.bar(range(plot_cols), nan_ratio.iloc[:plot_cols], color='skyblue')
    plt.xticks(range(plot_cols), nan_ratio.index[:plot_cols], rotation=90)
    plt.ylabel('NaN Ratio (%)')
    plt.title('NaN Value Distribution (First 20 Columns)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"NaN distribution plot saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Cleans NaN values from EEG feature matrix")
    parser.add_argument("--input", type=str,
                        default="output/normalized_features.csv",
                        help="Input CSV file (default: output/normalized_features.csv)")
    parser.add_argument("--output", type=str,
                        default="output/cleaned_features.csv",
                        help="Output CSV file (default: output/cleaned_features.csv)")
    parser.add_argument("--method", type=str,
                        choices=['fill_zero', 'fill_mean', 'fill_median', 'fill_value',
                                'drop_rows', 'drop_cols'],
                        default='fill_zero',
                        help="Cleaning method (default: fill_zero)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="NaN ratio threshold for drop_cols (between 0-1, default: 0.5)")
    parser.add_argument("--fill-value", type=float, default=0,
                        help="Value to be used for fill_value method (default: 0)")
    parser.add_argument("--plot", action="store_true",
                        help="Visualize NaN distribution")

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

    # Visualize NaN distribution
    if args.plot:
        plot_nan_distribution(df)

    # Clean the data
    cleaned_df = clean_data(
        df, method=args.method,
        threshold=args.threshold,
        fill_value=args.fill_value
    )

    # Save results
    cleaned_df.to_csv(args.output, index=False)
    print(f"Cleaned dataset saved: {args.output}")

    return 0

if __name__ == "__main__":
    main()
