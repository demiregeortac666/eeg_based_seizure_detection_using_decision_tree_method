#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
feature_selection.py

This script:
 1. Reads the feature matrix
 2. Performs correlation-based and importance-based feature selection
 3. Creates a smaller dataset with selected features
"""

import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

def correlation_filter(df, threshold=0.9, target_col='label', plot=False):
    """
    Filters out features with high correlation

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    threshold : float
        Correlation threshold (default: 0.9)
    target_col : str
        Target/label column name
    plot : bool
        Visualize correlation matrix

    Returns
    -------
    pd.DataFrame
        Filtered feature matrix
    """
    # Select only numeric features
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Exclude target variable
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()

    # Visualize correlation matrix
    if plot:
        plt.figure(figsize=(12, 10))
        plt.title("Feature Correlation Matrix", fontsize=15)
        plt.imshow(corr_matrix, cmap='coolwarm')
        plt.colorbar()
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.tight_layout()
        plt.savefig("output/correlation_matrix.png", dpi=150)
        print("✓ Correlation matrix saved: output/correlation_matrix.png")

    # Identify features with high correlation
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]

    print(f"ℹ️ Removed {len(to_drop)} highly correlated features.")
    print(f"  Threshold value: {threshold}")

    # Return remaining features
    return df.drop(columns=to_drop)

def feature_importance(df, target_col='label', n_features=30, plot=False):
    """
    Calculates feature importance using a Random Forest model

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    target_col : str
        Target/label column name
    n_features : int
        Number of features to select
    plot : bool
        Visualize feature importance

    Returns
    -------
    pd.DataFrame
        Features filtered by importance ranking
    """
    # Select numeric features, excluding categorical
    exclude_cols = [col for col in df.columns if
                     col == target_col or
                     df[col].dtype == 'object' or
                     df[col].nunique() < 5]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Standardize features
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Target variable
    y = df[target_col].values

    # Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)

    # Calculate feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Visualize feature importance
    if plot:
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance Ranking", fontsize=15)
        plt.bar(range(min(30, len(feature_cols))),
                importances[indices[:30]], color='royalblue')
        plt.xticks(range(min(30, len(feature_cols))),
                   [feature_cols[i] for i in indices[:30]], rotation=90)
        plt.tight_layout()
        plt.savefig("output/feature_importance.png", dpi=150)
        print("✓ Feature importance chart saved: output/feature_importance.png")

    # Create DataFrame with feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Export feature importance table
    importance_df.to_csv("output/feature_importance.csv", index=False)
    print(f"✓ Feature importance table saved: output/feature_importance.csv")

    # Select top n features
    top_features = [feature_cols[i] for i in indices[:n_features]]

    # Include all variables (including categorical)
    selected_cols = top_features + exclude_cols

    print(f"ℹ️ Selected {len(top_features)} features by importance ranking.")

    return df[selected_cols]

def univariate_selection(df, target_col='label', method='f_classif', n_features=30):
    """
    Univariate feature selection

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    target_col : str
        Target/label column name
    method : str
        Selection method ('f_classif' or 'mutual_info')
    n_features : int
        Number of features to select

    Returns
    -------
    pd.DataFrame
        DataFrame with selected features
    """
    # Exclude target variable and categorical variables
    exclude_cols = [col for col in df.columns if
                    col == target_col or
                    df[col].dtype == 'object' or
                    df[col].nunique() < 5]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Standardize features
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Target variable
    y = df[target_col].values

    # Selection method
    if method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=n_features)
    else:  # Default: f_classif
        selector = SelectKBest(f_classif, k=n_features)

    # Feature selection
    selector.fit(X_scaled, y)

    # Selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_cols[i] for i in selected_indices]

    # Include all variables (including categorical)
    selected_cols = selected_features + exclude_cols

    print(f"ℹ️ Selected {len(selected_features)} features using {method} method.")

    return df[selected_cols]

def main():
    parser = argparse.ArgumentParser(
        description="Performs feature selection for EEG feature matrix")
    parser.add_argument("--input", type=str,
                        default="output/balanced_dataset.csv",
                        help="Input CSV file (default: output/balanced_dataset.csv)")
    parser.add_argument("--output", type=str,
                        default="output/selected_features.csv",
                        help="Output CSV file (default: output/selected_features.csv)")
    parser.add_argument("--method", type=str, choices=['correlation', 'importance', 'f_classif', 'mutual_info'],
                        default='importance',
                        help="Feature selection method (default: importance)")
    parser.add_argument("--n_features", type=int, default=30,
                        help="Number of features to select (default: 30)")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Correlation threshold (for correlation method) (default: 0.9)")
    parser.add_argument("--plot", action="store_true",
                        help="Create visualizations")

    args = parser.parse_args()

    # Check input file
    if not os.path.exists(args.input):
        print(f"❌ Error: File not found: {args.input}")
        return 1

    # Check output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Reading: {args.input}")
    df = pd.read_csv(args.input)

    print(f"📊 Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Feature selection
    if args.method == 'correlation':
        selected_df = correlation_filter(df, threshold=args.threshold, plot=args.plot)
    elif args.method == 'importance':
        selected_df = feature_importance(df, n_features=args.n_features, plot=args.plot)
    elif args.method == 'f_classif':
        selected_df = univariate_selection(df, method='f_classif', n_features=args.n_features)
    elif args.method == 'mutual_info':
        selected_df = univariate_selection(df, method='mutual_info', n_features=args.n_features)

    # Save results
    selected_df.to_csv(args.output, index=False)
    print(f"✨ Dataset with selected features saved: {args.output}")
    print(f"  New dimensions: {selected_df.shape[0]} rows, {selected_df.shape[1]} columns")

    return 0

if __name__ == "__main__":
    main()
