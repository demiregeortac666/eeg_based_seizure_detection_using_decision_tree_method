#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV File Validation and Merging Script

This script validates and merges CSV files from the output/features directory.
It ensures all files have consistent column structures before combining them.
"""

import pandas as pd
import glob
import os

def main():
    print("Checking CSV files...")

    files = glob.glob("output/features/*.csv")

    if not files:
        print("No CSV files found in output/features directory.")
        return

    # Read first file as reference
    first_file = files[0]
    print(f"Reference file: {first_file}")

    try:
        first_df = pd.read_csv(first_file)
        header = first_df.columns
        column_count = len(header)
        print(f"Reference column count: {column_count}")
    except Exception as e:
        print(f"Error reading reference file: {str(e)}")
        return

    # Combine CSV files
    print("Merging CSV files...")
    all_dfs = []
    skipped_files = []

    for file in files:
        try:
            df = pd.read_csv(file)

            # Check column consistency
            if len(df.columns) == column_count:
                all_dfs.append(df)
            else:
                print(f"Column count mismatch: {file}, has {len(df.columns)} columns")
                skipped_files.append(file)

        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
            skipped_files.append(file)

    # Report results
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {len(all_dfs)} files")
    print(f"  Skipped: {len(skipped_files)} files")

    if skipped_files:
        print("\nSkipped files:")
        for f in skipped_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")

    # Combine and save
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        output_path = "output/features_combined.csv"
        combined_df.to_csv(output_path, index=False)

        print(f"\nCombined dataset saved: {output_path}")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Total columns: {len(combined_df.columns)}")

        # Show class distribution if 'label' column exists
        if 'label' in combined_df.columns:
            print(f"  Class distribution: {combined_df['label'].value_counts().to_dict()}")
    else:
        print("\nNo files could be combined.")

if __name__ == "__main__":
    main()
