#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
combine_features.py

This script:
 1. Finds all CSV files in the output/features/ directory
 2. Combines them all into a single CSV file
"""

import pandas as pd
import os
import glob
import argparse
import sys

def combine_feature_files(input_dir, output_file):
    """
    Combines all CSV files in the specified directory

    Parameters
    ----------
    input_dir : str
        Directory containing CSV files
    output_file : str
        Where to save the combined file
    """
    print(f"Starting feature file combination process...")
    print(f"Source directory: {input_dir}")
    print(f"Target file: {output_file}")

    # Find CSV files
    files = glob.glob(os.path.join(input_dir, '*.csv'))
    print(f"Found {len(files)} CSV files in total.")

    if len(files) == 0:
        print("Error: No CSV files found!")
        return False

    # Check the number of columns in each CSV file
    print("Analyzing CSV files...")
    column_counts = {}
    all_dfs = []

    for i, file in enumerate(files):
        try:
            # Read only the header
            header = pd.read_csv(file, nrows=0)
            column_count = len(header.columns)

            if column_count not in column_counts:
                column_counts[column_count] = []

            column_counts[column_count].append(file)

        except Exception as e:
            print(f"File analysis error: {file} - {e}")

    # Report column counts
    print("\nColumn count analysis:")
    for count, files_list in column_counts.items():
        print(f"  Files with {count} columns: {len(files_list)}")

    # Find the most common column count
    most_common_column_count = max(column_counts.items(), key=lambda x: len(x[1]))[0]
    print(f"\nMost common column count: {most_common_column_count}")
    print(f"Will use {len(column_counts[most_common_column_count])} files with this structure")

    # Combine only compatible files
    compatible_files = column_counts[most_common_column_count]

    # Get header information from first file
    first = True
    total_rows = 0
    skipped_files = 0

    # Process each file sequentially
    for i, file in enumerate(files):
        if file not in compatible_files:
            print(f"Skipping: {i+1}/{len(files)} - {os.path.basename(file)} (incompatible column count)")
            skipped_files += 1
            continue

        print(f"Processing: {i+1}/{len(files)} - {os.path.basename(file)}")

        try:
            df = pd.read_csv(file)
            total_rows += len(df)

            # Add headers for the first file
            if first:
                df.to_csv(output_file, index=False)
                first = False
            else:
                # Don't add headers for other files
                df.to_csv(output_file, mode='a', header=False, index=False)

        except Exception as e:
            print(f"Error: {file} - {e}")
            skipped_files += 1

    if total_rows == 0:
        print(f"Error: No files processed successfully!")
        return False

    print(f"\nCompleted! {len(files) - skipped_files} files combined ({skipped_files} files skipped).")
    print(f"Total {total_rows} rows written to {output_file}.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Combine feature files")
    parser.add_argument("--input-dir",
                      default="output/features",
                      help="Directory containing CSV files")
    parser.add_argument("--output",
                      default="output/features_combined.csv",
                      help="Combined file location")

    args = parser.parse_args()

    # Run combination process
    success = combine_feature_files(args.input_dir, args.output)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
