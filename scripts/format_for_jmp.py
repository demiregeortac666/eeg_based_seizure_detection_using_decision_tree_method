#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
format_for_jmp.py

This script:
 1. Reads the balanced dataset CSV file (default: output/balanced_dataset.csv)
 2. Replaces '-' and spaces in column names with '_'
 3. Converts the label column to categorical as 0->'NonSeizure', 1->'Seizure'
 4. Converts patient and file columns to string (nominal) type
 5. Saves the result prepared for JMP (default: output/balanced_dataset_jmp.csv)
"""

import os
import sys
import argparse
import pandas as pd

def format_for_jmp(input_path, output_path):
    """
    Prepares the dataset for JMP

    Parameters
    ----------
    input_path : str
        Path to the CSV file to read
    output_path : str
        Path to the CSV file to write
    """
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"❌ Error: File not found: {input_path}")
        return False

    try:
        # 1) Read CSV
        print(f"📂 Reading: {input_path}")
        df = pd.read_csv(input_path)

        print(f"📊 Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")

        # 2) Clean column names by replacing '-' and spaces
        clean_cols = []
        for c in df.columns:
            # Replace hyphens and spaces with underscores
            new_c = c.replace('-', '_').replace(' ', '_')
            if new_c != c:
                print(f"  ✓ '{c}' -> '{new_c}'")
            clean_cols.append(new_c)
        df.columns = clean_cols

        # 3) Convert label column to categorical
        if 'label' not in df.columns:
            print("❌ Error: Expected 'label' column not found.")
            return False

        # Check if label column has values other than 0 and 1
        invalid_labels = set(df['label'].unique()) - {0, 1}
        if invalid_labels:
            print(f"❌ Error: Unexpected values in 'label' column: {invalid_labels}")
            return False

        # Map label values
        df['label'] = df['label'].map({0: 'NonSeizure', 1: 'Seizure'})
        print(f"  ✓ label column converted to categorical: {df['label'].value_counts().to_dict()}")

        # 4) Convert patient and file columns to string (nominal) type
        for col in ('patient', 'file'):
            if col in df.columns:
                df[col] = df[col].astype(str)
                unique_values = len(df[col].unique())
                print(f"  ✓ '{col}' converted to string type ({unique_values} unique values)")
            else:
                print(f"⚠️  Warning: '{col}' column not found, skipping.")

        # 5) Save the new file
        df.to_csv(output_path, index=False)
        print(f"✨ Dataset formatted for JMP saved: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Prepares balanced dataset for JMP statistical analysis software")
    parser.add_argument("--input", type=str,
                        default=os.path.join("output", "balanced_dataset.csv"),
                        help="CSV file to read (default: output/balanced_dataset.csv)")
    parser.add_argument("--output", type=str,
                        default=os.path.join("output", "balanced_dataset_jmp.csv"),
                        help="CSV file to write (default: output/balanced_dataset_jmp.csv)")

    args = parser.parse_args()

    success = format_for_jmp(args.input, args.output)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
