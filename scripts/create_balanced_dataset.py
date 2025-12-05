#!/usr/bin/env python3
# scripts/create_balanced_dataset.py

import os
import argparse
import pandas as pd

def create_balanced_dataset(input_file, output_file, pos_ratio=2):
    """
    Creates a new dataset by balancing positive and negative samples

    Parameters
    ----------
    input_file : str
        Input CSV file
    output_file : str
        Output CSV file
    pos_ratio : int, default=2
        Negative/Positive sample ratio (2 = 2x negative samples)
    """
    print(f"Reading: {input_file}")
    df = pd.read_csv(input_file)

    # Basic check
    if "label" not in df.columns:
        raise ValueError("'label' column not found in the dataset!")

    # Separate classes
    seizure_samples = df[df.label==1]
    non_seizure_samples = df[df.label==0]

    print(f"Original dataset:")
    print(f"  Total samples: {len(df)}")
    print(f"  Seizure samples: {len(seizure_samples)}")
    print(f"  Normal samples: {len(non_seizure_samples)}")

    if len(seizure_samples) == 0:
        raise ValueError("No seizure samples found in the dataset!")

    # Select negative samples
    n_neg_samples = min(len(non_seizure_samples), len(seizure_samples) * pos_ratio)
    selected_non_seizure = non_seizure_samples.sample(
        n=n_neg_samples, random_state=42)

    # Create and shuffle the dataset
    balanced = pd.concat([seizure_samples, selected_non_seizure],
                         ignore_index=True).sample(frac=1, random_state=42)

    # Save
    balanced.to_csv(output_file, index=False)

    print("\n✅ Balanced dataset saved:")
    print(f"  File: {output_file}")
    print(f"  Total samples: {len(balanced)}")
    print(f"  Class distribution:")
    print(balanced.label.value_counts())

    return balanced

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a balanced dataset for EEG data")
    parser.add_argument("--input", type=str,
                        default="output/features_combined.csv",
                        help="Input feature CSV file")
    parser.add_argument("--output", type=str,
                        default="output/balanced_dataset.csv",
                        help="Output CSV file")
    parser.add_argument("--ratio", type=int, default=2,
                        help="Negative/Positive sample ratio (default: 2)")

    args = parser.parse_args()

    create_balanced_dataset(args.input, args.output, args.ratio)
