#!/usr/bin/env python3
# scripts/extract_all_features.py

import os
import sys
import numpy as np
import pandas as pd
from read_eeg_and_summary import segment_eeg_file
from feature_extraction import extract_features

# Relative paths for project root directory
DATA_ROOT = "data/physionet.org/files/chbmit/1.0.0"
OUTPUT_DIR = "output/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_segments(segs, lbls, chs, fs):
    """Check if segment data is valid"""
    if len(segs) == 0:
        return False, "Empty segment array"

    if fs <= 0:
        return False, "Invalid sampling frequency"

    if len(chs) == 0:
        return False, "Channel names not found"

    if len(lbls) != len(segs):
        return False, f"Segment ({len(segs)}) and label ({len(lbls)}) sizes don't match"

    return True, "OK"

def process_files(patients=None):
    """
    Extract features from EDF files in the CHB-MIT dataset

    Parameters
    ----------
    patients : list or None
        List of patient IDs to process (None: all patients)
    """
    all_dfs = []
    processed_count = 0
    error_count = 0
    skipped_count = 0

    # List all patients or specified patients
    patient_dirs = []
    if patients:
        for p in patients:
            pdir = os.path.join(DATA_ROOT, p)
            if os.path.isdir(pdir):
                patient_dirs.append((p, pdir))
            else:
                print(f"Warning: Patient directory not found: {p}")
    else:
        for p in sorted(os.listdir(DATA_ROOT)):
            pdir = os.path.join(DATA_ROOT, p)
            if os.path.isdir(pdir) and p.startswith('chb'):
                patient_dirs.append((p, pdir))

    # Show number of patients to process
    print(f"Processing {len(patient_dirs)} patients")

    # Process each patient
    for patient, pdir in patient_dirs:
        print(f"\nProcessing patient: {patient}")

        patient_files = []
        for fname in sorted(os.listdir(pdir)):
            if fname.endswith(".edf"):
                patient_files.append(fname)

        print(f"   Found {len(patient_files)} EDF files")

        patient_processed = 0
        patient_errors = 0

        for fname in patient_files:
            edf_path = os.path.join(pdir, fname)
            seizure_path = edf_path + ".seizures"

            print(f"\nProcessing: {edf_path}")

            # Segment & extract features
            try:
                segs, lbls, chs, fs = segment_eeg_file(edf_path, seizure_path)

                # Check if segment data is valid
                valid, reason = check_segments(segs, lbls, chs, fs)
                if not valid:
                    print(f"Invalid segment: {edf_path} - {reason}")
                    skipped_count += 1
                    continue

                # Extract features
                df = extract_features(segs, lbls, chs, fs)

                # Add metadata
                df["patient"] = patient
                df["file"] = fname

                # Add to combined dataset
                all_dfs.append(df)

                # Save file separately
                outp = os.path.join(OUTPUT_DIR, f"feat_{patient}_{fname}.csv")
                df.to_csv(outp, index=False)
                print(f"Saved: {outp}")
                print(f"   {len(df)} samples, {df.shape[1]} features")
                print(f"   Class distribution: {df['label'].value_counts().to_dict()}")

                patient_processed += 1
                processed_count += 1

            except Exception as e:
                print(f"Error: {edf_path} - {str(e)}")
                import traceback
                traceback.print_exc()
                patient_errors += 1
                error_count += 1

        print(f"\nPatient {patient} summary:")
        print(f"   Processed: {patient_processed}/{len(patient_files)} files")
        print(f"   Errors: {patient_errors} files")

    # Show processing results
    print("\nProcessing Results:")
    print(f"   Total processed: {processed_count} files")
    print(f"   Skipped: {skipped_count} files")
    print(f"   Errors: {error_count} files")

    # Combine all
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = "output/features_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined features saved: {combined_path}")
        print(f"   Total samples: {len(combined)}")
        print(f"   Total features: {combined.shape[1]}")
        print(f"   Class distribution: {combined['label'].value_counts().to_dict()}")
        return True
    else:
        print("\nNo valid data processed, combined file could not be created.")
        return False

def process_single_file(edf_path):
    """Extract features from a single EDF file"""
    if not os.path.exists(edf_path):
        print(f"Error: File not found: {edf_path}")
        return False

    seizure_path = edf_path + ".seizures"
    output_file = os.path.basename(edf_path)
    output_csv = os.path.join(OUTPUT_DIR, f"feat_{output_file}.csv")

    print(f"Processing file: {edf_path}")

    try:
        # Segment and extract
        segments, labels, ch_names, fs = segment_eeg_file(edf_path, seizure_path)

        # Check if segment data is valid
        valid, reason = check_segments(segments, labels, ch_names, fs)
        if not valid:
            print(f"Invalid segment: {edf_path} - {reason}")
            return False

        # Extract features
        features_df = extract_features(segments, labels, ch_names, fs)

        # Add file information
        features_df["file"] = os.path.basename(edf_path)

        # Save
        features_df.to_csv(output_csv, index=False)
        print(f"Feature matrix saved: {output_csv}")
        print(f"   Shape: {features_df.shape}")
        print(f"   Class distribution: {features_df['label'].value_counts().to_dict()}")
        return True

    except Exception as e:
        print(f"Error: {edf_path} - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from EEG files")
    parser.add_argument("--file", type=str, help="Single EDF file to process (optional)")
    parser.add_argument("--patient", type=str, nargs='+', help="Patient IDs to process (chb01, chb02, ...)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir != OUTPUT_DIR:
        OUTPUT_DIR = args.output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Output directory: {OUTPUT_DIR}")

    # Process single file or patient list or entire dataset
    if args.file:
        success = process_single_file(args.file)
    else:
        success = process_files(args.patient)

    # Exit based on success status
    sys.exit(0 if success else 1)
