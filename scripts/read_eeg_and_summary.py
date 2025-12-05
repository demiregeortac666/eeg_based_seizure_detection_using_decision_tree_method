# scripts/read_eeg_and_summary.py

import os
import numpy as np
import mne
import warnings
import re

def load_seizure_annotations(seizure_path, edf_path=None, verbose=True):
    """
    Load seizure intervals from both .seizures files and summary.txt

    Parameters
    ----------
    seizure_path : str
        Path to .seizures file
    edf_path : str, optional
        Path to EDF file, needed for reading from summary.txt
    verbose : bool
        Show/hide detailed output

    Returns
    -------
    list
        List of seizure intervals as (onset, offset) tuples
    """
    seizure_intervals = []

    # 1. Try reading from .seizures file
    try:
        if verbose:
            print(f"Reading seizure file: {seizure_path}")

        with open(seizure_path, "rb") as f:
            content = f.read().decode(errors="ignore")
            if verbose:
                print(f"File content: {content.strip()}")

            lines = content.splitlines()
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue

                # Check if it's in two-number format
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        onset, offset = float(parts[0]), float(parts[1])
                        seizure_intervals.append((onset, offset))
                        if verbose:
                            print(f"Found seizure interval: {onset}s - {offset}s")
                    except ValueError:
                        print(f"Invalid time value: {parts} in {seizure_path}")
    except FileNotFoundError:
        if verbose:
            print(f"Seizure file not found: {seizure_path}")

    # 2. If no seizure intervals found and edf_path provided, try reading from summary.txt
    if len(seizure_intervals) == 0 and edf_path is not None:
        try:
            # Get EDF filename
            edf_filename = os.path.basename(edf_path)
            patient_id = edf_filename.split('_')[0]  # e.g., "chb01"

            # Create summary.txt path
            summary_path = os.path.join(os.path.dirname(edf_path), f"{patient_id}-summary.txt")

            if os.path.exists(summary_path):
                if verbose:
                    print(f"Reading summary file for seizures: {summary_path}")

                with open(summary_path, "r", errors="ignore") as f:
                    content = f.read()

                    # Regex to match file and seizure information
                    file_pattern = re.compile(r"File Name: (.*?)\n.*?Number of Seizures in File: (\d+)(.*?)(?=File Name:|$)", re.DOTALL)
                    seizure_pattern = re.compile(r"Seizure Start Time: (\d+) seconds\nSeizure End Time: (\d+) seconds")

                    for match in file_pattern.finditer(content):
                        filename, seizure_count, seizure_info = match.groups()

                        # If this is the file we're looking for and has seizures
                        if filename == edf_filename and int(seizure_count) > 0:
                            if verbose:
                                print(f"Found {seizure_count} seizure(s) for {filename}")

                            # Find seizure start and end times
                            for sz_match in seizure_pattern.finditer(seizure_info):
                                onset, offset = int(sz_match.group(1)), int(sz_match.group(2))
                                seizure_intervals.append((onset, offset))
                                if verbose:
                                    print(f"Found seizure interval (from summary.txt): {onset}s - {offset}s")
        except Exception as e:
            print(f"Error processing summary file: {str(e)}")

    if len(seizure_intervals) > 0:
        if verbose:
            print(f"Loaded {len(seizure_intervals)} seizure interval(s)")
    else:
        if verbose:
            print(f"No seizure intervals found")

    return seizure_intervals

def is_seizure_window(start_t, end_t, intervals):
    """
    Check if the given time window overlaps with any seizure interval

    Parameters
    ----------
    start_t : float
        Window start time (seconds)
    end_t : float
        Window end time (seconds)
    intervals : list
        List of seizure intervals as (onset, offset) tuples

    Returns
    -------
    int
        1 if window overlaps with a seizure interval, 0 otherwise
    """
    # Perform interval checks
    window_duration = end_t - start_t
    overlap_threshold = 0.5  # At least 50% of window must be seizure

    for onset, offset in intervals:
        # Calculate overlap amount
        overlap_start = max(start_t, onset)
        overlap_end = min(end_t, offset)

        if overlap_end > overlap_start:  # If there's overlap
            overlap_duration = overlap_end - overlap_start
            overlap_ratio = overlap_duration / window_duration

            # If sufficient overlap, label window as seizure
            if overlap_ratio >= overlap_threshold:
                return 1

    return 0

def segment_eeg_file(edf_path, seizure_path,
                     window_sec=10, step_sec=5,
                     verbose=False):
    """
    Segment and label EEG file

    Parameters
    ----------
    edf_path : str
        Path to EEG file
    seizure_path : str
        Path to seizure annotation file
    window_sec : int
        Segment window length (seconds)
    step_sec : int
        Segment step length (seconds)
    verbose : bool
        Show/hide detailed output

    Returns
    -------
    segments : np.array
        Array of shape (N_windows, N_channels, N_samples)
    labels : np.array
        Array of shape (N_windows,) with 0/1 labels
    ch_names : list
        Channel names
    sfreq : float
        Sampling frequency
    """
    try:
        # Suppress channel name duplication warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                   message="Channel names are not unique")

            # Read EDF file
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

            # Select only EEG channels
            raw.pick_types(eeg=True)

            # Handle duplicate channel names
            ch_names = raw.info['ch_names']
            ch_name_counts = {}
            for i, name in enumerate(ch_names):
                if name in ch_name_counts:
                    ch_name_counts[name] += 1
                    # Give duplicate channel a unique name
                    new_name = f"{name}_{ch_name_counts[name]}"
                    raw.rename_channels({name: new_name})
                else:
                    ch_name_counts[name] = 0

            # Apply average reference
            raw.set_eeg_reference('average', projection=True)

            # Get data
            data = raw.get_data()
            ch_names = raw.info['ch_names']
            sfreq = raw.info['sfreq']
            total_samples = data.shape[1]
            total_duration = total_samples / sfreq

            if verbose:
                print(f"EEG data shape: {data.shape}")
                print(f"Total duration: {total_duration:.2f} seconds")

            # Load seizure annotations (from both .seizures and summary.txt)
            intervals = load_seizure_annotations(seizure_path, edf_path, verbose=verbose)

            # Calculate segment parameters
            win_samp = int(window_sec * sfreq)
            step_samp = int(step_sec * sfreq)

            # If total samples insufficient for segmentation, return empty
            if total_samples <= win_samp:
                print(f"File too short, cannot segment: {edf_path}")
                print(f"   Total samples: {total_samples}, Required: {win_samp}")
                return np.array([]), np.array([]), [], 0

            # Segment
            segments, labels = [], []
            num_labeled_segments = 0

            for start in range(0, total_samples - win_samp + 1, step_samp):
                end = start + win_samp
                seg = data[:, start:end]

                # Time values in seconds
                start_t = start / sfreq
                end_t = end / sfreq

                # Labeling
                lbl = is_seizure_window(start_t, end_t, intervals)
                if lbl == 1:
                    num_labeled_segments += 1

                segments.append(seg)
                labels.append(lbl)

            # If no segments created, return empty
            if len(segments) == 0:
                print(f"No segments created: {edf_path}")
                return np.array([]), np.array([]), [], 0

            # Label count check
            segments_array = np.array(segments)
            labels_array = np.array(labels)

            if verbose or num_labeled_segments > 0:
                print(f"Total {len(segments)} segments, {num_labeled_segments} seizure labels ({(num_labeled_segments/len(segments)*100):.2f}%)")

            return segments_array, labels_array, ch_names, sfreq

    except Exception as e:
        print(f"Error processing EEG file: {edf_path}")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return np.array([]), np.array([]), [], 0

# Test function - used during development
def test_seizure_loading(edf_path):
    """
    Test seizure loading process
    """
    seizure_path = edf_path + ".seizures"
    intervals = load_seizure_annotations(seizure_path, edf_path, verbose=True)
    return intervals
