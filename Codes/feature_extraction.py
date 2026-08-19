import numpy as np
import pandas as pd
import os
import time
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import welch
from itertools import groupby
from operator import itemgetter

# --- Configurations ---
ROOT_DIR = 'Dataset'
OUTPUT_FILE = '1sec50overSeg.csv'
FS = 50  # Frequency of 50Hz

# --- Sliding Window Configurations ---
WINDOW_SEC = 1.0      # Window size in seconds
OVERLAP_PERC = 0.50   # Overlap (0% = no overlap)

# --- Mathematical auxiliary functions ---
def root_mean_square(x): return np.sqrt(np.mean(x**2))
def zero_crossings(x): return ((x[:-1] * x[1:]) < 0).sum()
def waveform_length(x): return np.sum(np.abs(np.diff(x)))
def sign_slope_changes(x): return np.sum(np.diff(np.sign(np.diff(x))) != 0)
def willison_amplitude(x, threshold=0.01): return np.sum(np.abs(np.diff(x)) > threshold)
def mean_absolute_value(x): return np.mean(np.abs(x))
def diff_abs_std(x): return np.std(np.abs(np.diff(x)))
def log_detector(x): return np.exp(np.mean(np.log(np.abs(x) + 1e-6)))

def sample_entropy(x, m=2, r=0.2):
    if len(x) == 0: return 0
    x = np.array(x)
    N = len(x)
    r *= np.std(x)
    
    if N > 2000: x = x[:2000]
    N = len(x)

    def _phi(m):
        x_m = np.array([x[i:i + m] for i in range(N - m + 1)])
        if len(x_m) == 0: return 0
        C = np.sum([np.sum(np.all(np.abs(x_m - xm) <= r, axis=1)) - 1 for xm in x_m])
        return C / ((N - m + 1) * (N - m))
    
    phi_m1 = _phi(m + 1)
    phi_m = _phi(m)
    
    if phi_m == 0 or phi_m1 == 0: return 0
    return -np.log(phi_m1 / phi_m + 1e-10)

# --- Spectral Function ---
def spectral_features(signal, fs=50):
    n_samples = len(signal)
    if n_samples < 2: return 0, 0, 0, 0, 0
    nperseg = min(n_samples, 256)
    
    try:
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        if len(freqs) > 1:
            df = freqs[1] - freqs[0]
        else:
            df = 1.0
        total_power = np.sum(psd) * df
        if total_power <= 0: return 0, 0, 0, 0, 0
        
        mean_freq = np.sum(freqs * psd) / np.sum(psd)
        median_freq = freqs[np.searchsorted(np.cumsum(psd), np.sum(psd) / 2)]
        peak_freq = freqs[np.argmax(psd)]
        spectral_entropy = entropy(psd)
        return mean_freq, median_freq, peak_freq, spectral_entropy, total_power
    except Exception:
        return 0, 0, 0, 0, 0

def quaternion_to_euler(w, x, y, z):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

# --- NEW FUNCTION: Muscle Activation Detection ---
def detect_muscle_activation(df, fs, window_ms=50, threshold_ratio=0.10, min_duration_ms=150):
    """
    Detects the muscle activation segment based on the average energy of the EMG channels.
    """
    emg_cols = [f'emg{i}' for i in range(1, 9)]
    present_cols = [c for c in emg_cols if c in df.columns]
    
    if not present_cols:
        return 0, len(df)

    window_samples = max(1, int(fs * window_ms / 1000))
    min_samples = int(fs * min_duration_ms / 1000)

    # 1. Energy
    squared_df = df[present_cols] ** 2
    energy_per_channel = squared_df.rolling(window=window_samples, min_periods=1, center=True).mean()
    mean_energy = energy_per_channel.mean(axis=1).fillna(0)

    # 2. Threshold
    max_energy = mean_energy.max()
    threshold = threshold_ratio * max_energy

    # 3. Identify Activation
    is_active = mean_energy > threshold
    active_indices = np.where(is_active)[0]
    
    if len(active_indices) == 0:
        return 0, len(df)

    # 4. Group and Validate Duration
    valid_ranges = []
    for k, g in groupby(enumerate(active_indices), lambda ix: ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        if len(group) == 0: continue
        
        start, end = group[0], group[-1]
        length = end - start + 1
        
        if length >= min_samples:
            valid_ranges.append((start, end))
            
    if not valid_ranges:
        return 0, len(df)

    # Returns the largest continuous segment
    longest_range = max(valid_ranges, key=lambda r: r[1] - r[0])
    return longest_range[0], longest_range[1] + 1

# --- Extraction Function (Window) ---
def extract_features_from_window(df_window, fs=50):
    features = {}
    duration = len(df_window) / fs
    if duration == 0: duration = 1

    # EMG
    for i in range(1, 9):
        col_name = f'emg{i}'
        if col_name in df_window.columns:
            emg = df_window[col_name].values.astype(float)
            prefix = f'{col_name}_'
            features.update({
                prefix + 'rms': root_mean_square(emg),
                prefix + 'mav': mean_absolute_value(emg),
                prefix + 'zc': zero_crossings(emg) / duration, 
                prefix + 'wl': waveform_length(emg) / duration,
                prefix + 'ssc': sign_slope_changes(emg) / duration,
                prefix + 'willison': willison_amplitude(emg) / duration,
                prefix + 'diff_std': diff_abs_std(emg),
                prefix + 'var': np.var(emg),
                prefix + 'logdet': log_detector(emg),
                prefix + 'entropy': sample_entropy(emg),
            })
            mf, mdf, pf, ent, tp = spectral_features(emg, fs)
            features.update({
                prefix + 'mean_freq': mf, prefix + 'median_freq': mdf, prefix + 'peak_freq': pf,
                prefix + 'spectral_entropy': ent, prefix + 'total_power': tp
            })

    # IMU
    for sensor in ['acc', 'gyro']:
        for axis in ['x', 'y', 'z']:
            col_name = f'{sensor}_{axis}'
            if col_name in df_window.columns:
                data = df_window[col_name].values.astype(float)
                prefix = f'{col_name}_'
                features.update({
                    prefix + 'mean': np.mean(data), prefix + 'std': np.std(data), prefix + 'rms': root_mean_square(data),
                    prefix + 'skew': skew(data), prefix + 'kurt': kurtosis(data),
                    prefix + 'zcr': zero_crossings(data) / duration, prefix + 'wl': waveform_length(data) / duration,
                    prefix + 'energy': np.sum(data**2) / duration,
                })
                mf, mdf, pf, ent, tp = spectral_features(data, fs)
                features.update({
                    prefix + 'mean_freq': mf, prefix + 'median_freq': mdf, prefix + 'peak_freq': pf,
                    prefix + 'spectral_entropy': ent, prefix + 'total_power': tp
                })

    # Quaternion
    if 'quat_w' in df_window.columns:
        qw, qx, qy, qz = df_window['quat_w'].values, df_window['quat_x'].values, df_window['quat_y'].values, df_window['quat_z'].values
        roll, pitch, yaw = quaternion_to_euler(qw.mean(), qx.mean(), qy.mean(), qz.mean())
        features.update({
            'quat_mean_w': np.mean(qw), 'quat_mean_x': np.mean(qx), 'quat_mean_y': np.mean(qy), 'quat_mean_z': np.mean(qz),
            'quat_var': np.var([qw, qx, qy, qz]), 'quat_norm_mean': np.mean(np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
            'euler_roll': roll, 'euler_pitch': pitch, 'euler_yaw': yaw,
        })

    return features

# --- Main Processing ---
def process_data_folder():
    all_features = []
    
    column_mapping = {
        'EMG_0': 'emg1', 'EMG_1': 'emg2', 'EMG_2': 'emg3', 'EMG_3': 'emg4',
        'EMG_4': 'emg5', 'EMG_5': 'emg6', 'EMG_6': 'emg7', 'EMG_7': 'emg8',
        'acc_0': 'acc_x', 'acc_1': 'acc_y', 'acc_2': 'acc_z',
        'gyro_0': 'gyro_x', 'gyro_1': 'gyro_y', 'gyro_2': 'gyro_z',
        'quat_0': 'quat_w', 'quat_1': 'quat_x', 'quat_2': 'quat_y', 'quat_3': 'quat_z'
    }

    window_samples = int(WINDOW_SEC * FS)
    step_samples = int(window_samples * (1 - OVERLAP_PERC))
    
    print("--- Configuration ---")
    print(f"Window: {WINDOW_SEC}s, Overlap: {OVERLAP_PERC*100}%")
    
    total_files = sum([len(files) for r, d, files in os.walk(ROOT_DIR) if any(f.endswith('.csv') for f in files)])
    print(f"Total files: {total_files}")
    
    processed_count = 0
    start_time = time.time()
    
    try:
        for root, dirs, files in os.walk(ROOT_DIR):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        df = pd.read_csv(file_path)
                        df.rename(columns=column_mapping, inplace=True)
                        if 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

                        # Normalization
                        if 'quat_w' in df.columns and df['quat_w'].abs().max() > 2:
                            for col in ['quat_w', 'quat_x', 'quat_y', 'quat_z']: df[col] = df[col] / 16384.0
                            for col in ['acc_x', 'acc_y', 'acc_z']: df[col] = df[col] / 2048.0
                            for col in ['gyro_x', 'gyro_y', 'gyro_z']: df[col] = df[col] / 16.0

                        # --- 1. MUSCLE ACTIVATION DETECTION ---
                        act_start, act_end = detect_muscle_activation(df, FS)
                        
                        # Slices the dataframe to process only the active part
                        df_active = df.iloc[act_start:act_end].reset_index(drop=True)
                        
                        # --- 2. SLIDING WINDOW (in the active segment) ---
                        num_samples = len(df_active)
                        
                        if num_samples == 0:
                            # Fallback if nothing is detected (uses original)
                            df_active = df
                            num_samples = len(df)

                        if num_samples < window_samples:
                            range_starts = [0]
                            actual_window_size = num_samples
                        else:
                            range_starts = range(0, num_samples - window_samples + 1, step_samples)
                            actual_window_size = window_samples

                        filename_no_ext = file.replace('.csv', '')
                        parts = filename_no_ext.split('_')
                        class_label = filename_no_ext[:5] 
                        volunteer_id = parts[-2] if len(parts) >= 2 else 'Unknown'

                        window_idx = 0
                        for start in range_starts:
                            end = start + actual_window_size
                            df_window = df_active.iloc[start:end] # Uses df_active here!
                            
                            features = extract_features_from_window(df_window, fs=FS)
                            
                            features['class'] = class_label
                            features['Volunteer'] = volunteer_id
                            features['filename'] = file
                            features['win_idx'] = window_idx
                            features['win_start_sec'] = start / FS
                            features['activation_start_idx'] = act_start # Extra info
                            
                            all_features.append(features)
                            window_idx += 1
                        
                        processed_count += 1
                        elapsed_time = time.time() - start_time
                        files_per_min = (processed_count / elapsed_time) * 60 if elapsed_time > 0 else 0
                        
                        print(f"\rProgress: {processed_count}/{total_files} | Vel: {files_per_min:.1f} files/min | Windows: {window_idx}", end="", flush=True)

                    except Exception as e:
                        print(f"\n Error in {file}: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted!")
    
    finally:
        print("\n")
        if all_features:
            output_df = pd.DataFrame(all_features)
            first_cols = ['class', 'Volunteer', 'filename', 'win_idx', 'win_start_sec', 'activation_start_idx']
            cols = first_cols + [c for c in output_df.columns if c not in first_cols]
            output_df = output_df[cols]
            output_df.to_csv(OUTPUT_FILE, index=False)
            print(f' Saved in: {OUTPUT_FILE}')
        else:
            print("No data processed.")

if __name__ == "__main__":
    process_data_folder()