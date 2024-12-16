import numpy as np
from scipy.signal import blackman

def blackman_window(input_signal):
    fs = len(input_signal)  # Sampling frequency
    # Define window size and overlap
    window_size = int(0.15 * fs)  # 1.5 seconds window size
    overlap = int(0.05 * fs)  # 0.5 seconds overlap

    # Calculate the number of windows
    num_windows = (len(input_signal) - overlap) // (window_size - overlap)
    # print(f"{window_size}, {overlap}, {num_windows}, {len(input_signal)}")
    # Create an array to store the windowed segments
    windowed_segments = np.zeros((num_windows, window_size))
    # Apply the Blackman window and segment the signal
    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        window = blackman(window_size)  # Blackman window
        windowed_segments[i, :] = input_signal[start:end] * window

    return windowed_segments
