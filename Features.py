import numpy as np
from scipy.stats import skew, kurtosis, kstat
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

def calculate_skewness(signal):
    return skew(signal)

def calculate_kurtosis(signal):
    return kurtosis(signal)

def comulant_1(x):
    return kstat(x, n=1)

def comulant_2(x):
    return kstat(x, n=2)

def comulant_3(x):
    return kstat(x, n=3)

def parseval(x):
    time_domain_energy = np.sum(x**2)
    spectrum = fft(x)
    frequency_domain_energy = np.sum(np.abs(spectrum)**2)
    parseval_energy = time_domain_energy - frequency_domain_energy
    return parseval_energy

def spec_flatness(x):
    spectrum = np.abs(fft(x))
    geometric_mean = np.exp(np.mean(np.log(spectrum[spectrum > 0])))
    arithmetic_mean = np.mean(spectrum)
    spectral_flatness = geometric_mean / arithmetic_mean
    return spectral_flatness

def harmonics(x, fs):
    min_peak_height = 0.1
    min_peak_distance = 30
    spectrum = np.abs(fft(x))
    freqs = fftfreq(len(x), 1/fs)
    peaks, _ = find_peaks(spectrum, height=min_peak_height, distance=min_peak_distance)
    harmonic = freqs[peaks]
    return harmonic

def speed(signal):
    signal = np.array(signal)
    acceleration_magnitude = np.linalg.norm(signal, axis=1)
    acceleration_magnitude = np.abs(acceleration_magnitude - 9.81)
    delta_t = 1/50
    speed = np.cumsum(acceleration_magnitude) * delta_t
    return speed

def heading_direction(acc_x, acc_y, acc_z, mag_x, mag_y, mag_z, gyr_x, gyr_y, gyr_z):
    phi = np.arctan2(acc_y, acc_z)
    theta = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2))
    M_y = mag_x * np.sin(phi) * np.sin(theta) + mag_y * np.cos(phi) - mag_z * np.sin(phi) * np.cos(theta)
    M_x = mag_x * np.cos(theta) + mag_z * np.sin(theta)
    psi = np.arctan2(M_y, M_x)
    phi_star = gyr_x + gyr_y * np.sin(phi) * np.tan(theta) + gyr_z * np.cos(phi) * np.tan(theta)
    theta_star = gyr_y * np.cos(phi) - gyr_z * np.sin(phi)
    psi_star = gyr_y * (np.sin(phi) / np.cos(theta)) + gyr_z * (np.cos(phi) / np.cos(theta))
    G_Q = np.array([
        [np.cos(phi_star/2)*np.cos(theta_star/2)*np.cos(psi_star/2) + np.sin(phi_star/2)*np.sin(theta_star/2)*np.sin(psi_star/2)],
        [np.sin(phi_star/2)*np.cos(theta_star/2)*np.cos(psi_star/2) - np.cos(phi_star/2)*np.sin(theta_star/2)*np.sin(psi_star/2)],
        [np.cos(phi_star/2)*np.sin(theta_star/2)*np.cos(psi_star/2) + np.sin(phi_star/2)*np.cos(theta_star/2)*np.sin(psi_star/2)],
        [np.cos(phi_star/2)*np.cos(theta_star/2)*np.sin(psi_star/2) - np.sin(phi_star/2)*np.sin(theta_star/2)*np.cos(psi_star/2)]
    ])
    R = 0.5 * np.array([
        [0, -gyr_x, -gyr_y],
        [gyr_x, 0, -gyr_z],
        [gyr_y, gyr_z, 0]
    ]).dot(G_Q)
    G_x = 2 * (R[0, 0]*R[0, 2] + R[0, 1]*R[1, 2])
    G_y = 1 - 2 * (R[1, 2]**2 + R[0, 2]**2)
    heading_gyro = np.arctan2(G_x, G_y)
    heading_magnetometer = np.arctan2(M_y, M_x)
    final_heading_direction = np.degrees(np.mean([heading_gyro, heading_magnetometer]))
    return final_heading_direction

def stay_duration(speed, threshold=0.1):
    stationary_points = speed < threshold
    stationary_samples = np.sum(stationary_points)
    stay_duration = stationary_samples / 50
    return stay_duration

def pitch(signal):
    n = len(signal)
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result[:n // 2])
    frequencies = np.fft.fftfreq(n, d=1 / 50)[:n // 2]
    peaks, _ = find_peaks(fft_magnitude, height=np.max(fft_magnitude) * 0.1)
    if peaks.size > 0:
        fundamental_frequency = frequencies[peaks[0]]
        return fundamental_frequency
    else:
        return None

def harmonics_ratio(x, fs=50):
    spectrum = np.abs(fft(x)) ** 2
    freqs = fftfreq(len(x), 1 / fs)
    total_energy = np.sum(spectrum)
    harmonic_freqs = harmonics(x, fs)
    harmonic_indices = np.isin(freqs, harmonic_freqs)
    harmonic_energy = np.sum(spectrum[harmonic_indices])
    ratio = harmonic_energy / total_energy if total_energy > 0 else 0
    return ratio

def spectral_flux(signal, frame_size=1024, hop_size=512):
    num_frames = int((len(signal) - frame_size) / hop_size) + 1
    flux = np.zeros(num_frames)
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    for i in range(num_frames - 1):
        start = i * hop_size
        frame1 = signal[start:start + frame_size]
        frame2 = signal[start + hop_size:start + hop_size + frame_size]
        window = np.hamming(frame_size)
        frame1 = frame1 * window
        frame2 = frame2 * window
        spectrum1 = np.abs(fft(frame1))[:frame_size // 2]
        spectrum2 = np.abs(fft(frame2))[:frame_size // 2]
        diff = spectrum2 - spectrum1
        flux[i] = np.sum(diff[diff > 0])
    return flux
































