from scipy.signal import cheby1, lfilter

def cheby(input_signal, fs):
    order = 1
    rp = 10
    cutoff_freq = 100
    b, a = cheby1(order, rp, cutoff_freq / (0.5 * fs), btype='low', analog=False)
    filtered_signal = lfilter(b, a, input_signal)

    return filtered_signal
