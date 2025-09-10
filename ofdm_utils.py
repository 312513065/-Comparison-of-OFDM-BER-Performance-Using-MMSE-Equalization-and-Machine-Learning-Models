import numpy as np
from scipy.interpolate import interp1d
from config import K, CP, pilotValue, P, mu

def generate_zc_sequence(u, N_ZC):
    """
    u: root index (must be relatively prime to N_ZC)
    N_ZC: length of Zadoff-Chu sequence
    """
    n = np.arange(N_ZC)
    zc_seq = np.exp(-1j * np.pi * u * n * (n + 1) / N_ZC)
    return zc_seq
# Subcarriers
allCarriers = np.arange(K)
pilotCarriers = np.hstack([allCarriers[::K // P], np.array([K - 1])])
print(len(pilotCarriers))
P = len(pilotCarriers)
dataCarriers = np.delete(allCarriers, pilotCarriers)
payloadBitsPerOFDM = len(dataCarriers) * mu
zc_pilot = generate_zc_sequence(u=1, N_ZC=len(pilotCarriers))
print("zc_pilot" , zc_pilot)


def ofdm_modulate(symbols):
    carriers = np.zeros(K, dtype=complex)
    carriers[pilotCarriers] = zc_pilot 
    carriers[dataCarriers] = symbols
    time_signal = np.fft.ifft(carriers)
    return np.hstack([time_signal[-CP:], time_signal])

def channel(signal, h, SNRdb):
    convolved = np.convolve(signal, h)
    signal_power = np.mean(abs(convolved)**2)
    sigma2 = signal_power * 10**(-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j*np.random.randn(*convolved.shape))
    return convolved + noise

def ofdm_demodulate(rx_signal):
    rx_signal = rx_signal[:K + CP]
    rx_noCP = rx_signal[CP:]
    return np.fft.fft(rx_noCP)

def equalize(rx_freq):
    Hest_at_pilots = rx_freq[pilotCarriers] / pilotValue
    Hest_abs = interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear', fill_value="extrapolate")(allCarriers)
    Hest_phase = interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear', fill_value="extrapolate")(allCarriers)
    Hest = Hest_abs * np.exp(1j * Hest_phase)
    return rx_freq / Hest

def generate_one_ofdm_frame(channelResponse, mapping_table ,SNRdb):
    bits = np.random.binomial(1, 0.5, (payloadBitsPerOFDM,))
    bits_SP = bits.reshape((len(dataCarriers), mu))
    tx_symbols = np.array([mapping_table[tuple(b)] for b in bits_SP])
    tx_signal = ofdm_modulate(tx_symbols)
    rx_signal = channel(tx_signal, channelResponse, SNRdb)
    rx_frames = ofdm_demodulate(rx_signal)
    rx_equal = equalize(rx_frames)
    rx_data = rx_equal[dataCarriers]
    return rx_equal, tx_symbols

def generate_dataset(mapping_table, num_samples=50000, channelResponse=np.array([1, 0, 0.3 + 0.3j]),  SNR_range=(5, 25)):
    X_list, Y_list = [], []
    pilot_mask = np.zeros(K)
    pilot_mask[pilotCarriers] = 1
    pilot_vals = np.zeros(K, dtype=complex)
    pilot_vals[pilotCarriers] = pilotValue

    for _ in range(num_samples):
        SNRdb = np.random.uniform(*SNR_range)
        X_complex, Y_complex = generate_one_ofdm_frame(channelResponse, mapping_table, SNRdb)
        X_feat = np.stack([X_complex.real, X_complex.imag, pilot_mask, pilot_vals.real , pilot_vals.imag], axis=1)
        Y_feat = np.stack([Y_complex.real, Y_complex.imag], axis=1)
        X_list.append(X_feat)
        Y_list.append(Y_feat)

    return np.array(X_list), np.array(Y_list), pilotCarriers, dataCarriers, allCarriers
