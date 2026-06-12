import numpy as np
import matplotlib.pyplot as plt
from ece230b import *

# for remotely interfacing with Pluto
from remoteRF.drivers.adalm_pluto import *

# ---------------------------------------------------------------
# Digital communication system parameters.
# ---------------------------------------------------------------
fs = 30e6     # baseband sampling rate (samples per second)
ts = 1 / fs  # baseband sampling period (seconds per sample)
sps = 10     # samples per data symbol
T = ts * sps # time between data symbols (seconds per symbol)

# ---------------------------------------------------------------
# Pluto system parameters.
# ---------------------------------------------------------------
sample_rate = fs                # sampling rate, between ~600e3 and 61e6
tx_carrier_freq_Hz = 915e6      # transmit carrier frequency, between 325 MHz to 3.8 GHz
rx_carrier_freq_Hz = 915e6      # receive carrier frequency, between 325 MHz to 3.8 GHz
tx_rf_bw_Hz = sample_rate * 1   # transmitter's RF bandwidth, between 200 kHz and 56 MHz
rx_rf_bw_Hz = sample_rate * 1   # receiver's RF bandwidth, between 200 kHz and 56 MHz
tx_gain_dB = -10                # transmit gain (in dB), beteween -89.75 to 0 dB with a resolution of 0.25 dB
rx_gain_dB = 30                 # receive gain (in dB), beteween 0 to 74.5 dB (only set if AGC is 'manual')
rx_agc_mode = 'manual'          # receiver's AGC mode: 'manual', 'slow_attack', or 'fast_attack'
rx_buffer_size = 100e3          # receiver's buffer size (in samples), length of data returned by sdr.rx()
tx_cyclic_buffer = True         # cyclic nature of transmitter's buffer (True -> continuously repeat transmission)

# ---------------------------------------------------------------
# Initialize Pluto object using issued token.
# ---------------------------------------------------------------
sdr = adi.Pluto(token='myPT5Rmaw-4') # create Pluto object
sdr.sample_rate = int(sample_rate)   # set baseband sampling rate of Pluto

# ---------------------------------------------------------------
# Setup Pluto's transmitter.
# ---------------------------------------------------------------
sdr.tx_destroy_buffer()                   # reset transmit data buffer to be safe
sdr.tx_rf_bandwidth = int(tx_rf_bw_Hz)    # set transmitter RF bandwidth
sdr.tx_lo = int(tx_carrier_freq_Hz)       # set carrier frequency for transmission
sdr.tx_hardwaregain_chan0 = tx_gain_dB    # set the transmit gain
sdr.tx_cyclic_buffer = tx_cyclic_buffer   # set the cyclic nature of the transmit buffer

# ---------------------------------------------------------------
# Setup Pluto's receiver.
# ---------------------------------------------------------------
sdr.rx_destroy_buffer()                   # reset receive data buffer to be safe
sdr.rx_lo = int(rx_carrier_freq_Hz)       # set carrier frequency for reception
sdr.rx_rf_bandwidth = int(sample_rate)    # set receiver RF bandwidth
sdr.rx_buffer_size = int(rx_buffer_size)  # set buffer size of receiver
sdr.gain_control_mode_chan0 = rx_agc_mode # set gain control mode
sdr.rx_hardwaregain_chan0 = rx_gain_dB    # set gain of receiver

# ---------------------------------------------------------------
# Create transmit signal.
# ---------------------------------------------------------------
num_subcarriers = 2**10
idx_subs_data, idx_subs_pilots, idx_subs_nulls = get_subcarrier_indices()
np.save("data_idx", idx_subs_data)
np.save("pilots_idx", idx_subs_pilots)
np.save("nulls_idx", idx_subs_nulls)

symbols, constellation = gen_rand_qam_symbols(len(idx_subs_data), M=4)
np.save("symbols", symbols)
# add pilot sequence
pilots, constellation = gen_rand_qam_symbols(len(idx_subs_pilots), M=4)
constellation = np.array(constellation)
np.save("constellation", constellation)
pilots = np.array(pilots)
np.save("pilots", pilots)

tx_signal = np.zeros(num_subcarriers, dtype=complex) # nulls implied
tx_signal[idx_subs_pilots] = pilots
tx_signal[idx_subs_data] = symbols
np.save("ofdm_symbol", tx_signal)

tx_signal = np.fft.ifft(np.fft.ifftshift(tx_signal))
# add the Cyclic Prefix
L = num_subcarriers // 4
tx_signal = np.concatenate([tx_signal[num_subcarriers - L:], tx_signal])
# calculate PAPR in dB
# average power
ap = np.sum(np.abs(tx_signal) ** 2) / len(tx_signal)
# peak power
pp = np.max(np.abs(tx_signal) ** 2)

print("average power: " + str(ap))
print("peak power: " + str(pp))

# ---------------------------------------------------------------
# Transmit from Pluto!
# ---------------------------------------------------------------
tx_signal_scaled = tx_signal / np.max(np.abs(tx_signal)) * 2**14 # Pluto expects TX samples to be between -2^14 and 2^14 
sdr.tx(tx_signal_scaled) # will continuously transmit when cyclic buffer set to True
np.save("lab6_tx.npy", tx_signal_scaled)
# ---------------------------------------------------------------
# Receive with Pluto!
# ---------------------------------------------------------------
sdr.rx_destroy_buffer() # reset receive data buffer to be safe
for i in range(1): # clear buffer to be safe
    rx_data_ = sdr.rx() # toss them out
    
rx_signal = sdr.rx() # capture raw samples from Pluto
np.save("lab6_rx.npy", rx_signal)
# ---------------------------------------------------------------
# Clean up buffers once done receiving.
# ---------------------------------------------------------------
sdr.tx_destroy_buffer() # reset transmit data buffer to be safe
sdr.rx_destroy_buffer() # reset receive data buffer to be safe

# ---------------------------------------------------------------
# Take FFT of received signal.
# ---------------------------------------------------------------
# perform cross correlation
rx_data = np.load('lab6_rx.npy')
tx_data = np.load("lab6_tx.npy")
pilots = np.load("pilots.npy")
constellation = np.load("constellation.npy")
cc = np.correlate(rx_data, tx_data, "full")
magnitudes = np.abs(cc)
# extract the first peak which is the delay value
full_max = np.argmax(magnitudes[0:3000]) # got this range from looking at the graph
spike_index = full_max - len(tx_data) + 1 # because we used a full correlation which doesn't truncate
received = rx_data[spike_index:(len(tx_data) + spike_index)] # our single copy of the signal

# toss out Cyclic Prefix
received = received[num_subcarriers // 4 :]
# get back to freq domain
ofdm_symbols = np.fft.fftshift(np.fft.fft(received))

# estimate channel
y_pilots = ofdm_symbols[idx_subs_pilots]
x_pilots = pilots
h_at_pilots = y_pilots/x_pilots
# Interpolate between pilots to get channel estimate
all_idx = np.arange(num_subcarriers)
# Interpolate real and imaginary separately
H_real = np.interp(all_idx, idx_subs_pilots, h_at_pilots.real)
H_imag = np.interp(all_idx, idx_subs_pilots, h_at_pilots.imag)

H_full = H_real + 1j * H_imag
plt.plot(all_idx, np.abs(H_full), label="Interpolated Channel Response")
plt.scatter(idx_subs_pilots, np.abs(h_at_pilots), color="red", label="Pilots")
plt.title("Interpolated Channel Response Across Subcarriers")
plt.xlabel("Subcarrier Indices")
plt.ylabel("Magnitude")
plt.legend()
plt.show()
# equalize
equalized = ofdm_symbols / H_full
# extract just the data symbols for SER compuation
equalized = equalized[idx_subs_data]
symbols = np.load("symbols.npy")

# get SER
ser = symbol_error_rate(equalized, symbols, constellation)

# plot the complex plane with the constellation
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(equalized.real, equalized.imag,
           color='steelblue', alpha=0.4, s=15, label='Received symbols')

# Ideal constellation points
ax.scatter(symbols.real, symbols.imag,
           color='red', alpha=0.4, s=15, label='Transmitted Symbols')

# Axes through origin
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

ax.set_xlabel('In-phase (I)')
ax.set_ylabel('Quadrature (Q)')
ax.set_title('OFDM Data Symbols After Equalization (SER: '+ str(ser) + ')')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
