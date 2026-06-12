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
sdr = adi.Pluto(token='ZqFOcxlWwts') # create Pluto object
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

# first for 4 QAM
symbols, constellation = gen_rand_qam_symbols(len(idx_subs_data), M=4)
np.save("symbols_4", symbols)
# add pilot sequence
pilots, constellation = gen_rand_qam_symbols(len(idx_subs_pilots), M=4)
constellation = np.array(constellation)
np.save("constellation_4", constellation)
pilots = np.array(pilots)
np.save("pilots_4", pilots)

tx_signal = np.zeros(num_subcarriers, dtype=complex) # nulls implied
tx_signal[idx_subs_pilots] = pilots
tx_signal[idx_subs_data] = symbols
np.save("ofdm_symbol_4", tx_signal)

tx_signal_4 = np.fft.ifft(np.fft.ifftshift(tx_signal))
# add the Cyclic Prefix
L = num_subcarriers // 4
tx_signal_4 = np.concatenate([tx_signal_4[num_subcarriers - L:], tx_signal_4])
# calculate PAPR in dB
# average power
ap = np.sum(np.abs(tx_signal_4) ** 2) / len(tx_signal_4)
# peak power
pp = np.max(np.abs(tx_signal_4) ** 2)

print("average power: " + str(ap))
print("peak power: " + str(pp))

# Make another transmit signal for 16 QAM
symbols, constellation = gen_rand_qam_symbols(len(idx_subs_data), M=16)
np.save("symbols_16", symbols)
# add pilot sequence
pilots, constellation = gen_rand_qam_symbols(len(idx_subs_pilots), M=16)
constellation = np.array(constellation)
np.save("constellation_16", constellation)
pilots = np.array(pilots)
np.save("pilots_16", pilots)

tx_signal = np.zeros(num_subcarriers, dtype=complex) # nulls implied
tx_signal[idx_subs_pilots] = pilots
tx_signal[idx_subs_data] = symbols
np.save("ofdm_symbol_16", tx_signal)

tx_signal_16 = np.fft.ifft(np.fft.ifftshift(tx_signal))
# add the Cyclic Prefix
L = num_subcarriers // 4
tx_signal_16 = np.concatenate([tx_signal_16[num_subcarriers - L:], tx_signal_16])

# ---------------------------------------------------------------
# Transmit from Pluto!
# ---------------------------------------------------------------
# first for 4 qam
rx_array_4 = []
tx_signal_scaled = tx_signal_4 / np.max(np.abs(tx_signal_4)) * 2**14 # Pluto expects TX samples to be between -2^14 and 2^14 
np.save("lab6_tx_4", tx_signal_scaled)
# Try transmit gains from -40 to -10 dB, 20 trials each
for gain in range(-40, -5, 5):
    sdr.tx_hardwaregain_chan0 = gain
    trials = []
    for i in range(20):       
        sdr.tx(tx_signal_scaled) # will continuously transmit when cyclic buffer set to True
        # ---------------------------------------------------------------
        # Receive with Pluto!
        # ---------------------------------------------------------------
        sdr.rx_destroy_buffer() # reset receive data buffer to be safe
        for i in range(1): # clear buffer to be safe
            rx_data_ = sdr.rx() # toss them out
            
        rx_signal = sdr.rx() # capture raw samples from Pluto
        trials.append(rx_signal)
        # ---------------------------------------------------------------
        # Clean up buffers once done receiving.
        # ---------------------------------------------------------------
        sdr.tx_destroy_buffer() # reset transmit data buffer to be safe
        sdr.rx_destroy_buffer() # reset receive data buffer to be safe
    rx_array_4.append(trials)

np.save("rx_signal_arr_4", np.array(rx_array_4))

# Next do 16 qam
rx_array_16 = []
tx_signal_scaled = tx_signal_16 / np.max(np.abs(tx_signal_16)) * 2**14 # Pluto expects TX samples to be between -2^14 and 2^14 
np.save("lab6_tx_16", tx_signal_scaled)
# Try transmit gains from -40 to -10 dB, 20 trials each
for gain in range(-40, -5, 5):
    sdr.tx_hardwaregain_chan0 = gain
    trials = []
    for i in range(20):       
        sdr.tx(tx_signal_scaled) # will continuously transmit when cyclic buffer set to True
        # ---------------------------------------------------------------
        # Receive with Pluto!
        # ---------------------------------------------------------------
        sdr.rx_destroy_buffer() # reset receive data buffer to be safe
        for i in range(1): # clear buffer to be safe
            rx_data_ = sdr.rx() # toss them out
            
        rx_signal = sdr.rx() # capture raw samples from Pluto
        trials.append(rx_signal)
        # ---------------------------------------------------------------
        # Clean up buffers once done receiving.
        # ---------------------------------------------------------------
        sdr.tx_destroy_buffer() # reset transmit data buffer to be safe
        sdr.rx_destroy_buffer() # reset receive data buffer to be safe
    rx_array_16.append(trials)

np.save("rx_signal_arr_16", np.array(rx_array_16))



# perform cross correlation
rx_arr = np.load('rx_signal_arr_4.npy')
tx_data = np.load("lab6_tx_4.npy")
pilots = np.load("pilots_4.npy")
symbols = np.load("symbols_4.npy")
constellation = np.load("constellation_4.npy")

gains = [i for i in range(-40, -5, 5)]
ser_array_4 = []
for gain_index, gain in enumerate(gains):
    gain_ser = []
    for trial_index in range(20):
        rx_data = rx_arr[gain_index][trial_index]
        cc = np.correlate(rx_data, tx_data, "full")
        magnitudes = np.abs(cc)
        # extract the first peak which is the delay value
        full_max = np.argmax(magnitudes[0:8000]) # got this range from looking at the graph
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
        # frequency flat equalization
        equalized = ofdm_symbols / H_full
        # extract just the data symbols for SER compuation
        equalized = equalized[idx_subs_data]

        # get SER
        ser = symbol_error_rate(equalized, symbols, constellation)
        gain_ser.append(ser)
    ser_array_4.append(np.mean(gain_ser))

# repeat for 16 QAM
# perform cross correlation
rx_arr = np.load('rx_signal_arr_16.npy')
tx_data = np.load("lab6_tx_16.npy")
pilots = np.load("pilots_16.npy")
symbols = np.load("symbols_16.npy")
constellation = np.load("constellation_16.npy")

ser_array_16 = []
for gain_index, gain in enumerate(gains):
    gain_ser = []
    for trial_index in range(20):
        rx_data = rx_arr[gain_index][trial_index]
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
        # frequency flat equalization
        equalized = ofdm_symbols / H_full
        # extract just the data symbols for SER compuation
        equalized = equalized[idx_subs_data]

        # get SER
        ser = symbol_error_rate(equalized, symbols, constellation)
        gain_ser.append(ser)
    ser_array_16.append(np.mean(gain_ser))

# plot the two SER curves against the gains

x = np.arange(-40, -5, 5)

fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(x, ser_array_16,  'b-o', label='16-QAM')
ax.semilogy(x, ser_array_4, 'r-s', label='4-QAM')

ax.set_xlabel('Transmit Gain (dB)')
ax.set_ylabel('Symbol Error Rate (SER)')
ax.set_title('Average SER Across 20 Trials for 16-QAM and 4-QAM OFDM')
ax.set_xticks(x)
ax.legend()
ax.grid(True, which='both', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
