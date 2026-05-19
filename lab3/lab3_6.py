import numpy as np
import matplotlib.pyplot as plt
from ece230b import *

# for remotely interfacing with Pluto
from remoteRF.drivers.adalm_pluto import *

# ---------------------------------------------------------------
# Digital communication system parameters.
# ---------------------------------------------------------------
fs = 1e6     # baseband sampling rate (samples per second)
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
tx_gain_dB = -25                # transmit gain (in dB), beteween -89.75 to 0 dB with a resolution of 0.25 dB
rx_gain_dB = 40                 # receive gain (in dB), beteween 0 to 74.5 dB (only set if AGC is 'manual')
rx_agc_mode = 'manual'          # receiver's AGC mode: 'manual', 'slow_attack', or 'fast_attack'
rx_buffer_size = 5e5          # receiver's buffer size (in samples), length of data returned by sdr.rx()
tx_cyclic_buffer = True         # cyclic nature of transmitter's buffer (True -> continuously repeat transmission)

# ---------------------------------------------------------------
# Initialize Pluto object using issued token.
# ---------------------------------------------------------------
sdr = adi.Pluto(token='ZfJRzHhLwv4') # create Pluto object
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
# Create transmit signal. FOR 64 QAM
# ---------------------------------------------------------------
N = 1000 # number of samples to transmit
beta = 0.5
span = 8 # arbitrary, recycled from lab2
symbols, constellation = gen_rand_qam_symbols(N, M=64)
tx_symbols_64 = symbols
np.save("tx_symbols_64.npy", tx_symbols_64)
# add pilot sequence
pilots64, constellation64 = gen_rand_qam_symbols(50, M=64)
symbols = pilots64 + symbols
pilots64 = np.array(pilots64)
constellation64 = np.array(constellation64)
np.save("pilots64.npy", pilots64)
np.save("constellation64.npy", constellation64)
pulse_train = create_pulse_train(symbols, sps)

real = [z.real for z in pulse_train]
imaginary = [z.imag for z in pulse_train]
rrc = get_rrc_pulse(beta, span, sps)

real_y_axis = np.convolve(rrc, real)
imag_y_axis = np.convolve(rrc, imaginary)

complex_array = real_y_axis + 1j * imag_y_axis
tx_signal_scaled = complex_array / np.max(np.abs(complex_array)) * 2**14
tx_data_64 = tx_signal_scaled
np.save("tx_data_64.npy", tx_data_64)

# ---------------------------------------------------------------
# Transmit from Pluto!
# ---------------------------------------------------------------
tx_gain_dB = -50
rx_data_arr_64 = []
for i in range(0, 30, 5):
    # ---------------------------------------------------------------
    # Setup Pluto's transmitter.
    # ---------------------------------------------------------------
    sdr.tx_destroy_buffer()                   # reset transmit data buffer to be safe
    sdr.tx_rf_bandwidth = int(tx_rf_bw_Hz)    # set transmitter RF bandwidth
    sdr.tx_lo = int(tx_carrier_freq_Hz)       # set carrier frequency for transmission
    sdr.tx_hardwaregain_chan0 = tx_gain_dB    # set the transmit gain
    sdr.tx_cyclic_buffer = tx_cyclic_buffer   # set the cyclic nature of the transmit buffer
    
    tx_gain_dB = tx_gain_dB + i
    sdr.tx(tx_signal_scaled)

    sdr.rx_destroy_buffer() # reset receive data buffer to be safe
    for i in range(1): # clear buffer to be safe
        rx_data_ = sdr.rx() # toss them out
    rx_data_arr_64.append(sdr.rx()) # capture raw samples from Pluto

    # ---------------------------------------------------------------
    # Clean up buffers once done receiving.
    # ---------------------------------------------------------------
    sdr.tx_destroy_buffer() # reset transmit data buffer to be safe
    sdr.rx_destroy_buffer() # reset receive data buffer to be safe

np.save("rx_data_arr_64.npy", np.array(rx_data_arr_64))

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
# NOW FOR 256
# ---------------------------------------------------------------
N = 1000 # number of samples to transmit
beta = 0.5
span = 8 # arbitrary, recycled from lab2
symbols, constellation = gen_rand_qam_symbols(N, M=256)
tx_symbols_256 = symbols
np.save("tx_symbols_256.npy", tx_symbols_256)
# add pilot sequence
pilots256, constellation256 = gen_rand_qam_symbols(50, M=256)
symbols = pilots256 + symbols
pilots256 = np.array(pilots256)
np.save("pilots256.npy", pilots256)
constellation256 = np.array(constellation256)
np.save("constellation256.npy", constellation256)

pulse_train = create_pulse_train(symbols, sps)

real = [z.real for z in pulse_train]
imaginary = [z.imag for z in pulse_train]
rrc = get_rrc_pulse(beta, span, sps)

real_y_axis = np.convolve(rrc, real)
imag_y_axis = np.convolve(rrc, imaginary)

complex_array = real_y_axis + 1j * imag_y_axis
tx_signal_scaled = complex_array / np.max(np.abs(complex_array)) * 2**14
tx_data_256 = tx_signal_scaled
np.save("tx_data_256.npy", tx_data_256)


# ---------------------------------------------------------------
# Transmit from Pluto!
# ---------------------------------------------------------------
tx_gain_dB = -50
rx_data_arr_256 = []
for i in range(0, 30, 5):
    # ---------------------------------------------------------------
    # Setup Pluto's transmitter.
    # ---------------------------------------------------------------
    sdr.tx_destroy_buffer()                   # reset transmit data buffer to be safe
    sdr.tx_rf_bandwidth = int(tx_rf_bw_Hz)    # set transmitter RF bandwidth
    sdr.tx_lo = int(tx_carrier_freq_Hz)       # set carrier frequency for transmission
    sdr.tx_hardwaregain_chan0 = tx_gain_dB    # set the transmit gain
    sdr.tx_cyclic_buffer = tx_cyclic_buffer   # set the cyclic nature of the transmit buffer
    
    tx_gain_dB = tx_gain_dB + i
    sdr.tx(tx_signal_scaled)

    sdr.rx_destroy_buffer() # reset receive data buffer to be safe
    for i in range(1): # clear buffer to be safe
        rx_data_ = sdr.rx() # toss them out
    rx_data_arr_256.append(sdr.rx()) # capture raw samples from Pluto

    # ---------------------------------------------------------------
    # Clean up buffers once done receiving.
    # ---------------------------------------------------------------
    sdr.tx_destroy_buffer() # reset transmit data buffer to be safe
    sdr.rx_destroy_buffer() # reset receive data buffer to be safe

np.save("rx_data_arr_256.npy", np.array(rx_data_arr_256))

# ---------------------------------------------------------------
# Plot error rates
# ---------------------------------------------------------------

# loop for 64 QAM
ser_64 = []
tx_data = tx_data_64
symbols = tx_symbols_64
pilots = pilots64
for rx_data in rx_data_arr_64:
    # perform cross correlation
    cc = np.correlate(rx_data, tx_data, "full")
    magnitudes = np.abs(cc)
    # extract the first peak which is the delay value
    full_max = np.argmax(magnitudes[0:25000]) # got this range from looking at the graph
    spike_index = full_max - len(tx_data) + 1 # because we used a full correlation which doesn't truncate
    received = rx_data[spike_index:(len(tx_data) + spike_index)] # our single copy of the signal

    # do matched filtering
    beta = 0.5
    span = 8
    sps = 10
    rrc = get_rrc_pulse(beta, span, sps)
    #rx_pilot = received[:len(pilots) + 160]
    #rx_pilot = np.convolve(rx_pilot, rrc)
    matched_output = np.convolve(received, rrc)

    # extract the rx pilot, discard the convolution overhead and also down sample
    rx_pilot = matched_output[160:160 + (len(pilots) * 10): 10]

    # discard the convolution overhead and down sample
    rx_symbols = matched_output[160 + (len(pilots) * 10):160 + (len(pilots) * 10) + 10000:10]

    # estimate the channel
    N = len(pilots)
    X = np.zeros((N, 1), dtype=complex)
    Y = np.zeros((N, 1), dtype=complex)
    X[0:, 0] = pilots[: N]
    Y[0:, 0] = rx_pilot[: N]

    h_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    # equalize
    equalized = rx_symbols / h_hat
    ser_64.append(symbol_error_rate(equalized[0], symbols, constellation64))

# do it all again for 256 qam
ser_256 = []
tx_data = tx_data_256
symbols = tx_symbols_256
pilots = pilots256
for rx_data in rx_data_arr_256:
    # perform cross correlation
    cc = np.correlate(rx_data, tx_data, "full")
    magnitudes = np.abs(cc)
    # extract the first peak which is the delay value
    full_max = np.argmax(magnitudes[0:25000]) # got this range from looking at the graph
    spike_index = full_max - len(tx_data) + 1 # because we used a full correlation which doesn't truncate
    received = rx_data[spike_index:(len(tx_data) + spike_index)] # our single copy of the signal

    # do matched filtering
    beta = 0.5
    span = 8
    sps = 10
    rrc = get_rrc_pulse(beta, span, sps)
    #rx_pilot = received[:len(pilots) + 160]
    #rx_pilot = np.convolve(rx_pilot, rrc)
    matched_output = np.convolve(received, rrc)

    # extract the rx pilot, discard the convolution overhead and also down sample
    rx_pilot = matched_output[160:160 + (len(pilots) * 10): 10]

    # discard the convolution overhead and down sample
    rx_symbols = matched_output[160 + (len(pilots) * 10):160 + (len(pilots) * 10) + 10000:10]

    # estimate the channel
    N = len(pilots)
    X = np.zeros((N, 1), dtype=complex)
    Y = np.zeros((N, 1), dtype=complex)
    X[0:, 0] = pilots[: N]
    Y[0:, 0] = rx_pilot[: N]

    h_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    # equalize
    equalized = rx_symbols / h_hat
    ser_256.append(symbol_error_rate(equalized[0], symbols, constellation256))

x = np.arange(-50, -20, 5)

fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(x, ser_64,  'b-o', label='64-QAM')
ax.semilogy(x, ser_256, 'r-s', label='256-QAM')

ax.set_xlabel('Transmit Gain (dB)')
ax.set_ylabel('Symbol Error Rate (SER)')
ax.set_title('SER vs Transmit Gain for 64-QAM and 256-QAM')
ax.set_xticks(x)
ax.legend()
ax.grid(True, which='both', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()