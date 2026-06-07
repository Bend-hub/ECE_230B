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
tx_gain_dB = -15                # transmit gain (in dB), beteween -89.75 to 0 dB with a resolution of 0.25 dB
rx_gain_dB = 30                 # receive gain (in dB), beteween 0 to 74.5 dB (only set if AGC is 'manual')
rx_agc_mode = 'manual'          # receiver's AGC mode: 'manual', 'slow_attack', or 'fast_attack'
rx_buffer_size = 5e5          # receiver's buffer size (in samples), length of data returned by sdr.rx()
tx_cyclic_buffer = True         # cyclic nature of transmitter's buffer (True -> continuously repeat transmission)

# ---------------------------------------------------------------
# Initialize Pluto object using issued token.
# ---------------------------------------------------------------
sdr_tx = adi.Pluto(token='EV20AHAPWyk') # create Pluto object

sdr_rx = adi.Pluto(token='_e0ODVZm-9U')

sdr_tx.sample_rate = int(sample_rate)   # set baseband sampling rate of Pluto
sdr_rx.sample_rate = int(sample_rate)

# ---------------------------------------------------------------
# Setup Pluto's transmitter.
# ---------------------------------------------------------------
sdr_tx.tx_destroy_buffer()                   # reset transmit data buffer to be safe
sdr_tx.tx_rf_bandwidth = int(tx_rf_bw_Hz)    # set transmitter RF bandwidth
sdr_tx.tx_lo = int(tx_carrier_freq_Hz)       # set carrier frequency for transmission
sdr_tx.tx_hardwaregain_chan0 = tx_gain_dB    # set the transmit gain
sdr_tx.tx_cyclic_buffer = tx_cyclic_buffer   # set the cyclic nature of the transmit buffer

# ---------------------------------------------------------------
# Setup Pluto's receiver.
# ---------------------------------------------------------------
sdr_rx.rx_destroy_buffer()                   # reset receive data buffer to be safe
sdr_rx.rx_lo = int(rx_carrier_freq_Hz)       # set carrier frequency for reception
sdr_rx.rx_rf_bandwidth = int(sample_rate)    # set receiver RF bandwidth
sdr_rx.rx_buffer_size = int(rx_buffer_size)  # set buffer size of receiver
sdr_rx.gain_control_mode_chan0 = rx_agc_mode # set gain control mode
sdr_rx.rx_hardwaregain_chan0 = rx_gain_dB    # set gain of receiver

# ---------------------------------------------------------------
# Create transmit signal.
# ---------------------------------------------------------------
N = 5000 # number of samples to transmit
beta = 1
span = 20 # from piazza
symbols, constellation = gen_rand_qam_symbols(N, M=16)
np.save("tx_symbols", symbols)
# add pilot sequence
pilots, constellation = gen_rand_qam_symbols(300, M=16)
symbols = pilots + symbols
pilots = np.array(pilots)
constellation = np.array(constellation)
np.save("pilots", pilots)
np.save("constellation", constellation)
# add zadoff chu sequences
stf = np.tile(zadoff_chu_sequence(19, 1), 16)
ltf = np.tile(zadoff_chu_sequence(937, 1), 2)
symbols = np.array(symbols)
symbols = np.concatenate([ltf, symbols])
symbols = np.concatenate([stf, symbols])
pulse_train = create_pulse_train(symbols, sps)
#pilots_train = create_pulse_train(pilots, sps)

real = [z.real for z in pulse_train]
imaginary = [z.imag for z in pulse_train]
rrc = get_rrc_pulse(beta, span, sps)

real_y_axis = np.convolve(rrc, real)
imag_y_axis = np.convolve(rrc, imaginary)

complex_array = real_y_axis + 1j * imag_y_axis
tx_signal_scaled = complex_array / np.max(np.abs(complex_array)) * 2**14
# save transmit signal too
np.save("lab4_tx.npy", tx_signal_scaled)

# ---------------------------------------------------------------
# Transmit from Pluto!
# ---------------------------------------------------------------
sdr_tx.tx(tx_signal_scaled)

# ---------------------------------------------------------------
# Receive with Pluto
# ---------------------------------------------------------------
sdr_rx.rx_destroy_buffer() # reset receive data buffer to be safe
for i in range(1): # clear buffer to be safe
    rx_data_ = sdr_rx.rx() # toss them out
rx_signal = sdr_rx.rx() # capture raw samples from Pluto

# ---------------------------------------------------------------
# save data to file
# ---------------------------------------------------------------
np.save("lab4_rx.npy", rx_signal)

# ---------------------------------------------------------------
# Clean up buffers once done receiving.
# ---------------------------------------------------------------
sdr_tx.tx_destroy_buffer() # reset transmit data buffer to be safe
sdr_rx.tx_destroy_buffer()
sdr_rx.rx_destroy_buffer() # reset receive data buffer to be safe
sdr_rx.rx_destroy_buffer()

# ---------------------------------------------------------------
# Plot real and imaginary axis
# ---------------------------------------------------------------

# do matched filtering
beta = 1
span = 20
sps = 10
rrc = get_rrc_pulse(beta, span, sps)
rx_signal = np.load("lab4_rx.npy")
matched_output = np.convolve(rx_signal, rrc)
# discard convolution overhead
matched_output = matched_output[200: len(matched_output) - 200]
# do symbol synchronization (fractional timing offset)
rx_symbols, offset = symbol_synch_moe(matched_output,sps,upsample=8,plot=True)
# do frame synchronization (integer timing offset)
corr_magn_arr = []
max_correlation = [0, 0]
i = 0
while (1874 + i) <= 9000:
    corr = np.correlate(rx_symbols[i: 937 + i], rx_symbols[937 + i: 1874 + i], "full")
    corr_magn = np.max(np.abs(corr)) # lookin for the peak
    if corr_magn > max_correlation[1]:
        max_correlation = [i, corr_magn]
    corr_magn_arr.append(corr_magn)
    i += 1

if True:
    x_axis = np.arange(0, 9000 - 1874 + 1)
    plt.plot(x_axis, corr_magn_arr, label="|Correlation|")
    plt.plot(max_correlation[0], max_correlation[1], 'ro', label="Peak")
    plt.xlabel("Sliding Window Start Index")
    plt.ylabel("Correlation Magnitdue")
    plt.title('Frame Synchronization')
    plt.grid(True)
    plt.legend()
    plt.show()
    

# SCHMIDL COX FRAME CORRELATION ACTS LIKE IT STARTS AT LTF CAUSE THAT'S WHAT WE'RE CORRELATING
max_correlation[0] = max_correlation[0] - 324 # RE ADJUST BACK TO INCLUDE THE STF AND CORRELATION OVERHEAD 304 + 20
# extract just one copy of the transmitted signal
# Discard transmitter convolution overhead, was 200 on both ends but we downsampled everything by 10
rx_symbols = rx_symbols[max_correlation[0] + 20: max_correlation[0] + 20 + 5000 + 300 + 2178]

if True:

    # perform coarse CFO estimation
    # extract the STFs and split them into 2 slightly offset combinations
    x = rx_symbols[19 * 0:(19 * 15)]
    y = rx_symbols[19 * 1:(19 * 16)]
    coarse_cfo_phase = np.angle(np.sum(np.conj(x) * y))
    coarse_cfo = coarse_cfo_phase / (2 * np.pi * 19 * sps)
    print("Coarse_CFO: " + str(coarse_cfo))
    print("Maximum Unambiguous Coarse CFO: " + str(1/(2 * 19 * sps)))
    # adjust symbols by coarse CFO before estimating fine CFO
    n = (np.arange(max_correlation[0] + offset, max_correlation[0] + offset + len(rx_symbols)) + 20) * 10
    correction = np.exp(-1j * 2 * np.pi * coarse_cfo * n)
    rx_symbols = rx_symbols * correction

    # extract the LTFs
    x = rx_symbols[(19 * 16):(19 * 16) + 937]
    y = rx_symbols[(19 * 16) + 937:(19 * 16) + 1874]
    fine_cfo_phase = np.angle(np.sum(np.conj(x) * y))
    fine_cfo = fine_cfo_phase / (2 * np.pi * 937 * sps)
    print("Fine_CFO: " + str(fine_cfo))
    print("Maximum Unambiguous Fine CFO: " + str(1/(2 * 937 * sps)))
    # finally adjust by fine CFO
    correction = np.exp(-1j * 2 * np.pi * fine_cfo * n)
    rx_symbols = rx_symbols * correction

# extract pilots
# 16 * 19 = 304
# 304 + 1874 = 2178
rx_pilot = rx_symbols[2178:2178 + 300]
pilots = np.load("pilots.npy")
tx_symbols = np.load("tx_symbols.npy")
constellation = np.load("constellation.npy")

# estimate the channel
N = len(pilots)
X = np.zeros((N, 1), dtype=complex)
Y = np.zeros((N, 1), dtype=complex)
X[0:, 0] = pilots[: N]
Y[0:, 0] = rx_pilot[: N]

h_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

# cut off the metadata
rx_symbols = rx_symbols[2178 + 300:2178 + 300 + 5000]

# equalize
equalized = rx_symbols / h_hat

# get SER
ser = symbol_error_rate(equalized[0], tx_symbols, constellation)

# plot the complex plane with the constellation
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(equalized.real, equalized.imag,
        color='steelblue', alpha=0.4, s=15, label='Received symbols')

# Ideal constellation points
ax.scatter(pilots.real, pilots.imag,
        color='red', alpha=0.4, s=15, label='Transmitted Symbols')

# Axes through origin
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

ax.set_xlabel('Real Component')
ax.set_ylabel('Imaginary Component')
ax.set_title('Data Symbols After Equalization (SER: '+ str(ser) + ')')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
