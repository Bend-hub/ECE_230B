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
sdr_tx = adi.Pluto(token='UWnXBRGK7Yk') # create Pluto object

sdr_rx = adi.Pluto(token='DLHYylemv2U')

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

# add pilot sequence
pilots, constellation = gen_rand_qam_symbols(300, M=16)
symbols = pilots + symbols
pilots = np.array(pilots)
np.save("pilots", pilots)
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
#rx_pilot = received[:len(pilots) + 160]
#rx_pilot = np.convolve(rx_pilot, rrc)
matched_output = np.convolve(rx_signal, rrc)
# discard convolution overhead
matched_output = matched_output[200: len(matched_output) - 200]

# do symbol symchronization (fractional timing offset)
downsampled = symbol_synch_moe(matched_output,sps,upsample=8,plot=True)

if False:

    # extract the rx pilot, discard the convolution overhead
    # convolution added 200 but we downsampled everything by 10 after symbol synch
    rx_pilot = downsampled[20:20 + len(pilots)]

    # discard the convolution overhead and down sample
    rx_symbols = matched_output[20 + (len(pilots) * 10):20 + (len(pilots) * 10) + 10000]

    # estimate the channel
    N = len(pilots)
    X = np.zeros((N, 1), dtype=complex)
    Y = np.zeros((N, 1), dtype=complex)
    X[0:, 0] = pilots[: N]
    Y[0:, 0] = rx_pilot[: N]

    h_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    # equalize
    equalized = rx_symbols / h_hat

    # plot the complex plane with the constellation
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(equalized.real, equalized.imag,
            color='steelblue', alpha=0.4, s=15, label='Received symbols')

    # Ideal constellation points
    symbols = np.load("symbols.npy")
    ax.scatter(symbols.real, symbols.imag,
            color='red', alpha=0.4, s=15, label='Transmitted Symbols')

    # Axes through origin
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

    ax.set_xlabel('In-phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.set_title('Equalized Received 64-QAM Symbols vs Transmitted 64-QAM Symbols')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
