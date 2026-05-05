import numpy as np
import matplotlib.pyplot as plt
from ece230b import *
import matplotlib.pyplot as plt

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
rx_buffer_size = 1e6          # receiver's buffer size (in samples), length of data returned by sdr.rx()
tx_cyclic_buffer = True         # cyclic nature of transmitter's buffer (True -> continuously repeat transmission)

# ---------------------------------------------------------------
# Initialize Pluto object using issued token.
# ---------------------------------------------------------------
sdr = adi.Pluto(token='jbc7UvP43DU') # create Pluto object
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
N = 200 # number of samples to transmit
symbols, constellation = gen_rand_qam_symbols(N, M=4)
pulse_train = create_pulse_train(symbols, 10)

real = [z.real for z in pulse_train]
imaginary = [z.imag for z in pulse_train]

beta_list = [0.0, 0.25, 0.5, 0.75, 1.0]

rx_array = []
for b in beta_list:
    sdr.tx_destroy_buffer()
    span = 4
    sps = 10
    rc = get_rc_pulse(b, span, sps)
    real_y_axis = np.convolve(rc, real)
    imag_y_axis = np.convolve(rc, imaginary)
    real_y_axis = real_y_axis/max(rc)
    imag_y_axis = imag_y_axis/max(rc)
    # convert back to complex form for transmission
    complex_array = real_y_axis + 1j * imag_y_axis
    # scale the tx samples
    tx_signal_scaled = complex_array / np.max(np.abs(complex_array)) * 2**14
    # transmit from Pluto
    sdr.tx(tx_signal_scaled)
    # receive with Pluto
    sdr.rx_destroy_buffer() # reset receive data buffer to be safe
    for i in range(1): # clear buffer to be safe
        rx_data_ = sdr.rx() # toss them out
    rx_array.append(sdr.rx())
    sdr.tx_destroy_buffer() # reset transmit data buffer to be safe
    sdr.rx_destroy_buffer() # reset receive data buffer to be safe

# ---------------------------------------------------------------
# Take FFT of received signal.
# ---------------------------------------------------------------
frequency_arr = []
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
for i in range(len(beta_list)):
    frequency_arr.append(np.abs(np.fft.fftshift(np.fft.fft(rx_array[i]))))
    f = np.linspace(sample_rate/-2, sample_rate/2, len(frequency_arr[i]))
    ax1.plot(f/1e3, 10 * np.log10(np.abs(frequency_arr[i])**2), label="beta = " + str(beta_list[i]))
ax1.set_title('Power of Spectral Leakage in Increasing Order')
ax1.set_xlabel('Frequency (KHz)')
ax1.set_ylabel('Squared Magnitude in dB')
# now overlay it in reverse order
for i in range(len(frequency_arr) - 1, -1, -1):
    f = np.linspace(sample_rate/-2, sample_rate/2, len(frequency_arr[i]))
    ax2.plot(f/1e3, 10 * np.log10(np.abs(frequency_arr[i])**2), label="beta = " + str(beta_list[i]))
ax2.set_title('Power of Spectral Leakage in Decreasing Order')
ax2.set_xlabel('Frequency (KHz)')
ax2.set_ylabel('Squared Magnitude in dB')
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.show()
plt.close()
