import numpy as np
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
rx_buffer_size = 100e3          # receiver's buffer size (in samples), length of data returned by sdr.rx()
tx_cyclic_buffer = True         # cyclic nature of transmitter's buffer (True -> continuously repeat transmission)

# ---------------------------------------------------------------
# Initialize Pluto object using issued token.
# ---------------------------------------------------------------
sdr_tx = adi.Pluto(token='SEUDxYZgJos') # create Pluto object

sdr_rx = adi.Pluto(token='0603IsBKc-0')

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
N = 10000 # number of samples to transmit
t = np.arange(N) / sample_rate # time vector
tx_signal = 0.5*np.exp(2.0j*np.pi*100e3*t) # complex sinusoid at 100 kHz

# ---------------------------------------------------------------
# Transmit from Pluto!
# ---------------------------------------------------------------
tx_signal_scaled = tx_signal / np.max(np.abs(tx_signal)) * 2**14 # Pluto expects TX samples to be between -2^14 and 2^14 
sdr_tx.tx(tx_signal_scaled) # will continuously transmit when cyclic buffer set to True


# ---------------------------------------------------------------
# Receive with Pluto!
# ---------------------------------------------------------------
sdr_rx.rx_destroy_buffer() # reset receive data buffer to be safe
for i in range(1): # clear buffer to be safe
    rx_data_ = sdr_rx.rx() # toss them out
    

rx_signal = sdr_rx.rx() # capture raw samples from Pluto

# ---------------------------------------------------------------
# Clean up buffers once done receiving.
# ---------------------------------------------------------------
sdr_tx.tx_destroy_buffer() # reset transmit data buffer to be safe
sdr_rx.tx_destroy_buffer()
sdr_rx.rx_destroy_buffer() # reset receive data buffer to be safe
sdr_rx.rx_destroy_buffer()

# ---------------------------------------------------------------
# Take FFT of received signal.
# ---------------------------------------------------------------
padded_signal = np.pad(rx_signal, (len(rx_signal) * 3))
rx_fft = np.abs(np.fft.fftshift(np.fft.fft(padded_signal)))

peak_index = np.argwhere(rx_fft == max(rx_fft))     # unfortunately this is a 2d array
new_frequency = peak_index[0][0]/len(padded_signal) # result is how far along the x axis
new_frequency = new_frequency - 0.5                 # subtract the negative half to get positive frequency
new_frequency = new_frequency * 1000                # result is in MHz so convert to KHz
cfo = new_frequency - 100                           # calculate CFO
f = np.linspace(sample_rate/-2, sample_rate/2, len(rx_fft))

plt.figure()
plt.plot(f/1e3,rx_fft,color="black")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude")
plt.title('FFT of Received Signal with CFO of ' + str(cfo) + ' KHz')
plt.grid(True)
plt.show()
plt.savefig("lab1.pdf")