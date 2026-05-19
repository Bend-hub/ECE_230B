import numpy as np
import matplotlib.pyplot as plt
from ece230b import *

tx_data = np.load("tx_data_64.npy")
rx_data = np.load("rx_data_arr_64.npy")[2]

# perform cross correlation
cc = np.correlate(rx_data, tx_data, "full")
x_axis = np.arange(-len(tx_data) + 1, len(rx_data))
magnitudes = np.abs(cc) # this gets the magnitude of complex numbers
# extract the first peak which is the delay value
full_max = np.argmax(magnitudes[len(tx_data):25000]) + len(tx_data) # got this range from looking at the graph
spike_index = full_max - len(tx_data) + 1 # because we used a full correlation which doesn't truncate
print(len(tx_data))
print(spike_index)
plt.figure(num="lab3_3")
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("Cross Correlation Between Scaled Transmitted and Received Signals")
plt.plot(x_axis, magnitudes)
plt.show()
plt.close()

# get a singular copy of the received signal
received = rx_data[spike_index:(len(tx_data) + spike_index)]

# convolve with matched filter to maximize snr I guess
beta = 0.5
span = 8
sps = 10
rrc = get_rrc_pulse(beta, span, sps)
matched_output = np.convolve(received, rrc)
real = [z.real for z in matched_output]
imaginary = [z.imag for z in matched_output]
x_axis = np.arange(len(real))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(x_axis, real)
ax1.set_title('Real Values After Matched Filter Pulse Shaping')
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitudes')

ax2.plot(x_axis, imaginary)
ax2.set_title('Imaginary Values After Matched filter Pulse Shaping')
ax2.set_xlabel('Time')
ax2.set_ylabel('Amplitudes')

plt.tight_layout()
plt.show()
plt.close()


rc = np.convolve(rrc, rrc)
# normalize effective pulse shape to have unit energy
energy = np.sum(rc**2)
rc = rc / np.sqrt(energy)

# plot the effective pulse shape
plt.figure(num="lab3_4_1")
plt.title("Raised Cosine Derived from Convolution between two Root Raised Cosines, Unit Energy Normalized")
plt.plot(rc)
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.show()
plt.close()

real = [z.real for z in received]
imaginary = [z.imag for z in received]
x_axis = np.arange(len(tx_data))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(x_axis, real)
ax1.set_title('Pulsed Shaped In-Phase Values')
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitudes')

ax2.plot(x_axis, imaginary)
ax2.set_title('Pulse Shaped Quadrature Values')
ax2.set_xlabel('Time')
ax2.set_ylabel('Amplitudes')

plt.tight_layout()
plt.show()
plt.close()
