import numpy as np
import matplotlib.pyplot as plt
rx_data = np.load('lab3_rx.npy') # load

real = np.array([z.real for z in rx_data])

imaginary = np.array([z.imag for z in rx_data])


x_axis = np.arange(len(real)) / 10
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(x_axis, real)
ax1.set_title('Real Part of Received Signal')
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitudes')

ax2.plot(x_axis, imaginary)
ax2.set_title('Imaginary Part of Received Signal')
ax2.set_xlabel('Time')
ax2.set_ylabel('Amplitudes')

plt.tight_layout()
plt.show()
plt.close()