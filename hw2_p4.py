import numpy as np
import matplotlib.pyplot as plt

start = -4
stop = 4
samples = 801
sps = 100
B = 0.5 # rolloff factor

t = np.linspace(start, stop, samples)
f = np.fft.fftfreq(samples, (t[1] - t[0]))
sinc_pulse = np.sinc(t)
rcos_pulse = np.sinc(t/sps) * np.cos(np.pi*B*t/sps) / (1 - (2*B*t/sps)**2)

sinc_spec = np.abs(np.fft.fft(sinc_pulse))
rc_spec   = np.abs(np.fft.fft(rcos_pulse))

plt.plot(np.fft.fftshift(f), np.fft.fftshift(sinc_spec), label='Sinc')
plt.plot(np.fft.fftshift(f), np.fft.fftshift(rc_spec),   label=f'Raised Cosine B={B}')
plt.xlim(-2, 2)
plt.legend()
plt.xlabel('Frequency (normalized)')
plt.title('Bandwidth Comparison')
plt.show()
