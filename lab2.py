from ece230b import *
import matplotlib.pyplot as plt


N = 200
symbols, constellation = gen_rand_qam_symbols(N, M=4)

pulse_train = create_pulse_train(symbols, 10)
# convert from symbols to complex numbers
real = [z.real for z in pulse_train]
imaginary = [z.imag for z in pulse_train]
x_axis = np.linspace(0, 199, 2000)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(x_axis, real)
ax1.set_title('Real Part of Random Pulse Train')
ax1.set_xlabel('Sample Indices')
ax1.set_ylabel('Amplitudes')

ax2.plot(x_axis, imaginary)
ax2.set_title('Imaginary Part of Random Pulse Train')
ax2.set_xlabel('Sample Indices')
ax2.set_ylabel('Amplitudes')
plt.tight_layout()
plt.show()
#plt.savefig("lab2_2C.png")
plt.close()

# perform convolution with raised cosine
beta_list = [0.0, 0.25, 0.5, 0.75, 1.0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
for b in beta_list:
    span = 8
    sps = 10
    rc = get_rc_pulse(b, span, sps)
    real_y_axis = np.convolve(rc, real)
    imag_y_axis = np.convolve(rc, imaginary)
    real_y_axis = real_y_axis/max(rc)
    imag_y_axis = imag_y_axis/max(rc)
    x_axis = np.arange(len(real_y_axis)) / 10
    ax1.plot(x_axis, real_y_axis, label="beta = " + str(b))
    ax1.set_title('Pulsed Shaped In-Phase Values')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitudes')

    ax2.plot(x_axis, imag_y_axis, label="beta = " + str(b))
    ax2.set_title('Pulse Shaped Quadrature Values')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitudes')

# highlight no ISI
delay = (len(rc) - 1) // 2
sample_indices = np.linspace(delay/sps, (delay + N*sps)/sps, N*sps)
ax1.scatter(sample_indices, real, s=5)
ax2.scatter(sample_indices, imaginary, s=5)
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.show()
plt.close()

'''
beta_list = [0.0, 0.25, 0.5, 0.75, 1.0]

span = 8
sps = 10
n = np.arange(-span * sps, span * sps + 1) # arange is end exlusive
x_axis = n / sps

y_axis = get_rc_pulse(beta_list[0], span, sps)
plt.plot(x_axis, y_axis, marker='.', c='r', label="beta = 0.0")
y_axis = get_rc_pulse(beta_list[1], span, sps)
plt.plot(x_axis, y_axis, marker='.', c='g', label="beta = 0.25")
y_axis = get_rc_pulse(beta_list[2], span, sps)
plt.plot(x_axis, y_axis, marker='.', c='b', label="beta = 0.5")
y_axis = get_rc_pulse(beta_list[3], span, sps)
plt.plot(x_axis, y_axis, marker='.', c='y', label="beta = 0.75")
y_axis = get_rc_pulse(beta_list[4], span, sps)
plt.plot(x_axis, y_axis, marker='.', c='m', label="beta = 1.0")
plt.axhline(y=0, color='k', linestyle='-', label="X Axis")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Raised Cosines with Different Rolloff Factors')
plt.show()


x_axis = [z.real for z in constellation]
y_axis = [z.imag for z in constellation]

plt.figure()
plt.scatter(x_axis, y_axis, color='red')

plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.grid(True, linestyle='--')
plt.title('QPSK Constellation with Unit Average Energy')
plt.show()
'''
