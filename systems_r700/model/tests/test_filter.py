##
#  This is the top-level script for testing filter implementations
##

import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import systems_r700.model.src.common.utils as ut
from systems_r700.model.src.common.analog_filter import AnalogFilter
from systems_r700.model.src.common.awgn import Awgn

plt.close('all')

prec = 5
order = 3
ripple = 0.5
bw = 1.18e6
Fs = 20.46e6
noise_spectral_density = 4.1


len_samples = 8 * 1024
df = Fs / len_samples

thermal = Awgn()
thermal.set_noise_spectral_density(noise_spectral_density, Fs)
sig = thermal.gen_noise(len_samples)

an_filter = AnalogFilter(order, ripple, Fs, bw)

b = an_filter.b
a = an_filter.a
f,h = sp.freqz(b, a)

ut.create_figure_doc()
plt.plot(0.5 * Fs * f / np.pi / 1e6, 20*np.log10(np.abs(h)))
plt.xlabel('Frequency (MHz)')
plt.ylabel(r'Magnitude Response (dB)')
plt.xlim(0, Fs/2/1e6)

ut.create_figure_doc()
plt.plot(0.5 * Fs * f / np.pi / 1e6, np.angle(h))
plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel(r'Phase Response (Radians)')
plt.xlim(0, Fs/2/1e6)

f, d = sp.group_delay((b, a))
ut.create_figure_doc()
plt.plot(0.5 * Fs * f / np.pi / 1e6, d / Fs * 1e6)
plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel(r'Group Delay $({\mu}s)$')
plt.xlim(0, Fs/2/1e6)


filt_sig1 = an_filter.filter_signal(sig)

filt_sig2 = np.zeros(len_samples, dtype=complex)
for samp in range(len_samples):
    filt_sig2[samp] = an_filter.process(sig[samp])

ut.create_figure_doc(rect_cones=[0.11, 0.15, 0.83, 0.80])
plt.plot(np.real(filt_sig1 - filt_sig2), '.-')
#plt.plot(np.real(filt_sig2), '.-')
plt.xlabel('Samples')
plt.ylabel(r'Error in filtered signal')

step_len = 100
sig2 = np.ones(step_len)
an_filter = AnalogFilter(order, ripple, Fs, bw)
filt_sig3 = np.zeros(step_len)
for samp in range(step_len):
    filt_sig3[samp] = an_filter.process(sig2[samp])
t = np.arange(0, step_len)/float(Fs)
ut.create_figure_doc(rect_cones=[0.11, 0.15, 0.83, 0.80])
plt.plot(t*1000000, filt_sig3, '.-')
plt.xlabel('Time (us)')
plt.ylabel(r'Step Response')
plt.show()

