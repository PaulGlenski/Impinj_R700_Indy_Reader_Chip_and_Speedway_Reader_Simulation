##
#  This is the top-level test script for the Awgn class
##

import numpy as np
import scipy.signal as sp
import scipy.constants as consts
import matplotlib.pyplot as plt
import systems_r700.model.src.common.utils as ut
from systems_r700.model.src.common.awgn import Awgn

plt.close('all')

prec = 5
temp = 20
nf = 0.0
bw = 1.28e6
Fs = 20.46e6
load = 50

carrier_volt = 1.0

# Noise spectral density expressed in nV/root-Hz
#noise_spectral_density = 4.1
noise_spectral_density = 259.09

# anti-aliasing analog filter
gd = 250
a = 1.0
#b = sp.remez(2 * gd + 1, [0, bw, bw+bw/10.0, Fs/2], [1, 0], Hz=Fs)
b, a = sp.cheby1(3, 0.5, 2*bw/Fs)

f,h = sp.freqz(b, a)

ut.create_figure()
plt.plot(0.5 * Fs * f / np.pi / 1e6, 20*np.log10(np.abs(h)))

filter_loss = (10 ** (-0.5/20))
print filter_loss

thermal = Awgn()
thermal.set_noise_spectral_density(noise_spectral_density / filter_loss, Fs)

num_samples = 32*1024
f_idx = num_samples / 32
f = float(f_idx) / num_samples
df = Fs/num_samples

band_edge_idx = int(round(1.1e6/Fs*num_samples))


noise = thermal.gen_noise(num_samples + 2* gd)
sig = carrier_volt * np.exp(1j* 2 * np.pi * f * np.arange(num_samples + 2 * gd))
sig_plus_noise = sig + noise
sig_plus_noise_filt = sp.lfilter(b, a, sig_plus_noise)
noise_filt = sp.lfilter(b, a, noise)

#sig_plus_noise_filt = sp.convolve(sig_plus_noise, lpf, mode='full', method='direct')
#noise_filt = sp.convolve(noise, lpf, mode='full', method='direct')

sig = sig[0:num_samples]
sig_plus_noise = sig_plus_noise[0:num_samples]
sig_plus_noise_filt = sig_plus_noise_filt[2 * gd : 2 * gd + num_samples]
noise_filt = noise_filt[2 * gd : 2 * gd + num_samples]


print ''
print 'Theoretical SNR : ' + str(round(10 * np.log10((carrier_volt ** 2) / ((2 * (noise_spectral_density * 1e-9) ** 2) * (2*bw))), prec))
print 'Theoretical Noise VSD: ' + str(round(noise_spectral_density, prec))
print 'Time-domain Measurements:'
print '  Signal power  : ' + str(round(10 * np.log10(np.mean(np.abs(sig) ** 2)), prec)) + ' dBW'
print '  Noise power   : ' + str(round(10 * np.log10(np.mean(np.abs(noise_filt) ** 2)), prec)) + ' dBW'
print '  SNR           : ' + str(round((10 * np.log10(np.mean(np.abs(sig) ** 2))) -
                                       (10 * np.log10(np.mean(np.abs(noise_filt) ** 2))), prec)) + ' dB'
print '  Noise VSD     : ' + str(round(np.sqrt(np.mean(np.abs(noise_filt) ** 2) / (2*bw)) * 1e9, prec)) + ' nV/root-Hz'

noise_idx_passband = np.append(range(f_idx), range(f_idx+1, band_edge_idx))
noise_idx_passband = np.append(noise_idx_passband, range(num_samples-band_edge_idx, num_samples))
sig_mask = (np.array(range(num_samples)) == f_idx)

SIG = np.fft.fft(sig)
SIG_PLUS_NOISE = np.fft.fft(sig_plus_noise)
SIG_PLUS_NOISE_FILT = np.fft.fft(sig_plus_noise_filt)
scale = float(num_samples)
SIG = SIG / scale
SIG_PLUS_NOISE = SIG_PLUS_NOISE / scale
SIG_PLUS_NOISE_FILT = SIG_PLUS_NOISE_FILT / scale

print 'Frequency-domain Measurements:'
print '  Signal power  : ' + str(round(10 * np.log10(np.sum(np.abs(SIG_PLUS_NOISE_FILT[sig_mask]) ** 2)), prec)) + ' dBW'
print '  Noise power   : ' + str(round(10 * np.log10(np.sum(np.abs(SIG_PLUS_NOISE_FILT[~sig_mask]) ** 2)), prec)) + ' dBW'
print '  SNR           : ' + str(round((10 * np.log10(np.sum(np.abs(SIG_PLUS_NOISE_FILT[sig_mask]) ** 2))) -
                                       (10 * np.log10(np.sum(np.abs(SIG_PLUS_NOISE_FILT[~sig_mask]) ** 2))), prec)) + ' dB'
print '  Noise VSD     : ' + str(round(np.sqrt(np.mean(np.abs(SIG_PLUS_NOISE[noise_idx_passband]) ** 2) / df) * 1e9, prec)) + ' nV/root-Hz'
print '  Noise VSD     : ' + str(round(np.sqrt(np.mean(np.abs(SIG_PLUS_NOISE_FILT[noise_idx_passband]) ** 2) / df) * 1e9, prec)) + ' nV/root-Hz'
print ''


SIG[~sig_mask] = SIG[~sig_mask] / np.sqrt(df)
SIG_PLUS_NOISE[~sig_mask] = SIG_PLUS_NOISE[~sig_mask] / np.sqrt(df)
SIG_PLUS_NOISE_FILT[~sig_mask] = SIG_PLUS_NOISE_FILT[~sig_mask] / np.sqrt(df)
SIG_db = 20*np.log10(np.abs(SIG))
SIG_PLUS_NOISE_db = 20*np.log10(np.abs(SIG_PLUS_NOISE))
SIG_PLUS_NOISE_FILT_db = 20*np.log10(np.abs(SIG_PLUS_NOISE_FILT))
freq = np.array(range(num_samples))/float(num_samples) * Fs

# plot the figure for PSD
# ut.gen_figure()
# plt.plot(freq/1e6, SIG_PLUS_NOISE_FILT_db, '-', label='Signal + thermal noise + filter')
# plt.plot(freq/1e6, SIG_PLUS_NOISE_db, 'r-', label='Signal + thermal noise')
# plt.plot(freq/1e6, SIG_db, 'k:', label='Signal')
# plt.grid(True)
# plt.legend()
# #plt.title(r'PSD of signal with $F_s$ = ' + str(Fs/1e6) + ' MHz, CW at \n' + str(f*Fs/1e6) + ' MHz with 0 dB, and thermal noise $V_{rms}$ = ' + str(noise_v_rms * 1e6) + ' $\mu$V')
# plt.xlabel('Frequency (MHz)')
# plt.ylabel(r'Power Spectral Density $\left(\frac{dB}{Hz}\right)$')
# plt.xlim(0, Fs/2/1e6)


# plot the figure for nV/root-Hz
ut.create_figure()
plt.semilogy(freq/1e6, abs(SIG_PLUS_NOISE_FILT) * 1e9, 'm:', label='Signal + thermal noise + filter')
plt.semilogy(freq/1e6, abs(SIG_PLUS_NOISE) * 1e9, 'y:', label='Signal + thermal noise')
plt.semilogy(freq/1e6, abs(SIG) * 1e9, 'k:', label='Signal')
plt.semilogy(freq/1e6, noise_spectral_density * np.ones(len(freq)), 'b-', label='Specified Spectral Density')
plt.grid(True)
plt.legend()
#plt.title(r'PSD of signal with $F_s$ = ' + str(Fs/1e6) + ' MHz, CW at \n' + str(f*Fs/1e6) + ' MHz with 0 dB, and thermal noise $V_{rms}$ = ' + str(noise_v_rms * 1e6) + ' $\mu$V')
plt.xlabel('Frequency (MHz)')
plt.ylabel(r'RMS Voltage Spectral Density $\left(\frac{nV}{\sqrt{Hz}}\right)$')
plt.xlim(0, Fs/2/1e6)

# zoomed plot for nV/root-Hz
ut.create_figure()
plt.semilogy(freq/1e6, abs(SIG_PLUS_NOISE_FILT) / np.sqrt(1) * 1e9, 'm:', label='Signal + thermal noise + filter')
plt.semilogy(freq/1e6, abs(SIG_PLUS_NOISE) / np.sqrt(2) * 1e9, 'y:', label='Signal + thermal noise')
plt.semilogy(freq/1e6, abs(SIG) / np.sqrt(2) * 1e9, 'k:', label='Signal')
plt.semilogy(freq/1e6, noise_spectral_density * np.ones(len(freq)), 'b-', label='Specified Spectral Density')
plt.grid(True)
plt.legend()
#plt.title(r'PSD of signal with $F_s$ = ' + str(Fs/1e6) + ' MHz, CW at \n' + str(f*Fs/1e6) + ' MHz with 0 dB, and thermal noise $V_{rms}$ = ' + str(noise_v_rms * 1e6) + ' $\mu$V')
plt.xlabel('Frequency (MHz)')
plt.ylabel(r'RMS Voltage Spectral Density $\left(\frac{nV}{\sqrt{Hz}}\right)$')
plt.xlim(0, 2)
plt.ylim(0.01, 1000.0)

plt.show()