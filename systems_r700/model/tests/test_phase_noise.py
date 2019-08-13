import cmath
import numpy as np
import matplotlib.pyplot as plt
import phase_noise as pn

plt.close('all')

carrier_power_dbm = 0.0
carrier_amp = 10 ** ((carrier_power_dbm - 30.0)/20.0)

freq_hz = [1, 10 ** 3.25, 10 ** 4, 10 ** 5, 10 ** 6]
phase_noise = [-72.5, -72.5, -92.5, -123, -145]
Fs = 20.48e6

# freq_hz = [5e-8, 1e-3, 1e-2, 1e-1, 5e-1]
# phase_noise = [-50, -50, -70, -90, -103.9794]
# Fs = 1.0

# freq_hz = [1, 100., 200., 500.]
# phase_noise = [-40., -40., -70., -80.]
# Fs = 1000.0

p = pn.PhaseNoise(freq_hz, phase_noise, Fs)

n_samples = 8*p.n_fft
df = Fs / n_samples

f_idx = 200
Fc = (float(f_idx) / n_samples) * Fs
sig = np.zeros(n_samples, dtype=complex)
sig_plus_noise = np.zeros(n_samples, dtype=complex)
for s in range(n_samples):
    sig[s] = carrier_amp * cmath.exp(1j * 2 * cmath.pi * (Fc / Fs) * s)
    sig_plus_noise[s] = p.add_noise(sig[s])
SIG = np.fft.fft(sig)
scale = np.max(np.abs(SIG))
SIG = SIG / scale
SIG_PLUS_NOISE = np.fft.fft(sig_plus_noise) / scale
sig_mask = (np.array(range(n_samples)) == f_idx)
SIG[~sig_mask] = SIG[~sig_mask] / np.sqrt(df)
SIG_PLUS_NOISE[~sig_mask] = SIG_PLUS_NOISE[~sig_mask] / np.sqrt(df)


SIG_dbm = 20*np.log10(np.abs(SIG)) + 30.0
mask = (SIG_dbm < -500)
SIG_dbm[mask] = -500.0
SIG_PLUS_NOISE_dbm = 20*np.log10(np.abs(SIG_PLUS_NOISE)) + 30.0
freq = np.array(range(n_samples))/float(n_samples) * Fs
freq[0] = freq_hz[0]

pn = p.noise_samples - 1.0
PN = (1.0/p.n_fft) * np.fft.fft(pn) + 30.0
PN_dbm = 20*np.log10(np.abs(PN))

spec_freq = np.array(freq_hz)
spec_freq = np.append(np.flipud(-spec_freq), spec_freq)
spec_freq = spec_freq + Fc
spec_pn = np.array(phase_noise)
spec_pn = np.append(np.flipud(spec_pn), spec_pn)

int_freq = np.array(p.freq) * Fs
int_freq = np.append(np.flipud(-int_freq), int_freq)
int_freq = int_freq + Fc
int_pn = np.array(p.phase_noise)
int_pn = np.append(np.flipud(int_pn), int_pn)


plt.figure()
plt.hold(True)
plt.semilogx(spec_freq, spec_pn, 'o', markerfacecolor='None', label='Specified phase-noise points')
plt.semilogx(int_freq, int_pn, '.-', label='Interpolated phase-noise')
#plt.semilogx(p.freq * Fs, PN_dbc, '-')
plt.semilogx(freq, SIG_dbm, 'k.-', label='Pure Carrier')
plt.semilogx(freq, SIG_PLUS_NOISE_dbm, 'r:', label='Carrier with phase-noise')
plt.legend()
plt.grid(True)
#plt.xlim(freq_hz[0], Fs)
plt.hold(False)
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'Power Spectral Density $\left(\frac{dBm}{Hz}\right)$')
plt.title('PSD against linear frequency (Fs = ' + str(Fs / 1e6) + ' MHz, Fc = ' + str(Fc / 1e6) + ' MHz)')

plt.figure()
plt.hold(True)
plt.plot(spec_freq, spec_pn, 'o', markerfacecolor='None', label='Specified phase-noise points')
plt.plot(int_freq, int_pn, '-', label='Interpolated phase-noise')
#plt.plot(p.freq * Fs, PN_dbc, '-')
plt.plot(freq, SIG_dbm, 'k-', label='Pure Carrier')
plt.plot(freq, SIG_PLUS_NOISE_dbm, 'r:', label='Carrier with phase-noise')
plt.legend()
plt.grid(True)
#plt.xlim(freq_hz[0], freq_hz[-1]+Fc)
plt.hold(False)
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'Power Spectral Density $\left(\frac{dBm}{Hz}\right)$')
plt.title('PSD against linear frequency (Fs = ' + str(Fs / 1e6) + ' MHz, Fc = ' + str(Fc / 1e6) + ' MHz)')


plt.show()
#plt.waitforbuttonpress
#plt.close('all')