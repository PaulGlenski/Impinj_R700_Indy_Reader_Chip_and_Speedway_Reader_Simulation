##
#  This is the top-level test script for the system model
##

import numpy as np
import matplotlib.pyplot as plt
import systems_r700.model.src.common.utils as ut
from systems_r700.model.src.reader.reader_impairments import ReaderImpairments
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes
from systems_r700.model.src.simulation_engine.simulation_engine import SimulationEngine

plt.close('all')

attrib = ReaderAttributes()
# make any changes to reader attributes here
attrib.tx_.tx_pwr_dbm_ = 30.0
attrib.rx_.return_loss_db_ = [-np.inf]
attrib.rx_.reflection_phase_ = [np.pi/100]

impairments = ReaderImpairments()
# make any chnages to reader impairments here
# impairments.rx.temperature = 30

sim_eng = SimulationEngine(attrib, impairments)
sim_eng.process()

nfft = len(sim_eng.reader_.rfa_.cw_)/2
Fs = attrib.rx_.fs_


freq = np.array(range(-nfft/2, nfft/2))/float(nfft) * Fs
sig_mask = (np.array(range(nfft)) == 0)
df = Fs/nfft
passband_edge_idx = int(attrib.rx_.filter1_bw/Fs * nfft)
passband_range = np.array(range(-passband_edge_idx, passband_edge_idx)) + nfft/2
mask = (passband_range == nfft/2)
passband_range = passband_range[~mask]



# tmp = np.abs(np.fft.fft(np.array(sim_eng.reader_.rfa_.demod_out_[nfft:])))
# tmp = tmp / scale
# tmp[~sig_mask] = tmp[~sig_mask] / np.sqrt(df)
# DEMOD = np.fft.fftshift(20*np.log10(tmp))
#
# tmp = np.abs(np.fft.fft(np.array(sim_eng.reader_.rfa_.filter1_out_[nfft:])))
# tmp = tmp / scale
# tmp[~sig_mask] = tmp[~sig_mask] / np.sqrt(df)
# FILTER1 = np.fft.fftshift(20*np.log10(tmp))
#
# tmp = np.abs(np.fft.fft(np.array(sim_eng.reader_.rfa_.bb_[nfft:])))
# tmp = tmp / scale
# tmp[~sig_mask] = tmp[~sig_mask] / np.sqrt(df)
# ADC = np.fft.fftshift(20*np.log10(np.abs(np.fft.fft(np.array(sim_eng.reader_.rfa_.bb_[nfft:])))))

int_freq = np.array(sim_eng.reader_.rfa_.lo.phase_noise.freq) * Fs
int_freq = np.append(np.flipud(-int_freq), int_freq)
int_pn = np.array(sim_eng.reader_.rfa_.lo.phase_noise.phase_noise)
int_pn = np.append(np.flipud(int_pn), int_pn)

# illustrate phase noise in modulator
tmp = np.abs(np.fft.fft(np.array(sim_eng.reader_.rfa_.bb_tx_cw[nfft:])))
scale_in = tmp[0]
#scale = float(nfft)
#scale = 1.0
tmp = tmp / scale_in
tmp[~sig_mask] = tmp[~sig_mask]
BB_TX_CW = np.fft.fftshift(20*np.log10(tmp))
mask = (BB_TX_CW < -500)
BB_TX_CW[mask] = -500.0

tmp = np.abs(np.fft.fft(np.array(sim_eng.reader_.rfa_.cw_[nfft:])))
tmp = tmp / scale_in
tmp[~sig_mask] = tmp[~sig_mask]
MOD_CW = np.fft.fftshift(20*np.log10(tmp))

# ut.create_figure_doc(rect_cones=[0.14, 0.15, 0.84, 0.83])
# plt.semilogx(freq, BB_TX_CW, 'k:', label='Tx CW')
# plt.semilogx(freq, MOD_CW, 'y:', label='Tx modulated CW')
# plt.semilogx(attrib.rx_.phase_noise_freq_hz, attrib.rx_.phase_noise_dbc, 'o', markerfacecolor='None', label='Measured Phase Noise')
# plt.semilogx(int_freq, int_pn, 'b-', label='Interpolated Phase Noise Model')
# ut.annotate_figure('', 'Frequency (Hz)', r'PSD $\left(\frac{dBc}{Hz}\right)$', leg_loc=(0.01, 0.1))

# ut.create_figure_doc(rect_cones=[0.14, 0.15, 0.84, 0.83])
# plt.plot(freq / 1e6, BB_TX_CW, 'k:', label='Tx CW')
# plt.plot(freq / 1e6, MOD_CW, 'y:', label='Tx modulated CW')
# plt.plot(np.array(attrib.rx_.phase_noise_freq_hz) / 1e6, attrib.rx_.phase_noise_dbc, 'o', markerfacecolor='None', label='Measured Phase Noise')
# plt.plot(int_freq / 1e6, int_pn, 'b-', label='Interpolated Phase Noise Model')
# ut.annotate_figure('', 'Frequency (MHz)', r'PSD $\left(\frac{dBc}{Hz}\right)$', leg_loc=(0.01, 0.1))


# illustrate phase noise in demodulator
tmp = np.abs(np.fft.fft(np.array(sim_eng.reader_.rfa_.combined_[nfft:])))
scale = tmp[0]
tmp = tmp / scale
# tmp[~sig_mask] = tmp[~sig_mask]
DEMOD_IN = np.fft.fftshift(20*np.log10(tmp))

tmp = np.abs(np.fft.fft(np.array(sim_eng.reader_.rfa_.demod_out_[nfft:])))
tmp = tmp / scale
# tmp[~sig_mask] = tmp[~sig_mask]
DEMOD_OUT = np.fft.fftshift(20*np.log10(tmp))
mask = (DEMOD_OUT < -500)
DEMOD_OUT[mask] = -500.0

# ut.create_figure_doc(rect_cones=[0.14, 0.15, 0.84, 0.83])
# plt.semilogx(freq, DEMOD_IN, 'k:', label='Demodulator Input')
# plt.semilogx(freq, DEMOD_OUT, 'y:', label='Demodulator Output')
# plt.semilogx(attrib.rx_.phase_noise_freq_hz, attrib.rx_.phase_noise_dbc, 'bo', markerfacecolor='None', label='Specified Phase Noise')
# plt.semilogx(int_freq, int_pn, 'b-', label='Interpolated Phase Noise Model')
# ut.annotate_figure('', 'Frequency (Hz)', r'PSD $\left(\frac{dBc}{Hz}\right)$', leg_loc=(0.01, 0.3))

# ut.create_figure_doc(rect_cones=[0.14, 0.15, 0.84, 0.83])
# plt.plot(freq / 1e6, DEMOD_IN, 'k:', label='Demodulator Input')
# plt.plot(freq / 1e6, DEMOD_OUT, 'y:', label='Demodulator Output')
# plt.plot(np.array(attrib.rx_.phase_noise_freq_hz) / 1e6, attrib.rx_.phase_noise_dbc, 'bo', markerfacecolor='None', label='Specified Phase Noise')
# plt.plot(int_freq / 1e6, int_pn, 'b-', label='Interpolated Phase Noise Model')
# ut.annotate_figure('', 'Frequency (MHz)', r'PSD $\left(\frac{dBc}{Hz}\right)$', leg_loc=(0.01, 0.2))

# illustrate pre-filter1 thermal noise noise in demodulator
tmp = np.abs(np.fft.fft(np.array(sim_eng.reader_.rfa_.demod_out_[nfft:])))
tmp = tmp / float(nfft)
tmp[~sig_mask] = tmp[~sig_mask] / np.sqrt(df)
DEMOD_OUT = np.fft.fftshift(tmp) * 1e9

tmp = np.abs(np.fft.fft(np.array(sim_eng.reader_.rfa_.filter1_in_[nfft:])))
tmp = tmp / float(nfft)
tmp[~sig_mask] = tmp[~sig_mask] / np.sqrt(df)
FILTER1_IN = np.fft.fftshift(tmp) * 1e9

tmp = np.abs(np.fft.fft(np.array(sim_eng.reader_.rfa_.filter1_out_[nfft:])))
tmp = tmp / float(nfft)
tmp[~sig_mask] = tmp[~sig_mask] / np.sqrt(df)
FILTER1_OUT = np.fft.fftshift(tmp) * 1e9
print('Specified spectral-density in passband: ' + str(sim_eng.reader_.rfa_.filter1_input_noise_spectral_density))
print('Measured spectral-density in passband: ' + str(np.sqrt(np.mean(FILTER1_OUT[passband_range] ** 2))))


# ut.create_figure_doc(rect_cones=[0.14, 0.15, 0.84, 0.83])
# plt.semilogy(freq / 1e6, DEMOD_OUT, 'k:', label='Demodulator Output')
# plt.semilogy(freq / 1e6, FILTER1_IN, 'y:', label='Filter1 Input')
# plt.semilogy(freq / 1e6, FILTER1_OUT, 'm:', label='Filter1 Output')
# plt.semilogy(freq / 1e6, sim_eng.reader_.rfa_.filter1_input_noise_spectral_density * np.ones(len(freq)), 'b-', label=str(round(sim_eng.reader_.rfa_.filter1_input_noise_spectral_density,3)))
# ut.annotate_figure('', 'Frequency (MHz)', r'Spectral Density $\left(\frac{nV}{\sqrt{Hz}}\right)$', leg_loc=1, ylim=[1e-13, 1e11])

tmp = np.abs(np.fft.fft(np.array(sim_eng.reader_.rfa_.rx_adc_out_[nfft:])))
tmp = tmp / float(nfft)
tmp[~sig_mask] = tmp[~sig_mask] / np.sqrt(df)
RX_ADC = np.fft.fftshift(tmp) * 1e9

tmp = np.abs(np.fft.fft(np.array(sim_eng.reader_.rfa_.bb_[nfft:])))
tmp = tmp / float(nfft)
tmp[~sig_mask] = tmp[~sig_mask] / np.sqrt(df)
BB = np.fft.fftshift(tmp) * 1e9

final_spectral_density = np.sqrt(((sim_eng.reader_.rfa_.filter1_input_noise_spectral_density * sim_eng.reader_.rfa_.post_demod_rx_adc_gain_) ** 2) + (sim_eng.reader_.rfa_.rx_adc_noise_spectral_density ** 2))
print('Specified spectral-density in passband: ' + str(final_spectral_density))
print('Measured spectral-density in passband: ' + str(np.sqrt(np.mean(BB[passband_range] ** 2))))

ut.create_figure_doc(rect_cones=[0.14, 0.15, 0.84, 0.83])
plt.semilogy(freq / 1e6, BB, 'y:', label='ADC after noise')
plt.semilogy(freq / 1e6, RX_ADC, 'k:', label='ADC before noise')
plt.plot(freq / 1e6, sim_eng.reader_.rfa_.rx_adc_noise_spectral_density * np.ones(len(freq)), 'k-', label=str(round(sim_eng.reader_.rfa_.rx_adc_noise_spectral_density, 3)))
plt.semilogy(freq / 1e6, final_spectral_density * np.ones(len(freq)), 'b-', label=str(round(final_spectral_density, 3)))
ut.annotate_figure('', 'Frequency (MHz)', r'Spectral Density $\left(\frac{nV}{\sqrt{Hz}}\right)$')

ut.create_figure_doc(rect_cones=[0.14, 0.15, 0.84, 0.83])
plt.plot(freq / 1e6, BB, 'y:', label='ADC after noise')
plt.plot(freq / 1e6, RX_ADC, 'k:', label='ADC before noise')
plt.plot(freq / 1e6, sim_eng.reader_.rfa_.rx_adc_noise_spectral_density * np.ones(len(freq)), 'k-', label=str(round(sim_eng.reader_.rfa_.rx_adc_noise_spectral_density, 3)))
plt.plot(freq / 1e6, final_spectral_density * np.ones(len(freq)), 'b-', label=str(round(final_spectral_density, 3)))
ut.annotate_figure('', 'Frequency (MHz)', r'Spectral Density $\left(\frac{nV}{\sqrt{Hz}}\right)$', [-5, 5], [0, 1000])


plt.show()


# ut.create_figure()
# plt.semilogx(freq, BB_TX_CW, 'k:', label='Tx CW')
# plt.semilogx(freq, DEMOD, 'b', label='Demod')
# plt.semilogx(freq, FILTER1, 'r', label='Filter1')
# plt.semilogx(freq, ADC, 'g', label='ADC')
# plt.semilogx(attrib.rx_.phase_noise_freq_hz, attrib.rx_.phase_noise_dbc, 'o', markerfacecolor='None', label='Measured Phase Noise')
# ut.annotate_figure('', 'Frequency (MHz)', 'Amplitude (dB)')
#
# ut.create_figure()
# plt.plot(freq / 1e6, BB_TX_CW, 'k:', label='Tx CW')
# plt.plot(freq / 1e6, DEMOD, 'b', label='Demod')
# plt.plot(freq / 1e6, FILTER1, 'r', label='Filter1')
# plt.plot(freq / 1e6, ADC, 'g', label='ADC')
# ut.annotate_figure('', 'Frequency (MHz)', 'Amplitude (dB)')

plt.show()
