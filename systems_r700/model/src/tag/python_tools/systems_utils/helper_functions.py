import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.special as sp_special

def plot_spectrum(x, fs=0, rbw=0, plot_fig=0, title=''):
    # rbw = 100e3
    N = int(fs / rbw)
    # rbw = fs/npts
    overlap = 0.5
    npts = len(x)
    nseg = round(npts / N / overlap)

    # t = np.arange(0, npts/fs-1/fs, 1/fs)
    # x = np.exp(1j*fo*2*np.pi*t)

    start = 0
    stop = N

    for i in range(nseg - 1):
        xseg = x[start:stop]
        xlen = len(xseg)
        win = np.hanning(xlen)
        xwin = np.multiply(xseg, win)
        xf = scipy.fftpack.fft(xwin)
        start = start + int(N * 0.5)
        stop = start + N

        if i == 0:
            xmag = np.square(np.abs(xf))
        else:
            xmag = xmag + np.square(np.abs(xf))

    # print(i)

    xmag = xmag / nseg

    freq = np.arange(-fs / 2, fs / 2, rbw)
    ya = xmag
    yam = np.max(ya)
    mag_dB = 10 * np.log10(ya / yam)

    if plot_fig:
        fig, ax = plt.subplots()
        ax.plot(freq[0:len(mag_dB)] / 1e3, np.fft.fftshift(mag_dB))
        ax.set_xlabel('Frequency in kHz')
        ax.set_ylabel('Power spectrum (dB)')
        ax.grid(True)
        ax.set_title(title)
        # plt.show()

    if plot_fig:
        pass
        # return freq, np.fft.fftshift(ya), ax
    else:
        return freq, np.fft.fftshift(ya), mag_dB


def plot_ber_curves(EbNo_dB_vec,ber_array):
    plt.figure()
    plt.semilogy(EbNo_dB_vec, ber_array, linewidth=2, label='Simulation')
    plt.xlabel('EbNo (dB)')
    plt.ylabel('BER')
    plt.grid(True)
    plt.show()


def calculate_num_bits(ebno_db,
                       bit_errors_minimum=256,
                       ber_minimum=1e-6,
                       mod_type=None):

    ebno_linear = 10.0 ** (ebno_db / 10.0)
    ber = 0.5 * sp_special.erfc(np.sqrt(ebno_linear))
    bits_desired = bit_errors_minimum / ber
    bits_desired_log2 = np.ceil(np.log10(bits_desired) / np.log10(2.))

    # Calculate maximum number of bit errors and bits
    bits_maximum = bit_errors_minimum / ber_minimum
    bits_maximum_log2 = np.ceil(np.log10(bits_maximum) / np.log10(2.))

    ber_bits = int(np.minimum(bits_desired, bits_maximum))

    if '3b4b' in mod_type:
        ber_bits -= np.mod(ber_bits, 3)

    return int(ber_bits)


def cdf_plot(x, num_bins=40):
    pk_idx_ideal = np.bincount(x.astype(int)).argmax()
    idx_err = -np.abs(x - pk_idx_ideal)
    counts, bin_edges = np.histogram(idx_err, bins=num_bins, normed=True)
    cdf = np.cumsum(counts)

    plt.figure()
    plt.semilogy(bin_edges[1:], cdf, label='CDF plot',linewidth=2)
    plt.hold(True)
    plt.xlabel('Negative absolute error from peak pmf index')
    plt.ylabel('Probability')
    plt.grid(True)
