##
#  This class implements some utility functions that will be used
#  across multiple classes throughout this model.
##

import os
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D

def get_root_directory():
    file_directory = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.split(os.path.split(os.path.split(os.path.split(file_directory)[0])[0])[0])[0]
    return root_directory

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def lin2db(lin):
    db = 20.0 * np.log10(lin)
    return db


def db2lin(db):
    gain = 10.0 ** (db / 20.0)
    return gain


def dbm2watts(dbm):
    pwr = (10.0 ** (dbm/10.0)) * 1e-3
    return pwr

def dbm2amp(dbm, cmplx=True):
    pwr = dbm2watts(dbm)
    if cmplx:
        amp = np.sqrt(pwr)
    else:
        amp = np.sqrt(2.0*pwr)
    return amp


def amp2dbm(amp, cmplx=True):
    if cmplx:
        pwr = (amp ** 2.0)
    else:
        pwr = (amp ** 2.0)/2.0
    dbm = 10.0*np.log10(pwr/1e-3)
    return dbm


def get_spectrum(self, input_signal, fs, fft_count=1024):
    input_signal = input_signal[0:fft_count]
    num_samples = len(input_signal)
    k = np.arange(num_samples)
    T = float(num_samples)/float(fs)
    frq = k/T  # Frequency span (scaled for time)
    output = np.fft(input_signal)/num_samples  # Normalized FFT, as python implementation doesn't normalize
    frq = frq[range(num_samples/2)]
    output = output[range(num_samples/2)]
    return frq, np.real(output)


def compute_sensitivity(ebn0_db, ber, ber_threshold=1e-4):
    ber_threshold_log = np.log10(ber_threshold)

    # perform linear interpolation in the Log-dB domain
    idx = (np.log10(ber) <= ber_threshold_log).nonzero()
    if len(idx[0]) > 0:
        x1 = ebn0_db[idx[0][0]]
        y1 = np.log10(ber[idx[0][0]])
        x2 = ebn0_db[idx[0][0] - 1]
        y2 = np.log10(ber[idx[0][0] - 1])
        slope = (y1 - y2) / (x1 - x2)
        sensitivity_ebn0 = ((ber_threshold_log - y2) + slope * x2) / slope
    else:
        sensitivity_ebn0 = 0
    return sensitivity_ebn0


def q_func(val):
    return 0.5 * sp.erfc(val / np.sqrt(2.0))

def create_figure_doc(w=6, h=3, rect_cones=[0.10, 0.15, 0.88, 0.83], polar=False, ncols=1, nrows=1, sharex=False, sharey=False):
    if (nrows*ncols) > 1:
        fig, axarr = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey, num=None, figsize=(w,h), dpi=300, facecolor='w', edgecolor='k')
    else:
        fig = plt.figure(num=None, figsize=(w, h), dpi=300, facecolor='w', edgecolor='k')
        axarr = fig.add_axes(rect_cones, polar=polar)
    return fig, axarr


def create_figure(w=6, h=3, polar=False):
    fig = plt.figure(num=None, figsize=(w, h), facecolor='w', edgecolor='k')
    if polar:
        ax = fig.add_axes(polar=True)
    else:
        plt.axes()
    plt.grid(True)
    return fig

def create_3d_figure_doc(w=6, h=3):
    fig = plt.figure(num=None, figsize=(w, h), dpi=300, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d')
    ax.tick_params(axis='both', which='major', labelsize=6)
    plt.grid(False)
    return ax

def create_3d_figure(w=6, h=3):
    fig = plt.figure(num=None, figsize=(w, h), facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d')
    ax.tick_params(axis='both', which='major', labelsize=6)
    plt.grid(False)
    return ax

def annotate_figure(title='', xlabel='', ylabel='', zlabel='', xlim=None, ylim=None, legend=True, leg_loc=0, grid=True,
                    titlesize=6, labelsize=6, ticklabelsize=6, legendsize=6):
    ax = plt.gca()
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    if hasattr(ax, 'get_zlim'):
        ax.set_zlabel(zlabel, fontsize=labelsize)
    ax.tick_params(axis='both', which='major', labelsize=ticklabelsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if legend:
        plt.legend(loc=leg_loc, fontsize=legendsize)
    ax.grid(grid)

