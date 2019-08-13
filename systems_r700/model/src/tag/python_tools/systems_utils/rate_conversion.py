"""
rate_conversion.py
"""

import numpy as np
import matplotlib.pylab as plt
import os
import pickle
from datetime import datetime
import shlex
import subprocess


def hard_limit(x):
    """
    Hard limit x with threshold at zero.

    Args:
        x: Value (or array of values) to be sliced
    """
    return 2 * (x > 0) - 1


def get_dpll_gains(frac_noise_bw, damping_factor, kphs, kosc):
    """Return proportional(k1) and integral(k2) PLL gains

    Args:
        frac_noise_bw (float): (Bn)(Tsample) = Bn/Fsample = fractional noise BW

        damping_factor (float): Loop damping factor

        kphs (float): Phase detector gain

        kosc (float): NCO gain

    Returns:
        kp (float): Proportional filter gain

        ki (float): Integral filter gain
    """
    # Simplify the notation
    z = damping_factor

    # Compute the gains
    f1 = frac_noise_bw / (z + 1. / (4 * z))
    f2 = 1 + (2. * z) * f1 + f1 ** 2.
    kphs_kosc_kp = 4. * z * f1 / f2
    kphs_kosc_ki = 4 * f1 ** 2. / f2

    return kphs_kosc_kp / (kphs * kosc), kphs_kosc_ki / (kphs * kosc)


class TedClass(object):
    def __init__(self, pts, **kwargs):
        """Zero-crossing timing error detector

        Args:
            pts (int) : Number of sample points in simulation

            decimate (int) (optional): Rate at which error is computed.
                Defaults to 2

            gardner (Bool) (optional): True=use gardner algorithm
                False=use zero-crossing algorithm (default)
        """
        # optional args
        self.decimate = kwargs.pop('decimate', 2)
        self.gardner = kwargs.pop('gardner', False)
        self.quant = kwargs.pop('quant', None)
        assert len(kwargs) == 0, \
            "unrecognized params passed in: %s" % ','.join(kwargs.keys())

        self.err = np.zeros(pts)
        self.ted_sign = np.zeros(pts)
        self.buf = np.zeros(2)
        self.is_peak_sample = np.zeros(pts)

    def execute(self, x, smpl_idx):
        """
        Args:
            x (float): Next input sample

            smpl_idx (int): Sample index (from external loop counter)

        """
        # Compute an error at 1/SAMPLES_PER_FEATURE
        if smpl_idx % self.decimate == 0:
            if self.gardner:
                xin = x
                buf_1 = self.buf[1]
            else:
                xin = hard_limit(x)
                buf_1 = hard_limit(self.buf[1])

            self.err[smpl_idx] = self.buf[0] * (buf_1 - xin)
            self.is_peak_sample[smpl_idx] = 1
            self.ted_sign[smpl_idx] = xin - buf_1
        else:
            self.err[smpl_idx] = 0

        if (self.quant is not None) and (not self.quant.float_calc):
            self.err[smpl_idx] = quantize("Ted", self.err[smpl_idx],
                                          self.quant,
                                          smpl_idx=smpl_idx)

        # Shift at 2/SAMPLES_PER_FEATURE
        if smpl_idx % (self.decimate / 2) == 0:
            self.buf = np.array([x, self.buf[0]])


class FarrowClass(object):
    def __init__(self, **kwargs):
        """Farrow structure for polynomial interpolation.

        The polynomial filter defaults to a cubic interpolation and is
        implemented using Horner's rule. Here is the cubic example:

         (4 x 1) = (4 x 4)         * (4 x 1)

                [ 1/6, -1/2,  1/2, -1/6 ]   [x[n+2]]
            V = [ 0,    1/2, -1,    1/2 ] * [x[n+1]]
                [ -1/6, 1,   -1/2, -1/3 ]   [x[n]  ]
                [ 0,    0,    1,    0   ]   [x[n-1]]

            y = ((mu*V[3] + V[2])*mu + V[1])*mu + V[0]

        Args:
            interp (optional)(str): Interpolation method defaults to 'cubic'
            alpha (optional)(float): Coefficient parameter for parabolic
                interpolation. Defaults to 0.5.
        """
        self.c = None

        # optional args
        self.interp = kwargs.pop('interp', 'cubic')
        self.alpha = kwargs.pop('alpha', 0.5)
        self.quant = kwargs.pop('quant', None)
        assert len(kwargs) == 0, \
            "unrecognized params passed in: %s" % ','.join(kwargs.keys())

        # check interpolation method
        if self.interp not in ['cubic', 'parabolic']:
            raise ValueError("Interpolation method must be cubic or parabolic")

        # set interpolator coefficients
        self.set_coefficients()

    def __repr__(self):
        if self.interp == 'cubic':
            return 'Farrow(interp=%r)' % self.interp
        else:
            return 'Farrow(interp=%r, alpha=%r)' % (self.interp, self.alpha)

    def set_coefficients(self):
        """Create a matrix of coefficient values
        """
        if self.interp == 'cubic':
            self.c = np.matrix(((0, 0, 1, 0),
                                (-1. / 6, 1, -0.5, -1. / 3),
                                (0, 0.5, -1, 0.5),
                                (1. / 6, -0.5, 0.5, -1. / 6)))
        else:
            self.c = np.matrix(((0, 0, 1, 0),
                                (-self.alpha, self.alpha + 1, self.alpha - 1,
                                 -self.alpha),
                                (self.alpha, -self.alpha, -self.alpha,
                                 self.alpha)))

    def horners_rule(self, cvec, x):
        """Horner's method for polynomial calculation

            c[3]x^3 + c[2]x^2 + c[1]x + c[0] = ((c[3]x + c[2])x + c[1])x + c[0]
        """
        # p = 0
        # for c in cvec[-1::-1]:
        #     p = p * x + c
        p = np.polyval(cvec[-1::-1], x)
        return p

    def execute(self, basepoint_set, mu, smpl_idx=None):
        """Execute the Farrow filter

        Args:
            basepoint_set: Array of values corresponding to next basepoint set
            mu: Fractional interpolation interval

        """
        # Reverse and transpose the basepoint set vector which
        # we denote by x for convenience in this comment
        #                                     |x[n+2]|
        #                                     |x[n+1]|
        # |x[n-1], x[n], x[n+1], x[n+2]|  --> |x[n]  |
        #                                     |x[n-1]|
        x_matrix = np.mat(basepoint_set[::-1])
        v_matrix = self.c * np.transpose(x_matrix)
        v_array = np.squeeze(np.asarray(v_matrix))

        if (self.quant is None) or self.quant.float_calc:
            return self.horners_rule(v_array, mu)
        else:
            return quantize("Farrow", self.horners_rule(v_array, mu),
                            self.quant,
                            smpl_idx=smpl_idx)


# else:
#            return self.horners_rule(v_array, mu)


def quantize(block, x, quant, smpl_idx=None):
    """Quantize and check for overflow"""
    unsigned = quant.signed == 0
    integer = quant.integer
    fractional = quant.fractional
    rnd = quant.rounding

    y = x * 2 ** fractional

    max_int = (2 ** (fractional + integer + unsigned)) - 1
    if unsigned:
        min_int = 0
    else:
        min_int = -2 ** (fractional + integer)

    if y > max_int:
        print("clipped pos: ", block, smpl_idx, y, max_int)
        y = max_int
    if y < min_int:
        print("clipped neg: ", block, smpl_idx, y, min_int)
        y = min_int
    if rnd:
        return np.round(y) * 2 ** -fractional
    else:
        if type(y) == float:
            return np.floor(y) * 2. ** -fractional
        else:
            return (np.floor(np.real(y)) + np.floor(
                np.imag(y))) * 2. ** -fractional


class LoopFilterClass(object):
    # def __init__(self, **kwargs):
    def __init__(self, config=0, mode='drc'):
        """Proportional + Integral loop filter

        Args:
            kp: Proportional gain
            kp: Integral gain
        """
        # self.BnT = kwargs.pop('BnT', 0)
        # self.BnT_shift = kwargs.pop('BnT_shift', 0)
        # self.shift_idx = kwargs.pop('shift_idx', None)
        # self.zeta = kwargs.pop('zeta', 1)
        # self.kphs = kwargs.pop('kphs', 1)
        # self.kosc = kwargs.pop('kosc', 1)
        # self.quant = kwargs.pop('quant', None)
        # assert len(kwargs) == 0, \
        #     "unrecognized params passed in: %s" % ','.join(kwargs.keys())
        if mode is 'drc':
            self.shift_idx = config.trl_dict['drc_shift']
            self.kp_acq = config.trl_dict['drc_kp'][0]
            self.kp_trk = config.trl_dict['drc_kp'][1]
            self.ki_acq = config.trl_dict['drc_ki'][0]
            self.ki_trk = config.trl_dict['drc_ki'][1]
        self.integral = 0
        self.proportional = 0
        self.quant = None
        # self.set_gain(self.BnT)

        # self.kp, self.ki = get_dpll_gains(frac_noise_bw=self.BnT,
        #                                  damping_factor=self.zeta,
        #                                  kphs=self.kphs,
        #                                  kosc=self.kosc)

    def set_gain(self, BnT):
        """Setter for the filter gains

        Args:
            kp (float): Proportional gain
            ki (float): Integral gain
        """
        self.kp, self.ki = get_dpll_gains(frac_noise_bw=BnT,
                                          damping_factor=self.zeta,
                                          kphs=self.kphs,
                                          kosc=self.kosc)
        # self.kp = kp
        # self.ki = ki

    def _get_loop_gains(self, n):
        if n <= self.shift_idx:
            return self.kp_acq, self.ki_acq
        else:
            return self.kp_trk, self.ki_trk

    def execute(self, x, n):
        """Compute a filter output

        Args:
            x: Next filter input sample
        """

        kp, ki = self._get_loop_gains(n)
        
        # self.proportional = self.kp * x
        # self.integral += self.ki * x
        self.proportional = kp * x
        self.integral += ki * x

        if (self.quant is None) or self.quant.float_calc:
            return self.proportional + self.integral
        else:
            return quantize("LPF", self.proportional + self.integral,
                            self.quant)


class NcoClass(object):
    def __init__(self, pts, **kwargs):
        """Numerically controlled oscillator model (digital integrator)

        Args:
            pts: Number of samples in simulation

        Returns:

        """
        self.modulo = kwargs.pop('modulo', None)
        self.quant = kwargs.pop('quant', None)
        assert len(kwargs) == 0, \
            "unrecognized params passed in: %s" % ','.join(kwargs.keys())
        self.acc = np.zeros(pts)
        self.acc_in = 0

    def execute(self, x, smpl_idx):
        """NCO(z) = 1/(1-z**-1)

        Args:
            x: Next input sample
            smpl_idx: Sample index (from external counter)
        """
        if self.modulo is not None:
            self.acc[smpl_idx] = (self.acc[smpl_idx - 1] + x) % self.modulo
        else:
            self.acc[smpl_idx] = (self.acc[smpl_idx - 1] + x)

        if (self.quant is not None) and (not self.quant.float_calc):
            self.acc[smpl_idx] = quantize("nco", self.acc[smpl_idx],
                                          self.quant,
                                          smpl_idx=smpl_idx)


class AtanClass(object):
    def __init__(self, pts, **kwargs):
        """Phase detector maps rectangular into +/-Pi/2 radians"""
        self.quant = kwargs.pop('quant', None)
        assert len(kwargs) == 0, \
            "unrecognized params passed in: %s" % ','.join(kwargs.keys())
        self.atan = np.zeros(pts)

    def execute(self, i_in, q_in, smpl_idx):
        # TODO: this threshold is necessary but it's value should be reviewed.
        if True:
            in_bits = 5
            i_in = np.round(i_in * 2 ** in_bits) * 2 ** -in_bits
            q_in = np.round(q_in * 2 ** in_bits) * 2 ** -in_bits

        if np.abs(q_in) < 0.01:
            self.atan[smpl_idx] = 0.
        else:
            # 2PI factor
            if True:
                self.atan[smpl_idx] = np.arctan(q_in / i_in) / (np.pi / 2)
            else:
                self.atan[smpl_idx] = np.arctan2(q_in, i_in) / np.pi
                if i_in < 0 and q_in >= 0:
                    self.atan[smpl_idx] = -(1.0 - self.atan[smpl_idx])
                elif i_in < 0 and q_in < 0:
                    self.atan[smpl_idx] = self.atan[smpl_idx] + 1
                else:
                    pass

            self.atan[smpl_idx] /= 2.0

        if (self.quant is not None) and (not self.quant.float_calc):
            self.atan[smpl_idx] = quantize("atan",
                                           self.atan[smpl_idx], self.quant,
                                           smpl_idx=smpl_idx)


class MixerClass(object):
    def __init__(self, pts, **kwargs):
        self.quant = kwargs.pop('quant', None)
        self.up = kwargs.pop('up', True)
        assert len(kwargs) == 0, \
            "unrecognized params passed in: %s" % ','.join(kwargs.keys())
        self.out = np.zeros(pts, dtype=complex)

    def execute(self, phase, signal, smpl_idx):
        pol = 2 * int(self.up) - 1
        # 2PI factor
        # self.out[smpl_idx] = np.exp(1j * pol * phase) * signal
        self.out[smpl_idx] = np.exp(1j * 2 * np.pi * pol * phase) * signal

        if (self.quant is not None) and (not self.quant.float_calc):
            self.out[smpl_idx] = quantize("mxr", self.out[smpl_idx],
                                          self.quant,
                                          smpl_idx=smpl_idx)


class ShiftRegClass(object):
    def __init__(self, sr_len=5):
        """Simple shift register with clear and shift methods
        """
        self.len = sr_len
        self.d = np.zeros(sr_len)

    def reset(self):
        self.d = np.zeros(self.len)

    def shift(self, x):
        self.d[1:self.len] = self.d[0:self.len - 1]
        self.d[0] = x


class MatchedFilterClass(object):
    def __init__(self, pts, **kwargs):
        self.n_taps = kwargs.pop('n_taps', None)
        self.quant = kwargs.pop('quant', None)
        self.sr = ShiftRegClass(self.n_taps)
        assert len(kwargs) == 0, \
            "unrecognized params passed in: %s" % ','.join(kwargs.keys())
        self.out = np.zeros(pts)

    def execute(self, x, smpl_idx):
        self.sr.shift(x)
        if smpl_idx > self.n_taps / 2:
            self.out[smpl_idx] = np.mean(self.sr.d)
        else:
            self.out[smpl_idx] = x


class FixedRateConversion(object):
    def __init__(self, num_in_pts,
                 num_out_pts,
                 ts,
                 fcw,
                 farrow_obj,
                 nco_drc_obj,
                 cpp_drc=False):
        self.num_in_pts = num_in_pts
        self.npts = np.max([num_out_pts, num_in_pts])
        self.Ts = ts
        self.fcw = fcw
        self.mu = np.zeros(self.npts)
        self.shift = np.zeros(self.npts)
        self.Ti_vec = np.zeros(self.npts)
        self.y = np.zeros(self.npts)
        self.W = np.zeros(self.npts)
        self.y = np.zeros(self.npts)
        self.farrow = farrow_obj
        self.nco_drc = nco_drc_obj
        self.cpp_dir = os.path.join("./python_tools",
                                    "systems_utils",
                                    "cpp",
                                    "trl_main")
        self.cpp_drc = cpp_drc

    def execute(self, x, dt=0):
        """Fixed data interpolation
        """
        n = 1
        m = 1
        # If FCW is 1, then bypass the DRC
        if self.fcw == 1.0:
            self.y = x
        elif self.cpp_drc is True:
            self._write_inp_file(x=x)
            self._drc_cpp()
            self._read_out_from_file()
        else:
            while m < self.num_in_pts - 8:
                self.shift[n] = (int(self.nco_drc.acc[n - 1]) - int(
                    self.nco_drc.acc[n - 2])) % 3

                m += int(self.shift[n])
                self.mu[n] = (self.nco_drc.acc[n - 1] % 1) + dt
                self.Ti_vec[n] = (m + self.mu[n]) * self.Ts

                # Farrow
                self.y[n] = self.farrow.execute(x[m - 1:m + 3],
                                                self.mu[n])

                # NCO update (fixed control word)
                self.W[n] = self.fcw
                self.nco_drc.execute(self.W[n], n)

                # Output sample index
                n += 1

    def _write_inp_file(self, x=0):
        # Assuming real signals
        with open(os.path.join(self.cpp_dir, 'trl_input.txt'), 'w') as fd:
            for val in x:
                fd.write("%f\n" % val)

    def _drc_cpp(self):

        compile_cmd = "g++ -std=c++11 "
        compile_cmd += "-o " + self.cpp_dir + "/trl_main.exe "
        compile_cmd += self.cpp_dir + "/*.cpp"
        os.system(compile_cmd)

        # Execute the CPP code
        cpp_cmd = os.path.join(self.cpp_dir, "trl_main.exe")
        kp_acq = 0.0
        ki_acq = 0.0
        kp_trk = 0.0
        ki_trk = 0.0
        shift_idx = 10  # Arbitrary shift value; it is a dont care
        p = self.fcw * 8.
        cpp_cmd = cpp_cmd + ' -ki_acq ' + str(ki_acq) + " -kp_acq " + str(kp_acq) + " -p " + str(p)
        cpp_cmd = cpp_cmd + ' -ki_trk ' + str(ki_trk) + ' -kp_trk ' + str(kp_trk) + ' -shift ' + str(shift_idx)
        cpp_cmd += ' -farrow_config ' + str(1)

        os.system(cpp_cmd)

    def _read_out_from_file(self):

        read_file = os.path.join(self.cpp_dir, "output", "farrow.txt")
        lvec = len(self.y)
        self.y = self._read_file_into_var(read_file, lvec)

    def _read_file_into_var(self, read_file, lvec):
        n = 0
        yvec = np.zeros(lvec)

        with open(read_file, 'r') as fr:
            for lf_smpl in fr:
                # print "Index n is", n
                yvec[n] = float(lf_smpl)
                n += 1
                # Vector cannot be greater than assigned size
                if n == lvec:
                    break

        return yvec

    def plot_response(self, wv_obj, ts):
        """Plot loop responses

        Args:
            wv_obj (object): Waveform object
            ts: Input (to loop) sample period

        """
        # Output waveforms
        Ts_vec = np.arange(len(wv_obj.wvfm['y_up'])) * ts
        midx = np.where(self.Ti_vec > 0)[0]
        Ti_vec = self.Ti_vec[0:midx[-1]]
        win = 20
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(Ts_vec, np.real(wv_obj.wvfm['y_up']),
                marker='o', label='Input (I)')
        ax.plot(Ts_vec, np.imag(wv_obj.wvfm['y_up']),
                marker='s', label='Input (Q)')
        ax.hold(True)
        ax.plot(Ti_vec, self.y[0:midx[-1]], linestyle='None', marker='o',
                markersize=6, label='Output')
        ax.grid(True)
        ax.hold(False)
        ax.legend()

        # NCO
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(Ti_vec, self.nco_drc.acc[0:midx[-1]], marker='o',
                linestyle='-',
                label='NCO')
        ax.hold(True)
        ax.plot(Ti_vec, self.mu[0:midx[-1]], linestyle='-', label='Mu')
        ax.plot(Ti_vec, self.W[0:midx[-1]], label='NCO FCW')
        wavg = np.convolve(self.W[0:midx[-1]], np.ones(win))[
               int(win - 1):] / float(win)
        ax.plot(Ti_vec, wavg, label='NCO FCW Avg')
        ax.grid(True)
        ax.hold(False)
        ax.legend()

        # Error
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(Ti_vec, self.ted.err[0:midx[-1]], marker='o', linestyle='-',
                label='Error')
        ax.hold(True)
        eavg = np.convolve(self.ted.err[0:midx[-1]], np.ones(win))[
               int(win - 1):] / float(
            win)
        ax.plot(Ti_vec, eavg, linestyle='-', label='Error Avg')
        ax.grid(True)
        ax.hold(False)
        ax.legend()

        plt.show()


class TrlClass(object):
    def __init__(self, num_in_pts,
                 num_out_pts,
                 config=0,
                 farrow=None,
                 nco_drc=None,
                 nco_lf=None,
                 nco_rf=None,
                 ted=None,
                 lpf_ted=None,
                 lpf_lo=None,
                 lpf_rf=None,
                 atan_lf=None,
                 atan_rf=None,
                 mxr_up=None,
                 mxr_dn=None,
                 mf_ted=None,
                 float_calc=True,
                 ):

        """TED based digital phase lock loop class

        Args:
            num_in_pts: Number of input points
            num_out_pts: Number of output points
            ts: Input sampling interval (Ts)
            nom_drc_fcw: Nominal DRC NCO control word (Ti_nom/Ts)
            farrow: FarrowClass object
            nco_drc: NcoClass object
            ted: (optional) TedClass object if not open_loop
            lpf: (optional) LoopFilterClass object if not open loop
            open_loop: (optional) True=NCO input will be set to nom_fcw
                                  False=NCO input gets LPF output

        Returns:

        """
        # input args
        self.num_in_pts = num_in_pts
        self.npts = np.max([num_out_pts, num_in_pts])
        self.num_out_pts = num_out_pts
        # TODO:
        self.Ts = config.Ts
        self.mval = (1 if config.link_dict['mod_type']=='3b4b_baseband' else config.link_dict['m'])
        self.nom_lf_fcw = config.nom_lf_fcw
        self.open_loop = config.trl_dict['open_loop']
        self.float_calc = float_calc
        self.farrow_init_config = config.trl_dict['farrow_config']
        self.blf_est = config.link_dict['blf_est']
        self.blf_err = config.link_dict['blf_err']
        #TODO:
        self.nom_drc_fcw = config.nom_drc_fcw / (1. + self.blf_est)

        # #TODO:
        # if config.link_dict['fsample_hz'] == 20.48e6:
        #     self.Ts = config.Ts * 4
        #     self.nom_drc_fcw = (.25 * config.nom_drc_fcw) / (1. + self.blf_est)

        if config.link_dict['mod_type'] in ['3b4b_baseband', 'fm0']:
            self.baseband = 1
        else:
            self.baseband = 0

        self.cpp_dir = config.cpp_dir
        self.cpp_sim = config.cpp_sim
        self.compile_cpp = config.compile_cpp
        self.cpp_dump_all = config.trl_plot_fig

        # internal signal initialization
        self.mu = np.zeros(self.num_out_pts)
        self.shift = np.zeros(self.num_out_pts)
        self.Ts_vec = None
        self.Ti_vec = np.zeros(self.num_out_pts)
        self.y = np.zeros(self.num_out_pts)
        self.W_drc = np.zeros(self.num_out_pts)
        self.W_lo = np.zeros(self.num_out_pts)
        self.W_rf = np.zeros(self.num_out_pts)
        self.y_up = np.zeros(self.num_out_pts, dtype='complex')
        self.y_dn = np.zeros(self.num_out_pts, dtype='complex')
        self.y_lf = np.zeros(self.num_out_pts, dtype='complex')
        self.y_rf = np.zeros(self.num_out_pts, dtype='complex')
        self.lpf_lo_out = np.zeros(self.num_out_pts)
        self.lpf_rf_out = np.zeros(self.num_out_pts)
        self.lpf_ted_out = np.zeros(self.num_out_pts)
        self.kp_vec = np.zeros(self.num_out_pts)
        self.ki_vec = np.zeros(self.num_out_pts)

        # loop objects
        self.farrow = farrow
        self.nco_drc = nco_drc
        self.nco_lf = nco_lf
        self.nco_rf = nco_rf
        self.ted = ted
        # self.lpf_ted = lpf_ted
        self.lpf_ted = LoopFilterClass(config=config, mode='drc')
        # self.lpf_lo = lpf_lo
        # self.lpf_rf = lpf_rf
        self.lpf_lo = None
        self.lpf_rf = None
        self.atan_lf = atan_lf
        self.atan_rf = atan_rf
        self.mxr_up = mxr_up
        self.mxr_dn = mxr_dn
        self.mf_ted = mf_ted

        self.basepoint_reg = np.zeros(4)  # 4 element SR for farrow input

    def execute(self, x_up, x_dn):
        """Process input samples x through TED PLL

        Args:
            x : Vector of input samples

        The input data x is shifted through the Farrow interpolator structure
        as shown below:

                              Interp
                              Interval
                             |<----->|

        [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], ...]
                      |      |     |     |
                     m-1     m    m+1    m+2

         The 4-point vector m-1:m+2 is reversed inside the farrow

        """

        if self.farrow_init_config:
            n = 0
            m = 1
        else:
            n = 0
            m = 0
        dly = 0

        # TODO: Right now, this handles only real inputs complex needs to be
        # handled
        if self.cpp_sim:
            self._clean_up_dir()
            self._write_inp_to_file(x_up)
            self._execute_cpp()
            self._read_out_from_file()

            if self.cpp_dump_all:
                mvec = np.cumsum(self.shift)
                self.Ti_vec = (mvec + self.mu ) * self.Ts
                self.Ts_vec = np.arange(len(self.y_lf)) * self.Ts
                midx = np.where(self.Ti_vec > 0)[0]
                self.Ti_vec = self.Ti_vec[0:midx[-1]]
                self.W_drc = self.nom_drc_fcw + self.lpf_ted_out
            return

        # Each iteration corresponds to an input sample.  The loop must stop
        # when we have used all the input data (m index)
        while m < self.num_in_pts - 5 and n < self.num_out_pts:
            # Update the Farrow control based on previous NCO output
            #    m  = basepoint set
            #    mu = interpolant value
            # print "LPF gains ", self.lpf_ted.kp, self.lpf_ted.ki
            self.shift[n] = (int(self.nco_drc.acc[n - 1 - dly]) -
                             int(self.nco_drc.acc[n - 2 - dly])) % 4


            m += self.shift[n]
            self.mu[n] = self.nco_drc.acc[n - 1 - dly] % 1
            self.Ti_vec[n] = (m + self.mu[n]) * self.Ts

            self._shift_data_into_reg(x_up, m, n)

            # DRC up
            if self.farrow_init_config:
                self.y_up[n] = self.farrow.execute(x_up[int(m - 1):int(m + 3)],
                                                   self.mu[n],
                                                   smpl_idx=n)
            else:
                self.y_up[n] = self.farrow.execute(self.basepoint_reg,
                                                   self.mu[n],
                                                   smpl_idx=n)

            # DRC down (not used in FM0)
            # if self.mval == 1:
            if self.baseband:
                self.y_dn[n] = 0.
            else:
                self.y_dn[n] = self.farrow.execute(x_dn[int(m - 1):int(m + 3)],
                                                   self.mu[n],
                                                   smpl_idx=n).copy()

            # LO mixers
            up_phs = (self.nco_lf.acc[n - 1 - dly] -
                      self.nco_rf.acc[n - 1 - dly]) % 1
            self.mxr_up.execute(up_phs, self.y_up[n], n)
            dn_phs = (self.nco_lf.acc[n - 1 - dly] +
                      self.nco_rf.acc[n - 1 - dly]) % 1
            self.mxr_dn.execute(dn_phs, self.y_dn[n], n)

            # Conjugate sum up and down paths
            # if self.mval == 1:
            if self.baseband:
                self.y_lf[n] = np.conjugate(self.mxr_up.out[n])
            else:
                self.y_lf[n] = (np.conjugate(self.mxr_up.out[n]) +
                                self.mxr_dn.out[n]) / 2.

            # Sum up and down paths
            # if self.mval == 1:
            if self.baseband:
                self.y_rf[n] = self.mxr_up.out[n]
            else:
                self.y_rf[n] = (self.mxr_up.out[n] + self.mxr_dn.out[n]) / 2.

            # Compute the LO arctan()
            self.atan_lf.execute(np.real(self.y_lf[n]),
                                 np.imag(self.y_lf[n]), n)

            # Compute the RF arctan()
            self.atan_rf.execute(np.real(self.y_rf[n]),
                                 np.imag(self.y_rf[n]), n)

            # self.mf_ted.execute(np.real(self.y_up[n]), n)
            self.mf_ted.execute(np.real(self.y_lf[n]), n)

            # TED
            # if self.mval == 1: # or self.ted_enable:
            if n <= self.lpf_ted.shift_idx:
                self.kp_vec[n] = self.lpf_ted.kp_acq;
                self.ki_vec[n] = self.lpf_ted.ki_acq;
            else:
                self.kp_vec[n] = self.lpf_ted.kp_trk;
                self.ki_vec[n] = self.lpf_ted.ki_trk;
            if self.baseband:
                # self.ted.execute(np.real(self.y_up[n]), n)
                self.ted.execute(self.mf_ted.out[n], n)
                self.lpf_ted_out[n] = self.lpf_ted.execute(self.ted.err[n], n)

                # TED bandwidth gear shifting
                # if self.lpf_ted.shift_idx is not None:
                #     if n == self.lpf_ted.shift_idx:
                #         self.lpf_ted.set_gain(self.lpf_ted.BnT_shift)

            # LO LPF output
            # self.lpf_lo_out[n] = self.lpf_lo.execute(self.atan_lf.atan[n])

            # RF LPF output
            # self.lpf_rf_out[n] = self.lpf_rf.execute(self.atan_rf.atan[n])

            # LO NCO
            # if self.mval == 1:
            if self.baseband:
                self.nco_lf.acc[n] = 0
            else:
                if False:
                    self.W_lo[n] = self.mval / 8. * self.lpf_lo_out[n]
                else:
                    self.W_lo[n] = (self.mval / 8.) * (self.nom_lf_fcw +
                                                       self.lpf_lo_out[n])
                self.nco_lf.execute(self.W_lo[n], n)

            # RF NCO
            # self.W_rf[n] = self.mval / 8. * self.lpf_rf_out[n]
            # self.nco_rf.execute(self.W_rf[n], n)

            # DRC NCO
            # if self.mval == 1:
            if self.baseband:
                if self.open_loop:
                    self.W_drc[n] = self.nom_drc_fcw
                else:
                    self.W_drc[n] = self.nom_drc_fcw + self.lpf_ted_out[n]
            else:
                if self.open_loop:
                    self.W_drc[n] = self.nom_drc_fcw * (
                        1.0 - (self.nom_lf_fcw))
                else:
                    self.W_drc[n] = self.nom_drc_fcw * (
                    1.0 - (self.lpf_lo_out[n] +
                           self.nom_lf_fcw))

            self.nco_drc.execute(self.W_drc[n], n)
            # if self.ted_enable:
            #     self.nco_drc.acc[n] += self.lpf_ted_out[n]
                # self.nco_lf.acc[n] += self.lpf_ted_out[n]
            #TODO: print this output
            # Output sample index
            n += 1

        # print("The length of y_up and y_dn of the DRC is ", len(x_up))
        # plt.plot(np.real(self.y_up), np.real(self.y_dn), ".-")
        # plt.title("y_up and y_dn of the DRC")
        # plt.figure()
        #
        # print("The length of y_lf is ", len(self.y_lf))
        # plt.plot(np.real(self.y_lf), ".-")
        # plt.title("y_lf")
        # plt.figure()
        #
        # print("The length of y_rf is ", len(self.y_rf))
        # plt.plot(np.real(self.y_rf), ".-")
        # plt.title("y_rf")
        # plt.figure()
        #
        # print("The length of the TED outputs are ", len(self.kp_vec), len(self.ki_vec))
        # plt.plot(np.real(self.kp_vec), np.real(self.ki_vec), ".-")
        # plt.title("TED outputs")
        # plt.figure()
        #
        # print("The length of the NCO input is ", len(self.W_lo))
        # plt.plot(np.real(self.W_lo), ".-")
        # plt.title("NCO inputs")
        # plt.figure()
        #
        # print("The length of the NCO output is ", len(self.W_drc))
        # plt.plot(np.real(self.W_drc), ".-")
        # plt.title("NCO outputs")
        # plt.figure()

        # print "Total samples = {0}".format(n)
        # Update time vectors for plotting
        self.Ts_vec = np.arange(len(self.y_up)) * self.Ts
        midx = np.where(self.Ti_vec > 0)[0]
        self.Ti_vec = self.Ti_vec[0:midx[-1]]

        i = np.mean(np.real(self.y_lf[80 - 8:80])),
        q = np.mean(np.imag(self.y_lf[80 - 8:80])),

    def _shift_data_into_reg(self, x_up, m, n):
        shift = int(self.shift[n])
        self.basepoint_reg = np.roll(self.basepoint_reg, -shift)
        for i in np.arange(shift):
            self.basepoint_reg[int(4 + i - shift)] = x_up[int(m - shift + i)]

    def _clean_up_dir(self):

        file_list = ['farrow.txt', 'mu.txt', 'nco.txt', 'mf.txt', 'shift.txt',
                     'signal.txt', 'ted.txt', 'lpf.txt']

        for name in file_list:
            fname = os.path.join(self.cpp_dir, 'output', name)
            if os.path.isfile(fname):
                os.system("rm " + fname)

    def _write_inp_to_file(self, x=0):

        # Assuming real signals
        with open(os.path.join(self.cpp_dir, 'trl_input.txt'), 'w') as fd:
            for val in x:
                fd.write("%f\n" % val)

    def _execute_cpp(self):
        # Compile the CPP code
        if self.compile_cpp:
            compile_cmd = "g++ -std=c++11 "
            compile_cmd += "-o " + self.cpp_dir + "/trl_main.exe "
            compile_cmd += self.cpp_dir + "/*.cpp"
            os.system(compile_cmd)

        # Execute the CPP code
        cpp_cmd = os.path.join(self.cpp_dir, "trl_main.exe")
        # kp_acq = self.lpf_ted.kp
        # ki_acq = self.lpf_ted.ki
        # kp_acq = 0.097
        # kp_acq = 0.2
        kp_acq = self.lpf_ted.kp_acq
        # ki_acq = 0.001
        # ki_acq = 0.001
        ki_acq = self.lpf_ted.ki_acq
        # self.lpf_ted.set_gain(self.lpf_ted.BnT_shift)
        # kp_trk = self.lpf_ted.kp
        # ki_trk = self.lpf_ted.ki
        # kp_trk = 0.2
        kp_trk = self.lpf_ted.kp_trk
        # ki_trk = 0.0001
        ki_trk = self.lpf_ted.ki_trk
        shift_idx = self.lpf_ted.shift_idx
        p = 1./self.Ts
        cpp_cmd = cpp_cmd + ' -ki_acq ' + str(ki_acq) + " -kp_acq " + str(kp_acq) + " -p " + str(p)
        cpp_cmd = cpp_cmd + ' -ki_trk ' + str(ki_trk) + ' -kp_trk ' + str(kp_trk) + ' -shift ' + str(shift_idx)
        cpp_cmd += ' -b ' + str(self.blf_est)
        cpp_cmd += ' -bact ' + str(self.blf_err)

        if self.farrow_init_config:
            cpp_cmd += ' -farrow_config ' + str(1)

        if self.cpp_dump_all:
            cpp_cmd += ' -a'
        args = shlex.split(cpp_cmd)
        subprocess.check_call(args)
        # os.system(cpp_cmd)

    def _read_out_from_file(self):

        read_file = os.path.join(self.cpp_dir, "output", "farrow.txt")
        lvec = len(self.y_lf)
        self.y_lf = self._read_file_into_var(read_file, lvec)

        if self.cpp_dump_all:

            read_file = os.path.join(self.cpp_dir, "output", "mu.txt")
            self.mu = self._read_file_into_var(read_file, lvec)

            read_file = os.path.join(self.cpp_dir, "output", "shift.txt")
            self.shift = self._read_file_into_var(read_file, lvec)

            read_file = os.path.join(self.cpp_dir, "output", "mf.txt")
            self.mf_ted.out = self._read_file_into_var(read_file, lvec)

            read_file = os.path.join(self.cpp_dir, "output", "ted.txt")
            self.ted.err = self._read_file_into_var(read_file, lvec)

            read_file = os.path.join(self.cpp_dir, "output", "lpf.txt")
            self.lpf_ted_out = self._read_file_into_var(read_file, lvec)

            read_file = os.path.join(self.cpp_dir, "output", "nco.txt")
            self.nco_drc.acc = self._read_file_into_var(read_file, lvec)

            read_file = os.path.join(self.cpp_dir, "output", "kp.txt")
            self.kp_vec = self._read_file_into_var(read_file, lvec)

            read_file = os.path.join(self.cpp_dir, "output", "ki.txt")
            self.ki_vec = self._read_file_into_var(read_file, lvec)

    def _read_file_into_var(self, read_file, lvec):
        n = 0
        yvec = np.zeros(lvec)

        with open(read_file, 'r') as fr:
            for lf_smpl in fr:
                # print "Index n is", n
                yvec[n] = float(lf_smpl)
                n += 1
                # Vector cannot be greater than assigned size
                if n == lvec:
                    break

        return yvec

    def make_plot(self, ax, trace_list):
        """Create a plot given handle ax and trace list
        """
        ax.hold(True)
        for trace in trace_list:
            ax.plot(trace.x, trace.y,
                    marker=trace.marker,
                    markersize=trace.markersize,
                    linestyle=trace.linestyle,
                    label=trace.label,
                    )
        ax.grid(True)
        ax.hold(False)
        ax.legend()
        if trace.title:
            ax.set_title(trace.title)

    class Trace(object):
        """Container of parameters for single plot trace
        """

        def __init__(self, x, y,
                     marker='.',
                     linestyle='-',
                     markersize=4,
                     label=None,
                     title=None):
            self.x = x
            self.y = y
            self.marker = marker
            self.linestyle = linestyle
            self.markersize = markersize
            self.label = label
            self.title = title

    def plot_inputs_and_outputs(self, wv_obj, ddc, save_figs):
        """Plot loop input and output waveforms
        """
        midx = len(self.Ti_vec)
        #
        # Input waveforms
        #
        imax = np.min([len(self.Ts_vec), len(ddc.ddc_up_decimate)])
        base_str = 'IO'
        ebno_str = '{0:6.2f}'.format(wv_obj.EbNo_dB)
        blf_err_str = '{0:4.2f}'.format(wv_obj.blf_err)
        rf_str = '{0:6.2f}'.format(wv_obj.theta_rf_rad * 180. / np.pi)
        tstr = ''.join([base_str,
                        '(BLFerr%=',
                        blf_err_str,
                        ',EbNo=',
                        ebno_str,
                        ',RFdeg=',
                        rf_str,
                        ',pdiv8=',
                        '{0:6.3f}'.format(self.nom_drc_fcw),
                        ',M=',
                        '{0:2d}'.format(self.mval),
                        ')'])

        trace_list = [self.Trace(# x=self.Ts_vec[0:imax],
                                 x=np.arange(imax),
                                 y=np.real(ddc.ddc_up_decimate[0:imax]),
                                 label='Input Up (I)',
                                 marker='o',
                                 markersize=6),
                      self.Trace(#x=self.Ts_vec[0:imax],
                                 x=np.arange(imax),
                                 y=np.imag(ddc.ddc_up_decimate[0:imax]),
                                 label='Input Up (Q)',
                                 marker='.',
                                 markersize=6),
                      self.Trace(# x=self.Ts_vec[0:imax],
                                 x=np.arange(imax),
                                 y=np.real(ddc.ddc_dn_decimate[0:imax]),
                                 label='Input Dn (I)',
                                 marker='.'),
                      self.Trace(# x=self.Ts_vec[0:imax],
                                 x=np.arange(imax),
                                 y=np.imag(ddc.ddc_dn_decimate[0:imax]),
                                 label='Input Dn (Q)',
                                 marker='.',
                                 title=tstr),
                      ]
        fig = plt.figure()
        # ax = fig.add_subplot(3, 1, 1)
        ax = fig.add_subplot(4, 1, 1)

        self.make_plot(ax, trace_list=trace_list)
        #
        # y_lf output waveforms (and decision points)
        #
        trace_list = [self.Trace(# x=self.Ti_vec, y=np.real(self.y_lf[0:midx]),
                                 x=np.arange(midx),y=np.real(self.y_lf[0:midx]),
                                 label='y_lf (I)',
                                 marker='o',
                                 markersize=6),
                      self.Trace(# x=self.Ti_vec, y=np.imag(self.y_lf[0:midx]),
                                 x=np.arange(midx), y=np.imag(self.y_lf[0:midx]),
                                 label='y_lf (Q)',
                                 markersize=6),
                      # self.Trace(x=self.Ti_vec[0::4],
                      # y=np.real(self.y_lf[0:midx:4]),
                      #           label='Decision Points',
                      #           marker='s',
                      #           linestyle='None',
                      #           markersize=6),
                      ]
        # ax1 = fig.add_subplot(3, 1, 2, sharex=ax)
        ax1 = fig.add_subplot(4, 1, 2, sharex=ax)
        self.make_plot(ax1, trace_list=trace_list)
        #
        # y_rf output waveforms
        #
        trace_list = [self.Trace(# x=self.Ti_vec, y=np.real(self.y_rf[0:midx]),
                                 x=np.arange(midx), y=np.real(self.y_rf[0:midx]),
                                 label='y_rf (I)'),
                      self.Trace(# x=self.Ti_vec, y=np.imag(self.y_rf[0:midx]),
                                 x=np.arange(midx), y=np.imag(self.y_rf[0:midx]),
                                 label='y_rf (Q)'),
                      ]
        # ax2 = fig.add_subplot(3, 1, 3, sharex=ax)
        ax2 = fig.add_subplot(4, 1, 3, sharex=ax)
        self.make_plot(ax2, trace_list=trace_list)

        trace_list = [self.Trace(# x=self.Ti_vec, y=self.kp_vec[0:midx],
                                 x=np.arange(midx), y=self.kp_vec[0:midx],
                                 label='kp'),
                      self.Trace(# x=self.Ti_vec, y=self.ki_vec[0:midx],
                                 x=np.arange(midx), y=self.ki_vec[0:midx],
                                 label='ki'),
                      ]

        # ax2 = fig.add_subplot(3, 1, 3, sharex=ax)
        ax3 = fig.add_subplot(4, 1, 4, sharex=ax)
        self.make_plot(ax3, trace_list=trace_list)  #
        # Tx Data
        #
        # vec_len = len(wv_obj.wvfm['source_2blf']) - 4
        # trace_list = [self.Trace(x=np.arange(vec_len)/float(vec_len) * np.max(self.Ti_vec),  # nopep8
        #                          y=wv_obj.wvfm['source_2blf'][0:vec_len],
        #                          label='TX DATA'),
        #               ]
        # ax3 = fig.add_subplot(4, 1, 4)
        # self.make_plot(ax3, trace_list=trace_list)

        if save_figs:
            basepath = "/home/ksundstr/Documents/yk_design/yk_modem/docs/yk_rx"
            pngpath = "yk_rx_trl_pngs"
            fname = tstr.replace(' ', '_')
            fname = fname.replace('.', 'p')
            fname = fname.replace(',', '_')
            fname = fname.replace('(', '_')
            fname = fname.replace(')', '_')
            fname = fname.replace('%', '_')
            plt.savefig('/'.join([basepath, pngpath, fname]))

    def plot_drc(self, wv_obj, save_figs):
        """Plot DRC waveforms
        """
        midx = len(self.Ti_vec)
        win = 20
        base_str = 'DRC'
        ebno_str = '{0:6.2f}'.format(wv_obj.EbNo_dB)
        blf_err_str = '{0:4.2f}'.format(wv_obj.blf_err)
        rf_str = '{0:6.2f}'.format(wv_obj.theta_rf_rad * 180. / np.pi)
        tstr = ''.join([base_str,
                        '(BLFerr%=',
                        blf_err_str,
                        ',EbNo=',
                        ebno_str,
                        ',RFdeg=',
                        rf_str,
                        ',pdiv8=',
                        '{0:6.3f}'.format(self.nom_drc_fcw),
                        ',M=',
                        '{0:2d}'.format(self.mval),
                        ')'])

        plt_lpf = False
        if plt_lpf:
            nplts = 5
        else:
            nplts = 4
        #
        # Accumulator
        #
        trace_list = [self.Trace(# x=self.Ti_vec, y=self.nco_drc.acc[0:midx],
                                 x=np.arange(midx), y=self.nco_drc.acc[0:midx],
                                 label='NCO DRC',
                                 title=tstr),
                      ]
        fig = plt.figure()
        ax = fig.add_subplot(nplts, 1, 1)
        self.make_plot(ax, trace_list=trace_list)
        #
        # SHIFT
        trace_list = [self.Trace(# x=self.Ti_vec, y=self.shift[0:midx],
                                 x=np.arange(midx), y=self.shift[0:midx],
                                 label='SHIFT'),
                      ]
        ax1 = fig.add_subplot(nplts, 1, 2, sharex=ax)
        self.make_plot(ax1, trace_list=trace_list)
        #
        # MU
        #
        trace_list = [self.Trace(# x=self.Ti_vec, y=self.mu[0:midx],
                                 x=np.arange(midx), y=self.mu[0:midx],
                                 label='MU'),
                      ]
        ax1 = fig.add_subplot(nplts, 1, 3, sharex=ax)
        self.make_plot(ax1, trace_list=trace_list)
        #
        # FCW
        #
        avg = np.convolve(self.W_drc[0:midx], np.ones(win))[
              int(win - 1):] / float(win)
        trace_list = [self.Trace(# x=self.Ti_vec, y=self.W_drc[0:midx],
                                 x=np.arange(midx), y=self.W_drc[0:midx],
                                 label='FCW DRC'),
                      self.Trace(# x=self.Ti_vec, y=avg, label='FCW DRC (Avg)'
                                 x=np.arange(len(avg)), y=avg, label='FCW DRC (Avg)'
                                 )
                      ]
        ax2 = fig.add_subplot(nplts, 1, 4, sharex=ax)
        self.make_plot(ax2, trace_list=trace_list)
        #
        # LPF
        #
        if plt_lpf:
            trace_list = [self.Trace(x=self.Ti_vec, y=self.lpf_ted_out[0:midx],
                                     label='LPF_TED'),
                          ]
            ax1 = fig.add_subplot(nplts, 1, 5, sharex=ax)
            self.make_plot(ax1, trace_list=trace_list)

        if save_figs:
            basepath = "/home/ksundstr/Documents/yk_design/yk_modem/docs/yk_rx"
            pngpath = "yk_rx_trl_pngs"
            fname = tstr.replace(' ', '_')
            fname = fname.replace('.', 'p')
            fname = fname.replace(',', '_')
            fname = fname.replace('(', '_')
            fname = fname.replace(')', '_')
            fname = fname.replace('%', '_')
            plt.savefig('/'.join([basepath, pngpath, fname]))

    def plot_lo(self, wv_obj, save_figs):
        """Plot LO waveforms
        """
        base_str = 'LO'
        ebno_str = '{0:6.2f}'.format(wv_obj.EbNo_dB)
        blf_err_str = '{0:4.2f}'.format(wv_obj.blf_err)
        rf_str = '{0:6.2f}'.format(wv_obj.theta_rf_rad * 180. / np.pi)
        tstr = ''.join([base_str,
                        '(BLFerr%=',
                        blf_err_str,
                        ',EbNo=',
                        ebno_str,
                        ',RFdeg=',
                        rf_str,
                        ',pdiv8=',
                        '{0:6.3f}'.format(self.nom_drc_fcw),
                        ',M=',
                        '{0:2d}'.format(self.mval),
                        ')'])

        midx = len(self.Ti_vec)
        win = 20

        fig = plt.figure()
        ax = fig.add_subplot(3, 1, 1)
        #
        # LO NCO
        #
        trace_list = [self.Trace(x=self.Ti_vec, y=self.nco_lf.acc[0:midx],
                                 label='NCO LO',
                                 title=tstr),
                      ]
        self.make_plot(ax, trace_list=trace_list)
        #
        # LO ATAN
        #
        ax1 = fig.add_subplot(3, 1, 2, sharex=ax)

        # avg = np.convolve(self.atan_lo[0:midx], np.ones(win))[
        avg = np.convolve(self.atan_lf.atan[0:midx], np.ones(win))[
              int(win - 1):] / float(win)
        trace_list = [self.Trace(x=self.Ti_vec,
                                 # y=self.atan_lo[0:midx],
                                 y=self.atan_lf.atan[0:midx],
                                 label='ATAN LO'),
                      self.Trace(x=self.Ti_vec, y=avg,
                                 label='ATAN LO (Avg)')
                      ]
        self.make_plot(ax1, trace_list=trace_list)
        #
        # LO LPF
        #
        ax2 = fig.add_subplot(3, 1, 3, sharex=ax)
        avg = np.convolve(self.lpf_lo_out[0:midx], np.ones(win))[
              int(win - 1):] / float(win)
        trace_list = [self.Trace(x=self.Ti_vec, y=self.lpf_lo_out[0:midx],
                                 label='LO LPF Output'),
                      self.Trace(x=self.Ti_vec, y=avg,
                                 label='LO LPF Output (Avg)')
                      ]
        self.make_plot(ax2, trace_list=trace_list)

        if save_figs:
            basepath = "/home/ksundstr/Documents/yk_design/yk_modem/docs/yk_rx"
            pngpath = "yk_rx_trl_pngs"
            fname = tstr.replace(' ', '_')
            fname = fname.replace('.', 'p')
            fname = fname.replace(',', '_')
            fname = fname.replace('(', '_')
            fname = fname.replace(')', '_')
            fname = fname.replace('%', '_')
            plt.savefig('/'.join([basepath, pngpath, fname]))

    def plot_rf(self, wv_obj, save_figs):
        """Plot RF waveforms
        """
        base_str = 'RF'
        ebno_str = '{0:6.2f}'.format(wv_obj.EbNo_dB)
        blf_err_str = '{0:4.2f}'.format(wv_obj.blf_err)
        rf_str = '{0:6.2f}'.format(wv_obj.theta_rf_rad * 180. / np.pi)

        tstr = ''.join([base_str,
                        '(BLFerr%=',
                        blf_err_str,
                        ',EbNo=',
                        ebno_str,
                        ',RFdeg=',
                        rf_str,
                        ',pdiv8=',
                        '{0:6.3f}'.format(self.nom_drc_fcw),
                        ',M=',
                        '{0:2d}'.format(self.mval),
                        ')'])

        midx = len(self.Ti_vec)
        win = 20

        fig = plt.figure()
        ax = fig.add_subplot(3, 1, 1)
        #
        # RF NCO
        #
        trace_list = [self.Trace(x=self.Ti_vec, y=self.nco_rf.acc[0:midx],
                                 label='NCO RF',
                                 title=tstr),
                      ]
        self.make_plot(ax, trace_list=trace_list)
        #
        # LO ATAN
        #
        ax1 = fig.add_subplot(3, 1, 2, sharex=ax)

        # avg = np.convolve(self.atan_rf[0:midx], np.ones(win))[
        avg = np.convolve(self.atan_rf.atan[0:midx], np.ones(win))[
              int(win - 1):] / float(win)
        trace_list = [self.Trace(x=self.Ti_vec,
                                 # y=self.atan_rf[0:midx],
                                 y=self.atan_rf.atan[0:midx],
                                 label='ATAN RF'),
                      self.Trace(x=self.Ti_vec, y=avg,
                                 label='ATAN RF (Avg)')
                      ]
        self.make_plot(ax1, trace_list=trace_list)
        #
        # LO LPF
        #
        ax2 = fig.add_subplot(3, 1, 3, sharex=ax)
        avg = np.convolve(self.lpf_rf_out[0:midx], np.ones(win))[
              int(win - 1):] / float(win)
        trace_list = [self.Trace(x=self.Ti_vec, y=self.lpf_rf_out[0:midx],
                                 label='RF LPF Output'),
                      self.Trace(x=self.Ti_vec, y=avg,
                                 label='RF LPF Output (Avg)')
                      ]
        self.make_plot(ax2, trace_list=trace_list)

        if save_figs:
            basepath = "/home/ksundstr/Documents/yk_design/yk_modem/docs/yk_rx"
            pngpath = "yk_rx_trl_pngs"
            fname = tstr.replace(' ', '_')
            fname = fname.replace('.', 'p')
            fname = fname.replace(',', '_')
            fname = fname.replace('(', '_')
            fname = fname.replace(')', '_')
            fname = fname.replace('%', '_')
            plt.savefig('/'.join([basepath, pngpath, fname]))

    def plot_stuff(self):
        midx = len(self.Ti_vec)
        win = 200

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        tmp = np.convolve(np.ones(win), self.nco_lf.acc) / float(win)
        trace_list = [self.Trace(x=self.Ti_vec, y=tmp[0:midx],
                                 label='NCO - W',
                                 title='LO'),
                      ]
        self.make_plot(ax, trace_list=trace_list)
