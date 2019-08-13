"""rx_waveforms.py

This module contains a number of helper functions used to generate random data
and encode as RFID reverse link waveforms.

"""
from collections import namedtuple
import numpy as np
import scipy.signal as ss
import matplotlib.pylab as plt

from systems_r700.model.src.tag.python_tools.systems_utils.rate_conversion import NcoClass
from systems_r700.model.src.tag.python_tools.systems_utils.rate_conversion import FarrowClass
from systems_r700.model.src.tag.python_tools.systems_utils.rate_conversion import FixedRateConversion
from systems_r700.model.src.tag.python_tools.systems_utils.ber import get_noise_std_deviation

class FilterBaseClass(object):
    def __init__(self):
        pass

    def _get_coefficients(self):
        self.h = np.array([1])

    def get_frequency_response(self):
        self.w, self.H = ss.freqz(self.h)

    def filter_data(self, x):
        return np.convolve(x, self.h)[int(self.N / 2.0) - 1:int(-self.N / 2.0)]


class RaisedCosineFilterClass(FilterBaseClass):
    def __init__(self, alpha=0.98, N=80, T=1, fs=10, mval=1):
        """Raised Cosine Filter (LPF or BPF based on M value)"""
        FilterBaseClass.__init__(self)
        self.alpha = alpha
        self.N = N
        self.T = T
        self.fs = fs
        self.mval = mval
        self.t = np.arange(-N / 2., N / 2.) / fs
        self.h = None
        self.H = None
        self.w = None

        self._get_coefficients()

    def _get_coefficients(self):
        # raised cosine impulse response
        num = np.sinc(self.t / self.T) * np.cos(np.pi * self.alpha *
                                                self.t / self.T)
        den = 1 - (2 * self.alpha * self.t / self.T) ** 2
        den = [1 if i == 0 else i for i in den]
        self.h = num / den / self.mval

        # set dc gain to zero
        self.h /= np.sum(self.h)

        # modulate to BLF for miller signals
        if self.mval > 1:
            self.h *= np.cos(2 * np.pi * self.mval * self.t)


class FirFilterClass(FilterBaseClass):
    def __init__(self, ntaps=512, fcut=0.5, pass_zero=True):
        """FIR Filter (LPF or BPF based on pass_zero value)"""
        FilterBaseClass.__init__(self)
        self.N = ntaps
        self.fcut = fcut
        self.h = None
        self.H = None
        self.w = None
        self.pass_zero = pass_zero

        self._get_coefficients()

    def _get_coefficients(self):
        self.h = ss.firwin(numtaps=self.N,
                           cutoff=self.fcut,
                           pass_zero=self.pass_zero)


class RandomBitGenerator(object):
    def __init__(self, num_bits, antipodal=False):
        """Return a vector of binary or antipodal random data.
        """
        self.num_bits = num_bits
        self.data = self.__random_bit_generator(antipodal)

    def __random_bit_generator(self, antipodal):
        """Return random data from uniform distribution
        """
        data = np.round(np.random.uniform(0, 1, self.num_bits))
        if antipodal:
            return 2 * data - 1
        else:
            return data


class BasicStateClass(object):
    def __init__(self, state_name, next_state, symbol):
        """Modulation state container: state name, next state, output symbol

        Args:
            state_name: Corresponding to Gen2 specification, eg. "S1"
            next_state: List of two elements [next state if b_in=0,
                                              next state if b_in = 1]
            symbol: List of two feature samples, e.g. [1, -1], that compose
                     symbol

        """
        self._name = state_name
        self._next_state = next_state
        self._symbol = symbol

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def next_state(self):
        return self._next_state

    @next_state.setter
    def next_state(self, value):
        self._next_state = value

    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, value):
        self._symbol = value


class EncoderFsmClass(object):
    def __init__(self, state_list, state):
        """Encoder state machine for reverse link modulation

        Args:
            state_list: A list of BasicStateClass objects
            state: List index for start state

        """
        self.state_list = state_list
        self.state = self.state_list[state - 1]

    def get_output_symbol(self):
        """Return the symbol, sampled at 2 smpls/sym, to be output
        """
        return self.state.symbol

    def move_to_next_state(self, b_in):
        """Transition to next state based on b_in

        Args:
            b_in (int): Input bit value 0/1
        """
        self.state = self.state_list[self.state.next_state[int(b_in)] - 1]


def symbol_encoder(bits, encode):
    """Generator that yields encoded symbols

    Args:
        bits: Vector if binary intput bits
        encode: One of "fm0", "miller", "square", or "bpsk"

    Returns:
        out: Feature samples, e.g. [1, -1] for each encoded symbol

    """
    # Define all reverse link modulation state lists.
    # Each state list is composed of BasicStateClass objects.
    fm0_states = [BasicStateClass("S1", [3, 4], [1, 1]),
                  BasicStateClass("S2", [2, 1], [1, -1]),
                  BasicStateClass("S3", [3, 4], [-1, 1]),
                  BasicStateClass("S4", [2, 1], [-1, -1])]

    miller_states = [BasicStateClass("S1", [4, 2], [1, 1]),
                     BasicStateClass("S2", [4, 3], [1, -1]),
                     BasicStateClass("S3", [1, 2], [-1, 1]),
                     BasicStateClass("S4", [1, 3], [-1, -1])]

    square_states = [BasicStateClass("S1", [1, 1], [1, -1])]

    bpsk_states = [BasicStateClass("S1", [1, 2], [-1, -1]),
                   BasicStateClass("S2", [1, 2], [1, 1])]

    # Create the encoder generator
    if encode == "fm0":
        enc = EncoderFsmClass(fm0_states, 2)
    elif encode == "miller":
        enc = EncoderFsmClass(miller_states, 2)
    elif encode == "square":
        enc = EncoderFsmClass(square_states, 1)
    elif encode == "bpsk":
        enc = EncoderFsmClass(bpsk_states, 2)
    else:
        raise ValueError('Unknown encoding type {0:s}.'.format(encode))

    # Iterate over the generator object
    for b in bits:
        enc.move_to_next_state(b)
        out = enc.get_output_symbol()
        # print b, enc.state.name, out
        yield out[0]
        yield out[1]


def create_symbol_sequence(bits, encode='fm0'):
    """Return symbol sequence sampled twice per symbol

    Args:
        bits:  Array of input bits
        encode: One of 'fm0', 'miller', 'square', or 'bpsk'

    Returns:
        x: Array of symbols sampled twice per symbol

    """
    x = np.zeros(2 * len(bits))
    # create the symbol generator object
    symbol_generator = symbol_encoder(bits, encode=encode)
    # iterate over the generator
    for k, s in enumerate(symbol_generator):
        x[k] = s
    return x


def zero_stuff(x, up_factor):
    """
    Insert k-1 zeros between each sample in x.

    Args:
        x: Array to be zero-stuffed
        up_factor: Stuffing factor (k-1 zeros between samples)
    """
    pts = len(x) * up_factor
    out = np.zeros(pts)
    idx = np.arange(0, pts, up_factor)
    out[idx] = x
    return out


def get_interpolated_rate(drc_out_smpls_per_sym,
                          df=0,
                          verbose=False):
    """Compute the interpolated (output) sample rate

    Args:
        drc_out_smpls_per_sym: DRC output rate (samples/symbols)
        df: Tag fractional frequency error
        verbose: True=Print output, False=silent
    """
    fi = drc_out_smpls_per_sym / (1. + df)

    if verbose:
        print('Nominal DRC output rate={0:8.3f}'.format(drc_out_smpls_per_sym))
        print('df={0:8.3f}'.format(df))
        print('fi={0:8.3f}'.format(fi))
        print('Ti={0:8.3f}'.format(1. / fi))

    return fi

#TODO
def validate_mod_type(modulation, mval):
#     """Confirm valid modulation type and M-value
#
#     Args:
#         modulation (str): One of 'fm0', 'miller', or 'bpsk'
#         mval: Gen-2 M value (1, 2, 4, or 8)
#     """
    if modulation not in ['fm0', 'miller', 'bpsk', 'square']:
        raise ValueError('Unsupported mod_type %s' % modulation)
    if modulation == 'fm0' and not mval == 1:
        raise ValueError('M-value must be 1 for %s' % modulation)
    if modulation == 'miller' and mval < 2:
        raise ValueError('M-value must be > 1 for %s' % modulation)


def get_unity_norm_scaler(x):
    mi = np.max(np.abs(np.real(x)))
    mq = np.max(np.abs(np.imag(x)))
    if (mi > 0) or (mq > 0):
        norm_val = 1.0 / np.max([mi, mq])
    else:
        norm_val = 1.0
    return norm_val


def eye_diagram(x, smpls_per_bit):
    xc = np.copy(x)
    print("len(x) = " + str(len(x)))
    print("smpls_per_bit = " + str(smpls_per_bit))
    return np.reshape(xc, (len(x) // smpls_per_bit, smpls_per_bit))


def smooth_data(x, fs_per_symbol, mval=1, lpf_type='FIR'):
    """
    LPF preceeding the interpolator
    """
    # Parameters
    _fs_div_2 = fs_per_symbol / 2.
    _ntaps = 127

    if lpf_type == 'FIR':
        # Instantiate FIR filter
        if mval == 1:
            foff = 0
            fcut = 2
        elif mval == 2:
            foff = mval
            fcut = np.array([-1.7, 1.7])
        else:
            foff = mval
            fcut = np.array([-2.0, 2.0])

        fcut = (foff + fcut) / _fs_div_2

        smooth_filt = FirFilterClass(ntaps=_ntaps,
                                     fcut=fcut,
                                     pass_zero=(foff == 0))
    elif lpf_type == 'RaisedCosine':
        # Instantiate raised cosine filter
        x = zero_stuff(x, fs_per_symbol / mval / 2)

        smooth_filt = RaisedCosineFilterClass(alpha=0.99,
                                              T=0.5,
                                              fs=fs_per_symbol,
                                              mval=mval)
    else:
        raise ValueError('LPF type must be FIR or RaisedCosine.')

        # Smooth the 8blf waveform
    #TODO: Specifies That It Only Smooths An 8 BLF Wavefrom
    return smooth_filt.filter_data(x), smooth_filt


def digital_rate_convert(x,
                         f_drc_in,
                         f_drc_out,
                         n_in_smpls,
                         n_out_smpls,
                         dt):
    """
    Interpolate signal from f_drc_in to f_drc_out
    """
    # Instantiate the farrow interpolator and NCO
    farrow = FarrowClass()
    nco = NcoClass(n_out_smpls)

    # Perform rate conversion
    drc = FixedRateConversion(n_in_smpls,
                              n_out_smpls,
                              1. / f_drc_in,
                              f_drc_in / f_drc_out,
                              farrow,
                              nco)
    drc.execute(x, dt=dt)
    return drc


def generic_plot(fig, ax, x, y,
                 linestyle='-',
                 marker=None,
                 x_lim=None,
                 x_ticks=None,
                 x_label=None,
                 y_lim=None,
                 y_ticks=None,
                 y_label=None,
                 title_str=None,
                 subplots=1,
                 subplot_idx=0,
                 share_x=True
                 ):
    """
    Plot the preamble and pilot_tone
    """

    if ax is None:
        ax = [fig.add_subplot(subplots, 1, 1)]
    elif share_x:
        ax.append(fig.add_subplot(subplots, 1, subplot_idx + 1, sharex=ax[0]))
    else:
        ax.append(fig.add_subplot(subplots, 1, subplot_idx + 1))

    if x is None:
        x = np.arange(0, len(y))

    ax[subplot_idx].plot(x, y, marker=marker, linestyle=linestyle)
    ax[subplot_idx].grid(True)

    # Optional plot parameters
    if x_lim is not None:
        ax[subplot_idx].set_xlim(x_lim)

    if y_lim is not None:
        ax[subplot_idx].set_ylim(y_lim)

    if x_ticks is not None:
        ax[subplot_idx].set_xticks(x_ticks)

    if title_str is not None:
        ax[subplot_idx].set_title(title_str)

    if x_label is not None:
        ax[subplot_idx].set_xlabel(x_label)

    if y_label is not None:
        ax[subplot_idx].set_ylabel(y_label)

    return ax


class DdcClass(object):
    """
    Dual downconverter class
    """

    def __init__(self,
                 x,
                 lpf_fcut=1.0,
                 fblf_norm=1,
                 phase_rad=0,
                 phase_bits=None):
        self.x = x
        self.lpf_fcut = lpf_fcut
        self.fblf_norm = fblf_norm
        self.phase_rad = phase_rad
        self.phase_bits = phase_bits
        self.pts = len(x)
        self.ddc_up = np.zeros(self.pts)
        self.ddc_dn = np.zeros(self.pts)

        self.lo_up = self._create_blf_lo(pol=1)
        self.lo_dn = self._create_blf_lo(pol=-1)
        self._dual_mixer()
        self._ddc_filter()

    def _create_blf_lo(self, pol):
        """
        Return a complex exponential x(t) = exp( j*2*pi*f_blf*t)

            where t = n(Ti)
                    = n/fs  where fs=drc output sample rate

                  Therefore f_blf*t = n*(f_blf / fs)
                                    = n*fblf_norm
                                    = n / drc_out_smpls_per_blf

            x(n) = exp(j*2*pi*n / drc_out_smpls_per_blf)

        Args:
            pts: Length of array to return
            pol: 1=mix up, -1=mix dn
            mxr_phs_rad: mixer phase offset in radians (default=0)
            mxr_phs_bits: phase wordsize

        """
        n = np.arange(self.pts)
        if self.phase_bits is not None:
            # subtracting 3 bits from mixer_phs_bits accounts for the
            # divide by 8 applied to the nom_fcw in yk_rx_wvfm.py
            fxd_shift = 2 ** (self.phase_bits - 3)
            self.fblf_norm = np.round(self.fblf_norm * fxd_shift) / fxd_shift
        lo_phs = (2 * np.pi * n * self.fblf_norm) + self.phase_rad
        return np.exp(pol * 1j * lo_phs)

    def _dual_mixer(self):
        """
        Create y_up and y_dn complex signals
        """
        self.y_mix_up = np.multiply(self.lo_up, self.x)
        self.y_mix_dn = np.multiply(self.lo_dn, self.x)

    def _ddc_filter(self):
        # The sample rate is 8*M
        # fcut = 2.0 * fscale / (2 * self.mval)

        self.ddc_filt = FirFilterClass(ntaps=79,
                                       fcut=self.lpf_fcut,
                                       pass_zero=True)

        self.ddc_up = self.ddc_filt.filter_data(self.y_mix_up)
        self.ddc_dn = self.ddc_filt.filter_data(self.y_mix_dn)


class PreambleClass(object):
    def __init__(self,
                 mod_type='fm0',
                 mval=1,
                 tr_extend=None,
                 n_bits_trext=12,
                 fs_base=1,
                 plot=False,
                 ):
        """
        Generate a Gen2 reverse link preamble including (optional) pilot tone

        #TODO: ATTRIBUTES
        Args:

            mod_type : One of ['fm0', 'miller', 'square']
            mval : One of [1, 2, 4, 8]
            tr_extend: True/False = pilot tone enabled/disabled
            fs_base: samples per BLF
            plot: True=create plot, False=do not create plot

        """
        self.mod_type = mod_type
        self.mval = mval
        #self.num_bits = 0
        self.num_preamble_bits = 0
        self.num_pilot_bits = 0
        self.tr_extend_on = tr_extend
        self.fs_base = fs_base
        self.n_bits_trext = n_bits_trext

        validate_mod_type(mod_type, mval)

        # Create the pilot tone and the preamble data sequence
        self._create_pilot_tone()
        self._create_preamble()

        # Plot
        if plot:
            self._plot()

    def _create_preamble_bit_pattern_2blf(self):
        """
        Create preamble bit pattern sampled at 2blf
        """
        # Generate the 2BLF sequence
        bit_dict = {'fm0': [1, 0, 1, 0, 1, 1],
                    'miller': [0, 1, 0, 1, 1, 1],
                    'square': [1, 0, 1, 0, 1, 1],
                    }
        return create_symbol_sequence(bit_dict[self.mod_type],
                                      encode=self.mod_type)

    def _create_preamble(self):
        """
        Create Gen2 reverse link preamble sampled at fs_base
        """
        # Internal parameters/constants
        _const_cw_sym = 4  # miller preamble always has 4 CW symbols
        _const_violation_sym = 4  # fm0 violation 4th symbol indexed from zero

        # Get the bit pattern
        s = self._create_preamble_bit_pattern_2blf()

        if self.mod_type == 'fm0':
            # Create the violation
            s[_const_violation_sym * 2:] *= -1
        elif self.mod_type == 'miller':
            # Prepend four symbols of CW
            s = np.concatenate([np.ones(_const_cw_sym * 2), -s])
        else:
            pass

        # Bit count bookkeeping
        self.num_preamble_bits = len(s) / 2
        self.num_bits += self.num_preamble_bits

        # Repeat at fs_base rate
        self.preamble = np.repeat(s, self.mval * self.fs_base / 2)

    def _create_pilot_tone(self):
        """
        Create tr_extend (pilot tone)
        """
        # Create the pilot tone vector at fs_base rate
        tr_ext_dict = {True: np.ones(self.mval * self.n_bits_trext *
                                     self.fs_base),
                       False: np.empty([0]),
                       }
        self.tr_extend = tr_ext_dict[self.tr_extend_on]

        if self.tr_extend_on and self.mod_type == 'fm0':
            # FM0 is a sequence of zero data (not CW) so invert alternate
            # half symbols
            for i in np.arange(self.fs_base // 2):
                self.tr_extend[i::self.fs_base] = -1
            self.tr_extend *= -1

        # Pilot gets created first so (total) num_bits equals pilot bits
        self.num_pilot_bits = len(self.tr_extend) / (self.mval * self.fs_base)
        self.num_bits = self.num_pilot_bits

    def _plot(self):
        """
        Plot the pilot tone and preamble
        """
        # Constants
        _const_ylim = (-1.5, 1.5)
        _title_str = ' '.join([self.mod_type.upper(),
                               ':',
                               'Pilot Tone + Preamble'])

        # Setup x and y
        x = np.arange(0, self.num_bits, 1. / (self.fs_base * self.mval))
        y = np.concatenate([self.tr_extend, self.preamble])

        # Call plot function
        fig = plt.figure(figsize=(12, 3))
        ax = generic_plot(fig, None, x, y,
                          marker='.',
                          x_ticks=np.arange(self.num_bits + 1),
                          y_lim=_const_ylim,
                          title_str=_title_str,
                          x_label='Symbols',
                          )


class PacketClass(object):
    def __init__(self,
                 mod_type='fm0',
                 mval=1,
                 num_data_bits=0,
                 bits=None,
                 preamble_obj=None,
                 fs_base=1,
                 plot=False,
                 n_pad_zero_bits=2,
                 ):
        self.mod_type = mod_type
        self.mval = mval
        self.num_data_bits = num_data_bits
        self.preamble = preamble_obj
        self.bits = bits
        self.fs_base = fs_base
        self.n_pad_zero_bits = n_pad_zero_bits

        self._create_packet()

        # If miller modulate  by subcarrier
        if self.mval > 1:
            self._create_subcarrier()
            self.packet = np.multiply(self.packet_baseband, self.subc)
        else:
            self.packet = self.packet_baseband

        if plot:
            self._plot()

    def _create_packet(self):
        """Return array of symbols sampled at 2*f_blf_tag
        """
        if self.bits is not None:
            # Accept user input symbol sequence
            self.num_data_bits = len(self.bits)
            s = create_symbol_sequence(self.bits, encode=self.mod_type)
        else:
            # Generate the symbol sequence
            b = RandomBitGenerator(self.num_data_bits)
            self.bits = b.data
            s = create_symbol_sequence(self.bits, encode=self.mod_type)

        # Repeat data up to the fs_base rate
        data = np.repeat(s, self.mval * self.fs_base / 2)

        # Set polarity based on last preamble sample
        if self.preamble.preamble[-1] == 1:
            data = -1 * data
        pad = np.zeros(int(2 * self.n_pad_zero_bits * self.mval * self.fs_base / 2))

        # Concatenate the pilot tone, preamble, and data
        self.num_bits = self.num_data_bits + self.preamble.num_bits + \
            2 * self.n_pad_zero_bits
        self.packet_baseband = np.concatenate([pad,
                                               self.preamble.tr_extend,
                                               self.preamble.preamble,
                                               data,
                                               pad,
                                               ])

    def _create_subcarrier(self):
        """Return a square wave
        """
        self.subc = create_symbol_sequence(np.zeros(int(self.num_bits * self.mval)),
                                           encode='square')
        self.subc = np.repeat(self.subc, self.fs_base / 2)

    def _plot(self):
        """
        Create three plots:
        1) baseband packet
        2) subcarrier
        3) subcarrier modulated packet
        """
        # Constants
        _const_ylim = (-1.5, 1.5)
        _title_str = [' '.join([self.mod_type.upper(), ':',
                                'Baseband Packet']),
                      ' '.join([self.mod_type.upper(), ':',
                                'Subcarrier']),
                      ' '.join([self.mod_type.upper(), ':',
                                'Final Packet']),
                      ]
#TODO y, imax slices converted to int()
        # Plot baseband packet
        subplots = 3 - (self.mval == 1)
        i = 0
        bmax = min(32, self.num_bits)
        imax = bmax * self.fs_base * self.mval
        x = np.arange(0, bmax, 1. / (self.fs_base * self.mval))
        y = self.packet_baseband[0:int(imax)]
        fig = plt.figure(figsize=(12, 3 * 3))
        ax = generic_plot(fig, None, x, y,
                          marker='.',
                          x_ticks=np.arange(bmax + 1),
                          y_lim=_const_ylim,
                          title_str=_title_str[0],
                          subplots=subplots,
                          )

        if self.mval > 1:
            i += 1
            # Plot subcarrier
            x = np.arange(0, bmax, 1. / (self.fs_base * self.mval))
            y = self.subc[0:imax]
            _ = generic_plot(fig, ax, x, y,
                             marker='.',
                             x_ticks=np.arange(bmax + 1),
                             y_lim=_const_ylim,
                             title_str=_title_str[1],
                             subplots=subplots,
                             subplot_idx=i,
                             )

        # Plot subcarrier modulated packet
        #TODO: Imax changed to int()
        i += 1
        x = np.arange(0, bmax, 1. / (self.fs_base * self.mval))
        y = self.packet[0:int(imax)]
        _ = generic_plot(fig, ax, x, y,
                         marker='.',
                         x_ticks=np.arange(bmax + 1),
                         y_lim=_const_ylim,
                         title_str=_title_str[2],
                         x_label='Symbols',
                         subplots=subplots,
                         subplot_idx=i,
                         )


class GenerateRevWaveform:
    link = namedtuple('link', 'mod_type mval tr_extend blf_hz fsample_hz')

    #TODO:

    def __init__(self,
                 link=link('miller', 4, 0, 320e3, 3e6),
                 num_bits=8,
                 theta_rf_rad=1,
                 blf_err=0,
                 EbNo_dB=100,
                 dt=0,
                 n_pad_zero_bits=2,
                 drc_in_smpls_per_blf=5,
                 n_bits_trext=0,
                 ):

        """
        The waveform is generated by normalizing the tag symbol rate to
        one:
            f_sym_tag = 1

        This also sets the tag's BLF:

            f_blf_tag = mval

        The waveform generator sets an integer number of samples per BLF at the
        Digital Rate Converter (DRC) input:

            drc_in_smpls_per_blf = 8 (samples/BLF)

        The DRC input sample rate is:

            f_drc_in = drc_in_smpls_per_blf * f_blf_tag

        The tag's BLF (f_blf_tag) is equal to the reader's target BLF (f_blf)
        plus some frequency error (f_err):

            f_blf_tag = f_blf + f_err

        Solving for the reader's target BLF (f_blf) gives:

            f_blf_tag = [ 1 + (f_err/f_blf) ] f_blf
            f_blf = f_blf_tag / (1 + frac_blf_err)

        where frac_blf_err = f_err/f_blf is the fractional tag frequency error.

        Assuming frac_blf_err=0 (zero tag frequency error) the nominal DRC
        output rate (samples/BLF) is the ratio of the decimation filter output
        sample rate (Hz) to the target BLF (Hz):

            drc_out_smpls_per_blf = fs_hz/f_blf_hz (samples/BLF)

        The actual sample rate (samples/BLF) at the DRC output is:

            f_drc_out = drc_out_smpls_per_blf  * f_blf
                      = drc_out_smpls_per_blf * f_blf_tag/(1 + frac_blf_err)

        Example:
            Assume a miller-4 signal with BLF = 320 kHz and a +5% frequency
            error.
                mod_type = 'miller-4'  Modulation is miller M=4
                mval = 4
                blf_hz = 320e3
                fs_hz = 3e6
                frac_blf_err = 0.05

            Computing the tag and reader BLF gives:

                f_blf_tag = mval = 4
                f_blf = 4 / 1.05 = 3.81 samples/BLF

            The input sample rate to the DRC is:

                f_drc_in = drc_in_smpls_per_blf * f_tag_blf

            The nominal DRC output rate is:

                drc_out_smpls_per_blf = fs_hz / blf_hz
                                  = 3e6 / 320e3
                                  = 9.375 samples/BLF

            The actual DRC output sample rate is:

                f_drc_out = drc_out_smpls_per_blf * f_blf
                          = 9.375 * 3.81
                          = 35.71 samples/symbol
        """
        #TODO

        self.num_bits = num_bits
        self.mod_type = link.mod_type
        self.mval = link.mval
        self.tr_extend = link.tr_extend
        self.theta_rf_rad = theta_rf_rad
        self.blf_err = blf_err
        self.EbNo_dB = EbNo_dB
        self.dt = dt
        self.n_pad_zero_bits = n_pad_zero_bits
        self.is_baseband = bool(link.mval == 1)
        self.drc_in_smpls_per_blf = drc_in_smpls_per_blf
        self.drc_out_smpls_per_blf = link.fsample_hz / link.blf_hz
        self.f_drc_in = drc_in_smpls_per_blf * link.mval
        self.n_in_smpls = 0
        self.n_out_smpls = 0
        self.f_drc_out = None
        self.drc = None
        self.lo_up = None
        self.lo_dn = None
        self.bits = None
        self.num_pre_bits = 0
        self.num_trext_bits = n_bits_trext
        self.wvfm = {'lpf_in': None,
                     'lpf_out': None,
                     'y_up_ddc_out': None,
                     'y_dn_ddc_out': None,
                     'y_up': None,
                     'y_dn': None}

        # Validate the modulation type and M value
        validate_mod_type(link.mod_type, link.mval)

    def get_num_bits(self, num_bits):
        return num_bits

    def generate_waveform(self,
                          bits,
                          plot,
                          phase_rad,
                          phase_bits,
                          lpf_type):
        """Create the final waveform
        """
        if lpf_type == 'FIR':
            # Sample at 8 blf
            fs_base = 8
        else:
            # Sample at 2 blf
            fs_base = 2

        # Compute the DRC output rate (samples per symbol)
        self.f_drc_out = get_interpolated_rate(
            self.drc_out_smpls_per_blf * self.mval,
            df=self.blf_err,
            verbose=False,
        )

        # Instantiate preamble object
        self.preamble = PreambleClass(mod_type=self.mod_type,
                                      mval=self.mval,
                                      tr_extend=self.tr_extend,
                                      n_bits_trext=self.num_trext_bits,
                                      fs_base=fs_base,
                                      plot=plot)

        # Instantiate packet object
        self.packet = PacketClass(mod_type=self.mod_type,
                                  mval=self.mval,
                                  num_data_bits=self.num_bits,
                                  preamble_obj=self.preamble,
                                  bits=bits,
                                  fs_base=fs_base,
                                  plot=plot,
                                  n_pad_zero_bits=self.n_pad_zero_bits)

        # Update number of bits and the vector of bit values
        self.num_bits = self.packet.num_bits
        self.bits = self.packet.bits

        self.wvfm['lpf_in'] = self.packet.packet

        # Compute the number of DRC input and output samples
        self.n_in_smpls = int(self.f_drc_in * self.num_bits)
        self.n_out_smpls = int(self.n_in_smpls *
                               self.f_drc_out / self.f_drc_in)

        self.wvfm['lpf_out'], self.smooth_filt = \
            smooth_data(self.wvfm['lpf_in'],
                        self.f_drc_in,
                        mval=self.mval,
                        lpf_type=lpf_type)

        self.drc = digital_rate_convert(self.wvfm['lpf_out'],
                                        self.f_drc_in,
                                        self.f_drc_out,
                                        self.n_in_smpls,
                                        self.n_out_smpls,
                                        self.dt,
                                        )

        # TODO:  This appears to be a hack to our signal source and should be
        # reviewed more carefully.  Throw out FIR Group Delay samples worth of
        # data
        self.n_remove_smpls = 32 * self.n_out_smpls / float(self.n_in_smpls)

        # Apply RF phase offset
        self.y_rf = self.drc.y * np.exp(1j * self.theta_rf_rad)

        # Add AWGN
        # sig_var = E[(x-mu)**2]
        #         = (1/N) sum[ (x[n] - mu)**2 ]
        #         = energy per sample
        # Eb = sig_var * samples per symbol
        self.sig_var = np.var(self.drc.y)
        # TODO: Review the Eb calculation!
        self.Eb = self.sig_var * self.f_drc_out
        self.sigma = get_noise_std_deviation(self.Eb, self.EbNo_dB)
        ni = np.random.normal(loc=0, scale=self.sigma, size=len(self.y_rf))
        nq = np.random.normal(loc=0, scale=self.sigma, size=len(self.y_rf))

        self.y_rf += (ni + 1j * nq)

        if self.is_baseband:
            self.wvfm['y_up_ddc_out'] = self.y_rf
            self.wvfm['y_dn_ddc_out'] = np.zeros(len(self.y_rf))
        else:
            self.ddc = DdcClass(self.y_rf,
                                lpf_fcut=2.0 / (4 * self.mval),
                                fblf_norm=1. / self.drc_out_smpls_per_blf,
                                phase_rad=phase_rad,
                                phase_bits=phase_bits)
            self.wvfm['y_up_ddc_out'] = self.ddc.ddc_up
            self.wvfm['y_dn_ddc_out'] = self.ddc.ddc_dn

        # Decimate
        self.wvfm['y_up'] = self.wvfm['y_up_ddc_out'][0::self.mval]
        self.wvfm['y_dn'] = self.wvfm['y_dn_ddc_out'][0::self.mval]

        # Normalize
        norm_up = get_unity_norm_scaler(self.wvfm['y_up'])
        norm_dn = get_unity_norm_scaler(self.wvfm['y_dn'])
        norm = np.min([norm_up, norm_dn])
        self.wvfm['y_up'] *= norm
        self.wvfm['y_dn'] *= norm
        self.n_out_smpls = len(self.wvfm['y_up'])

        wvfm_I = np.real(self.wvfm['y_up'])
        wvfm_Q = np.imag(self.wvfm['y_up'])

        if plot:
            self._plot_sig_and_eye(self.wvfm['lpf_out'], self.f_drc_in,
                                   name='lpf_out')
            self._plot_sig_and_eye(np.real(self.wvfm['y_up']),
                                   self.f_drc_in // self.mval,
                                   name='y_up (I)')
            self._plot_sig_and_eye(np.imag(self.wvfm['y_up']),
                                   self.f_drc_in // self.mval,
                                   name='y_up (Q)')

        return self.wvfm['y_up']

    def _plot_sig_and_eye(self, sig, smpls_per_bit, name=''):
        """
        Create three plots:
        """
        # Constants
        _const_ylim = (-1.5, 1.5)
        _title_str = [
            ' '.join([self.mod_type.upper(), ':', name, '(time domain)']),
            ' '.join([self.mod_type.upper(), ':', name, '(eye)']),
        ]
        _start_eye_bit = self.num_bits / 4
        _stop_eye_bit = 2 * _start_eye_bit + _start_eye_bit

        # Time domain
        bmax = min(32, self.num_bits)
        imax = bmax * smpls_per_bit
        x = np.arange(0, bmax, 1. / smpls_per_bit)
        y = sig[0:int(imax)]
        fig = plt.figure(figsize=(12, 3 * 2))
        ax = generic_plot(fig, None, x, y,
                          marker='.',
                          x_ticks=np.arange(bmax + 1),
                          y_lim=_const_ylim,
                          title_str=_title_str[0],
                          subplots=2,
                          share_x=False,
                          )

        # Eye diagram
        #TODO Changed to int()
        y = eye_diagram(sig[int(_start_eye_bit * smpls_per_bit):
                        int(_stop_eye_bit * smpls_per_bit)],
                        smpls_per_bit)
        _ = generic_plot(fig, ax, None, np.transpose(y),
                         marker='.',
                         x_ticks=np.arange(8),
                         y_lim=_const_ylim,
                         title_str=_title_str[1],
                         subplots=2,
                         subplot_idx=1,
                         share_x=False,
                         )
