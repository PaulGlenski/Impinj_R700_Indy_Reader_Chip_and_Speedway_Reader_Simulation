# ------------------------------------------------------------------------
# Python imports
# ------------------------------------------------------------------------
import numpy as np
import matplotlib.pylab as plt
from collections import namedtuple
from datetime import datetime

# ------------------------------------------------------------------------
# My imports
# ------------------------------------------------------------------------
from yk_design.python_tools.systems_utils.system_config import ConfigClass
from yk_design.python_tools.systems_utils.rx_waveforms import GenerateRevWaveform
from yk_design.python_tools.systems_utils.rate_conversion import NcoClass
from yk_design.python_tools.systems_utils.rate_conversion import TedClass
from yk_design.python_tools.systems_utils.rate_conversion import FarrowClass
from yk_design.python_tools.systems_utils.rate_conversion import LoopFilterClass
from yk_design.python_tools.systems_utils.rate_conversion import TrlClass
from yk_design.python_tools.systems_utils.rate_conversion import AtanClass
from yk_design.python_tools.systems_utils.rate_conversion import MixerClass
from yk_design.python_tools.systems_utils.rate_conversion import MatchedFilterClass
from yk_design.python_tools.systems_utils.cm_rx import CmRxClass
from yk_design.python_tools.systems_utils.channel_filter import ChannelFilterClass
from yk_design.python_tools.systems_utils.ddc import DdcClass
from yk_design.python_tools.systems_utils.dmf import DmfClass
from yk_design.python_tools.systems_utils.pmf import PmfClass
from yk_design.python_tools.systems_utils.decode_bits import DecodeBitsClass
from yk_design.python_tools.systems_utils.decode_symbols import DecodeSymbolClass
from yk_design.python_tools.systems_utils.revlink_modes import revlink
from yk_design.python_tools.systems_utils.helper_functions import plot_ber_curves
from yk_design.python_tools.systems_utils.helper_functions import calculate_num_bits

# ------------------------------------------------------------------------
# Parameter Definitions
# ------------------------------------------------------------------------
#
# Define a record for each loop component type that specifies some simple
# quantization.
#
# float_calc = True : Loop operates in floating point mode
# float_calc = Flase: Loop operates in fixed point mode and uses these records
#

class SimulationParameters():
    def __init__(self):
        self.float_calc = True
        self.QuantRecord = namedtuple('QuantRecord', ('signed',
                                                      'integer',
                                                      'fractional',
                                                      'rounding',
                                                      'float_calc'))

        self.lpf_quant_record = self.QuantRecord(signed=True,
                                       integer=0,
                                       fractional=9,
                                       rounding=True,
                                       float_calc=self.float_calc)

        self.farrow_quant_record = self.QuantRecord(signed=True,
                                          integer=0,
                                          fractional=10,
                                          rounding=True,
                                          float_calc=self.float_calc)

        self.ted_quant_record = self.QuantRecord(signed=True,
                                       integer=0,
                                       fractional=9,
                                       rounding=True,
                                       float_calc=self.float_calc)

        self.nco_quant_record = self.QuantRecord(signed=False,
                                       integer=0,
                                       fractional=15,
                                       rounding=True,
                                       float_calc=self.float_calc)

        self.atan_quant_record = self.QuantRecord(signed=True,
                                        integer=1,
                                        fractional=8,
                                        rounding=False,
                                        float_calc=self.float_calc)

        self.mxr_quant_record = self.QuantRecord(signed=True,
                                       integer=0,
                                       fractional=10,
                                       rounding=True,
                                       float_calc=self.float_calc)

        self.mf_quant_record = self.QuantRecord(signed=True,
                                      integer=0,
                                      fractional=10,
                                      rounding=True,
                                      float_calc=self.float_calc)

        #
        # Define a configuration record for any loop component that can be configured.
        #
        # Loop Filters
        #
        self.LpfRecord = namedtuple('LpfRecord', ('BnT',
                                             'BnT_shift',
                                             'shift_idx',
                                             'zeta',
                                             'kphs',
                                             'kosc',
                                             'quant'))
        lpf_ted_record = self.LpfRecord(  # BnT=0.12,
            BnT=0.02,
            # BnT_shift=0.05,
            BnT_shift=0.0,
            # shift_idx=90,
            shift_idx=150,
            # zeta=1.0,
            zeta=1.0,
            # kphs=2.7 / (2 * np.pi),
            kphs=4. / (2 * np.pi),
            kosc=1.0,
            quant=self.lpf_quant_record)

        self.lpf_lf_record = self.LpfRecord(BnT=0.03,
                                  BnT_shift=0.01,
                                  shift_idx=None,
                                  zeta=1.0,
                                  kphs=2.7 / (2 * np.pi),
                                  kosc=1.0,
                                  quant=self.lpf_quant_record)

        self.lpf_rf_record = self.LpfRecord(BnT=0.03,
                                  BnT_shift=0.01,
                                  shift_idx=None,
                                  zeta=1.0,
                                  kphs=2.7 / (2 * np.pi),
                                  kosc=1.0,
                                  quant=self.lpf_quant_record)
        #
        # Farrow filters
        #
        self.FarrowRecord = namedtuple('FarrowRecord', ('interp', 'quant'))
        self.farrow_record = self.FarrowRecord(interp='cubic',
                                     quant=self.farrow_quant_record)
        #
        # Timing Error Detector
        #
        self.TedRecord = namedtuple('TedRecord', ('decimate',
                                             'gardner',
                                             'quant'))
        self.ted_record = self.TedRecord(decimate=4,
                               gardner=False,
                               quant=self.ted_quant_record)
        #
        # Numerically Controlled Oscillator
        #
        self.NcoRecord = namedtuple('NcoRecord', ('modulo', 'quant'))
        self.nco_drc_record = self.NcoRecord(modulo=4, quant=self.QuantRecord(signed=False,
                                                               integer=2,
                                                               fractional=8,
                                                               rounding=True,
                                                               float_calc=self.float_calc))
        self.nco_lf_record = self.NcoRecord(modulo=1, quant=self.nco_quant_record)
        self.nco_rf_record = self.NcoRecord(modulo=1, quant=self.nco_quant_record)
        #
        # LF and RF Phase Detectors (ATAN tables)
        #
        self.AtanRecord = namedtuple('AtanRecord', ('quant'))
        self.atan_lf_record = self.AtanRecord(quant=self.atan_quant_record)
        self.atan_rf_record = self.AtanRecord(quant=self.atan_quant_record)
        #
        # Mixer
        #
        self.MixerRecord = namedtuple('MixerRecord', ('up', 'quant'))
        self.mxr_up_record = self.MixerRecord(up=True, quant=self.mxr_quant_record)
        self.mxr_dn_record = self.MixerRecord(up=False, quant=self.mxr_quant_record)
        #
        # Matched filter
        #
        self.MfRecord = namedtuple('MfRecord', ('n_taps', 'quant'))
        self.mf_record = self.MfRecord(n_taps=
                             3, quant=self.mf_quant_record)


    def get_loop_components(self, max_pts):
        """Create a dictionary with instances of every loop component
        """
        d = {'farrow': FarrowClass(**self.farrow_record._asdict()),
             'ted': TedClass(max_pts, **self.ted_record._asdict()),
             # 'lpf_ted': LoopFilterClass(**lpf_ted_record._asdict()),
             # 'lpf_lo': LoopFilterClass(**lpf_lf_record._asdict()),
             # 'lpf_rf': LoopFilterClass(**lpf_rf_record._asdict()),
             'nco_drc': NcoClass(max_pts, **self.nco_drc_record._asdict()),
             'nco_lf': NcoClass(max_pts, **self.nco_lf_record._asdict()),
             'nco_rf': NcoClass(max_pts, **self.nco_rf_record._asdict()),
             'atan_lf': AtanClass(max_pts, **self.atan_lf_record._asdict()),
             'atan_rf': AtanClass(max_pts, **self.atan_rf_record._asdict()),
             'mxr_up': MixerClass(max_pts, **self.mxr_up_record._asdict()),
             'mxr_dn': MixerClass(max_pts, **self.mxr_dn_record._asdict()),
             'mf_ted': MatchedFilterClass(max_pts, **self.mf_record._asdict()),
             }
        return d

    def get_float_calc(self):
        return self.float_calc

