import pickle
import numpy as np
import os

from systems_r700.model.src.tag.python_tools.systems_utils.helper_functions import calculate_num_bits

class ConfigClass(object):

    def __init__(self, link=0, num_bits=30, packet_size=30, adjusted_packet_size=30):

        self.cc_i_sign = 1
        self.cc_i = 1
        self.cc_q_sign = 1
        self.cc_q = 1
        #Updated during the CC Process of Simulaiton

        self.link_dict = {'mod_type': link.mod_type,
                          'm': link.mval,
                          'blf_hz': link.blf_hz,
                          'fsample_hz': link.fsample_hz,
                          'tr_extend': link.tr_extend,
                          'pilot_sym': 12,
                          'compute_num_bits':False,  # computes num_bits_to_sim
                          # 'num_bits': 180000 * 6,  # Override if write_to_file
                          'num_bits': num_bits,
                          'num_bits_packet': packet_size,
                          'adjusted_packet_size': adjusted_packet_size,
                          'max_bits': 180000 * 6,
                          'blf_err': 0.0,
                          'blf_est': 0.0,
                          'theta_rf_rad': 0,
                          'dt': 0
                          }

        self.cm_dict = {'cm_bypass': 0,
                        'hpf_fc_hz': 10e3,
                        'n_hpf_order': 1,
                        'lpf_order': 3
                        }

        self.chan_filt_dict = {'ch_bypass': 0}

        """ farrow_config : 1 --> Load farrow with the first four valid samples
            farrow_config : 0 --> Load farrow one sample at a time; initialized
            to zero"""
        self.trl_dict = {'trl_bypass': 0,
                         'open_loop': 0,
                         'farrow_config': 1,
                         # 'drc_kp': (0.4, 0.4),  # (ACQ, TRK)
                         'drc_kp': (0.4, 0.4),  # (ACQ, TRK)
                         # 'drc_ki': (0.01, 0.001),  # (ACQ, TRK)
                         'drc_ki': (0.01, 0.0001),  # (ACQ, TRK)
                         # 'drc_shift': 150
                         'drc_shift': 180}  # Sample at which ACQ --> TRK

        self.pmf_dict = {'window': 64,
                         'win_start': 53,  # (Use 85 or 180 for 12 or 24 pilot
                         'barker_type': 'barker13'}  # barker11 or barker13

        self.ebno_array = np.arange(100, 101)

        # # File and I/O options
        self.write_inp_wfm_to_file = 0  # Program will quit after file write
        self.run_number = np.arange(1)  # Run number to create many files
        self.read_inp_wfm_from_file = 0
        self.write_to_noise_file = 0
        self.read_from_noise_file = 0
        self.save_output = 0
        self.pmf_save_output = 1
        self._cpp_dir()
        self.compile_cpp = 0
        self.cpp_sim = 0  # Use TRL CPP for faster code execution

        # Plotting options
        self.trl_plot_fig = 0  # Plotting TRL output

        # # These settings are determined by PMF output
        self.num_trials = 1000  # Make number of trials 0 for BER sweep
        self.pmf_peak_idx = 0
        self.peamble_len = None
        self.pmf_sign = None
        self.cnt = 0
        self.pmf_peak_idx_array = np.zeros(
            np.max([len(self.ebno_array), self.num_trials]))
        self.ber_thresh = 1  # Setting to 1 bypasses BER thresh check

        #
        # # Hard or soft decision decoding for 3b4b
        self.decode_dict = {'hard_decision_decoding': True}

        self.bit_errs = 0
        self.cnt = 0
        self.ebno_cnt = 0
        self.trial = 0

        self.EbNo_dB_vec = self.ebno_array
        self.EbNo_dB = self.EbNo_dB_vec[self.ebno_cnt]
        self.blf_err = self.link_dict['blf_err']
        self.blf_est = self.link_dict['blf_est']
        self.cnt_thresh = np.max([len(self.EbNo_dB_vec), self.num_trials])
        self.ber_array = np.zeros(self.cnt_thresh)

        if self.link_dict['compute_num_bits'] is True:
            self.mod_type = self.link_dict['mod_type']
            self.num_bits = calculate_num_bits(self.EbNo_dB_vec[self.ebno_cnt],
                                          mod_type=self.mod_type)
            self.num_bits_max = int(1.08e7)
            self.bits = np.min([self.num_bits, self.num_bits_max])
        else:
            self.bits = self.link_dict['num_bits']

        self._set_timing_config()

        self._validate_settings()

    def _cpp_dir(self):
        self.cpp_dir = os.path.join("./python_tools",
                                    "systems_utils",
                                    "cpp",
                                    "trl_main")

    def _set_timing_config(self):
        # --------------------------------------------------------------------
        # Sampling parameter setup
        # --------------------------------------------------------------------
        self.baseline_out_samples_per_symbol = 8.
        fs_hz = self.link_dict['fsample_hz']
        blf_hz = self.link_dict['blf_hz']
        blf_err = self.link_dict['blf_err']

        fs_in_samples_per_symbol = fs_hz / blf_hz
        self.Ts = 1. / fs_in_samples_per_symbol
        Ti_nom = 1. / self.baseline_out_samples_per_symbol
        self.nom_drc_fcw = Ti_nom / self.Ts

        # Nominal LF loop seed
        self.nom_lf_fcw = 1.0 - 1.0 / (1 + self.link_dict['blf_est'])

        # Compute the actual Ti/Ts for use in estimating number of loop
        # output samples
        self.Ti_Ts = fs_in_samples_per_symbol / 8. / (1. + blf_err)
        # --------------------------------------------------------------------

    def set_pmf_config(self, pmf_obj=0):
        self.pmf_peak_idx = pmf_obj.peak_idx
        self.preamble_len = pmf_obj.preamble_len
        self.pmf_sign = pmf_obj.sign

        self.pmf_peak_idx_array[self.cnt] = self.pmf_peak_idx

    def _link_all_dicts(self):
        self.config_dict = self.link_dict.copy()
        self.config_dict.update(self.cm_dict)
        self.config_dict.update(self.chan_filt_dict)
        self.config_dict.update(self.trl_dict)
        self.config_dict.update(self.pmf_dict)
        self.config_dict.update(self.decode_dict)

    def save_output_to_file(self):

        hard_decoding_flag = self.decode_dict['hard_decision_decoding']
        cm_str = str(self.cm_dict['hpf_fc_hz'] / 1.e3) + '_hpf_khz'
        if self.cm_dict['cm_bypass']:
            cm_str='cm_bypass'

        if self.trl_dict['trl_bypass']:
            trl_str = 'trl_byp'
        else:
            trl_str = 'with_trl'

        pickle_name = '_'.join([self.link_dict['mod_type'],
                                'M'+str(self.link_dict['m']),
                                'blf',
                                str(int(self.link_dict['blf_hz'] / 1.e3)),
                                'fs',
                                str(self.link_dict['fsample_hz'] / 1.e6),
                                'MHz',
                                cm_str,
                                trl_str,
                                str(hard_decoding_flag),
                                'hard_decoding'
                                ])

        pickle_name += '.pkl'

        self._link_all_dicts()

        with open(pickle_name, 'w') as pickle_file:
            pickle.dump([self.ebno_array, self.ber_array, self.config_dict],
                        pickle_file,
                        pickle.HIGHEST_PROTOCOL)

    def save_pmf_index_array_to_file(self):
        ebno_db = self.ebno_array[0]
        cm_str = str(self.cm_dict['hpf_fc_hz'] / 1.e3) + '_hpf_khz'
        if self.cm_dict['cm_bypass']:
            cm_str='cm_bypass'

        if self.trl_dict['trl_bypass']:
            trl_str = 'trl_byp'
        else:
            trl_str = 'with_trl'

        pickle_name = '_'.join(['pmf_peak_idx_',
                                self.link_dict['mod_type'],
                                'M' + str(self.link_dict['m']),
                                'blf',
                                str(int(self.link_dict['blf_hz'] / 1.e3)),
                                'fs',
                                str(self.link_dict['fsample_hz'] / 1.e6),
                                'MHz',
                                cm_str,
                                trl_str,
                                str(self.pmf_dict['window']),
                                'pmf_window',
                                self.pmf_dict['barker_type'],
                                str(ebno_db),
                                'EbNo_dB'
                                ])

        pickle_name += '.pkl'
        self._link_all_dicts()

        with open(pickle_name, 'w') as pickle_file:
            pickle.dump(
                [self.ebno_array, self.pmf_peak_idx_array, self.config_dict],
                pickle_file,
                pickle.HIGHEST_PROTOCOL)

    def _validate_settings(self):
        """ Ensure that system settings do not have a conflict"""
        mod_type = self.link_dict['mod_type']
        hd_flag = bool(self.decode_dict['hard_decision_decoding'])
        trl_plot = bool(self.trl_plot_fig)
        cm_bypass = bool(self.cm_dict['cm_bypass'])
        num_bits = self.link_dict['num_bits']
        ebno_len = len(self.ebno_array)
        num_trials = self.num_trials
        trl_bypass = bool(self.trl_dict['trl_bypass'])
        compute_num_bits = bool(self.link_dict['compute_num_bits'])
        save_output = bool(self.save_output)

        # File I/O options
        write_inp_to_file = bool(self.write_inp_wfm_to_file)
        read_inp_from_file = bool(self.read_inp_wfm_from_file)
        write_noise_to_file = bool(self.write_to_noise_file)
        read_noise_from_file = bool(self.read_from_noise_file)

        if 'bpsk' in mod_type and hd_flag is False:
            raise ValueError('Hard decision decoding cannot be false for BPSK')
        elif 'fm0' in mod_type and hd_flag is False:
            raise ValueError('Hard decision decoding cannot be false for FM0')
        elif 'miller' in mod_type and hd_flag is False:
            raise ValueError(
                'Hard decision decoding cannot be false for miller')
        elif cm_bypass is False and num_bits < 30:
            raise ValueError('Need more points for CMRX filtering')
        elif write_inp_to_file and read_inp_from_file:
            raise ValueError('Cannot both read and write input to file')
        elif write_noise_to_file and read_noise_from_file:
            raise ValueError('Cannot read from and write to noise file')
        elif ebno_len > 1 and num_trials != 0:
            raise ValueError('Non-zero num_trials for BER EbNo sweep')
        elif trl_plot is True and num_bits > 1000:
            raise ValueError("Too many bits to plot; reconfigure settings")
        elif trl_plot is True and trl_bypass is True:
            raise ValueError('TRL cannot be plotted if it is bypassed')
        elif trl_plot is True and compute_num_bits is True:
            raise ValueError('TRL plot enabled when compute bits is True')
        elif compute_num_bits is True and write_inp_to_file is True:
            raise ValueError('Cannot compute bits if writing input to file')
        elif save_output and ebno_len == 1:
            raise ValueError('Dont save one EbNo point to output file')