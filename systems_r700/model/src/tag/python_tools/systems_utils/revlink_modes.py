from collections import namedtuple

#
# Setup a dictionary of objects used to configure the reverse link
#
RevLink = namedtuple('RevLink', ('mod_type', 'mval', 'tr_extend', 'blf_hz',
                                 'fsample_hz', 'data_rate'))

revlink = {'fm0_640kHz': RevLink(mod_type='fm0',
                                 mval=1,
                                 tr_extend=True,
                                 blf_hz=640e3,
                                 fsample_hz=5.12e6,
                                 data_rate=640e6),  #TODO: The data_rate is the blf_frequency * 10^3 in order to convert
           'fm0_640kHz_8LF': RevLink(mod_type='fm0',#TODO: it to bit/second
                                     mval=1,
                                     tr_extend=True,
                                     blf_hz=640e3,
                                     fsample_hz=5.12e6, #TODO: 20.48e6
                                     data_rate=640e6),
           'miller_4_320kHz': RevLink(mod_type='miller',
                                      mval=4,
                                      tr_extend=True,
                                      blf_hz=320e3,
                                      fsample_hz=3e6,
                                      data_rate=320e6 / 4),
           'miller_4_320kHz_8LF': RevLink(mod_type='miller',
                                          mval=4,
                                          tr_extend=True,
                                          blf_hz=320e3,
                                          fsample_hz=2.56e6,
                                          data_rate=320e6 / 4),
           '3b4b_baseband': RevLink(mod_type='3b4b_baseband',
                                    mval=0,
                                    tr_extend=True,
                                    blf_hz=640e3,
                                    fsample_hz=6e6,
                                    data_rate=640e6),
           '3b4b_baseband_p8': RevLink(mod_type='3b4b_baseband',
                                    mval=0,
                                    tr_extend=True,
                                    blf_hz=640e3,
                                    fsample_hz=640e3*8.,
                                    data_rate=640e6),
           '3b4b_subcarrier_m1': RevLink(mod_type='3b4b_subcarrier',
                                         mval=1,  # Any value greater than 0
                                         tr_extend=True,
                                         blf_hz=640e3,
                                         fsample_hz=6e6,
                                         data_rate=640e6),
           '3b4b_subcarrier_m2': RevLink(mod_type='3b4b_subcarrier',
                                    mval=2, # Any value greater than 0
                                    tr_extend=True,
                                    blf_hz=640e3,
                                    fsample_hz=6e6,
                                    data_rate=640e6/2),
           '3b4b_subcarrier_m4': RevLink(mod_type='3b4b_subcarrier',
                                         mval=4,  # Any value greater than 0
                                         tr_extend=True,
                                         blf_hz=640e3,
                                         fsample_hz=6e6,
                                         data_rate=640e6 / 4),
           'bpsk_subcarrier_m2': RevLink(mod_type='bpsk',
                                         mval=2,  # Any value greater than 0
                                         tr_extend=True,
                                         blf_hz=640e3,
                                         fsample_hz=6e6,
                                         data_rate=640e6 / 2)}
