import numpy as np
import matplotlib.pyplot as plt


class DecodeBitsClass(object):

    def __init__(self,
                 x=0,
                 config=0,
                 plot_fig=False):

        self.x = x  # Output samples of the TRL
        self.win_start = config.pmf_dict['win_start']
        self.plot_fig = plot_fig
        self.peak_idx = config.pmf_peak_idx
        self.preamble_len = config.preamble_len
        self.mod_type = config.link_dict['mod_type']
        self.y = None  # Output bits
        self.num_bits = config.link_dict['num_bits_packet']
        self.sign_val = config.pmf_sign
        self.hard_decision_decoding = config.decode_dict[
            'hard_decision_decoding']
        self.last_word = 0

    def execute(self):
        self._adjust_num_bits()
        self._decode_bits()
        if self.plot_fig:
            self._plot_output()

    def _adjust_num_bits(self):
        if '3b4b' in self.mod_type:
            self.num_bits = int(self.num_bits * 4/3.)

    def _last_word_size(self):
        if '3b4b' in self.mod_type:
            self.last_word = 4
        else:
            self.last_word = 2

    def _decode_bits(self):

        self._last_word_size()
        start = self.win_start + self.peak_idx

        """Last word is ignored due to boundary condition in TRL"""
        #TODO: COMMENT OUT THE SUBTRACTION OF SELF.LAST_WORD IN ORDER TO RETURN ALL OF THE BITS
        if self.mod_type is 'fm0' or self.mod_type is 'miller':
            num_bits = int(self.preamble_len/2 + self.num_bits - self.last_word)
        else:
            num_bits = int(self.preamble_len + self.num_bits - self.last_word)

        # Feature samples
        xvec = self.x[start:start+num_bits*8:4].copy()
        xlen = int(len(xvec)/2)
        xmat = xvec.reshape(xlen, 2)

        if '3b4b' in self.mod_type or 'bpsk' in self.mod_type:
            xbits = np.sum(xmat, axis=1)
        else:
            xbits = xvec

        # Output bits
        if self.sign_val == 1:
            if self.hard_decision_decoding:
                self.y = (xbits > 0)*1
            else:
                self.y = xbits
        else:
            if self.hard_decision_decoding:
                self.y = (xbits < 0)*1
            else:
                self.y = -xbits

    def _plot_output(self):
        # Plotting the auto-correlation function
        fig, ax = plt.subplots()
        ax.plot(self.y)
        ax.set_xlabel('Bit index')
        ax.set_ylabel('Bits')
        ax.grid(True)
        plt.show()

