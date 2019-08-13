import numpy as np
import matplotlib.pyplot as plt


class PmfClass(object):

    def __init__(self,
                 x=0,
                 config=0,
                 plot_fig=False):
        self.x = x  # Output samples of the TRL
        self.mod_type = config.link_dict['mod_type']
        self.preamble_taps = np.zeros(22)  # 11 taps or 22 features
        self.preamble_len = 0
        self.window = config.pmf_dict['window']
        self.win_start = config.pmf_dict['win_start']
        self.wlen = 22
        self.xcorr = np.zeros(config.pmf_dict['window'])
        self.plot_fig = plot_fig
        self.peak_idx = 0
        self.peak_val = 0
        self.sign = 1
        self.barker_type = config.pmf_dict['barker_type']

    def _set_preamble_taps(self):
        # Each preamble tap is a feature
        bit_dict = {'fm0': [1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1],
                    'miller': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                               1, 1, 1, -1, -1, 1, 1, -1],
                    '3b4b': [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1,
                             -1, -1, -1, -1, 1, 1, -1, -1]
                    if self.barker_type is 'barker11' else
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1,
                     1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
                    'bpsk': [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1,
                             -1, -1, -1, -1, 1, 1, -1, -1],
                    }

        """Mod type for 3b4b encoding scheme can be baseband or subcarrier
        and hence it is treated as a special case"""
        if '3b4b' in self.mod_type:
            self.preamble_taps = np.array(bit_dict['3b4b'])
        else:
            self.preamble_taps = np.array(bit_dict[self.mod_type])

    def execute(self):
        self._set_preamble_taps()
        self._set_preamble_len()
        self._get_corr()
        self._get_pk_idx_and_val()
        self._get_sign_val()

        if self.plot_fig:
            self._plot_corr()

    def _set_preamble_len(self):
        """Return preamble length in features for FM0, Miller or in bits
         for the other modes"""
        if self.mod_type == 'fm0' or self.mod_type == 'miller':
            self.preamble_len = int(len(self.preamble_taps))
        else:
            self.preamble_len = int(len(self.preamble_taps)//2)

    def _get_corr(self):
        wstart = self.win_start

        if self.mod_type == 'fm0' or self.mod_type == 'miller':
            wlen = self.preamble_len
        else:
            wlen = 2*self.preamble_len

        for i in range(self.window):
            xvec = self.x[wstart+i:wstart+i+wlen*4:4]
            self.xcorr[i] = np.sum(np.multiply(xvec, self.preamble_taps))

    def _get_pk_idx_and_val(self):
        self.peak_idx = np.argmax(np.abs(self.xcorr))
        self.peak_val = self.xcorr[self.peak_idx]

    def _get_sign_val(self):
        if self.peak_val < 0:
            self.sign = -1
        else:
            self.sign = 1

    def _plot_corr(self):
        # Plotting the auto-correlation function
        xmax = np.argmax(self.xcorr)
        wmin = xmax - 16
        wmax = xmax + 16
        fig, ax = plt.subplots()
        ax.plot(self.xcorr)
        ax.set_xlabel('Correlation index')
        ax.set_ylabel('Correlation')
        ax.grid(True)
        plt.axvspan(wmin, wmax, alpha=0.1, color='red')
        plt.show()



