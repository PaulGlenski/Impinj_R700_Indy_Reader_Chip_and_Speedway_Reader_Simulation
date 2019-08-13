import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ss


class DmfClass(object):

    def __init__(self, x, plot_fig):
        self.x = x  # Output samples of the TRL
        self.h = np.ones(4)  # DMF filter coefficients
        self.y = np.zeros(len(x))
        self.plot_fig = plot_fig

    def execute(self):
        self.y = ss.lfilter(self.h*1.0, 1.0, self.x*1.0)
        if self.plot_fig:
            self._plot_waveform()

    def _plot_waveform(self):
        fig, ax = plt.subplots()
        ax.plot(self.y, linewidth=2)
        ax.set_title('DMF output')
        ax.grid(True)
        plt.show()
