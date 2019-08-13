'''
Created on Nov 2, 2017

@author: zchen
'''

import numpy as np
from systems_r700.model.src.reader import vga
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes
import matplotlib.pyplot as plt
from systems_r700.model.src.common import simple_plot as sp

attrib = ReaderAttributes()
input_voltages = np.arange(0, 1.3, .1)
gc_voltages = np.arange(0, 2.2, .1)
vga = vga.Vga(attrib.tx_)
plots = []
plotter = sp.SimplePlot(line_style='--')
for iv in input_voltages:
    plot = {"x": [], "y": [], "label": "Modulator Input Voltage: {}".format(iv)}
    for gc in gc_voltages:
        val = vga.process(iv, gc)[0]
        plot["x"].append(gc)
        plot["y"].append(val)
    plots.append(plot)
plotter.multi_plot(plots, "Output Power (V) vs GC (V)", "Gain Control Voltage (V)", "Output Voltage (V)")
plt.show()