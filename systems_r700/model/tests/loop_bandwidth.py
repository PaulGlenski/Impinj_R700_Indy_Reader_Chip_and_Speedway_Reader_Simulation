'''
Created on Oct 19, 2017

@author: zchen
'''

import numpy as np
import matplotlib.pyplot as plt
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes
from systems_r700.model.src.simulation_engine.simulation_engine import SimulationEngine

plt.close('all')

attrib = ReaderAttributes()
attrib.tx_.tx_pwr_dbm_ = 25.0
num_iterations = []
errors = []
loop_gains = np.arange(100, 7000, 100)

for i in loop_gains:
    attrib.tx_.tx_ramp_up_loop_gain = i
    sim_eng = SimulationEngine(attrib)
    sim_eng.process_tx_rampup()
    tx_pwr = np.array(sim_eng.reader_.rfa_.tx_pwr)
    num_iterations.append(len(tx_pwr))
    errors.append(abs(tx_pwr[-1] - attrib.tx_.tx_pwr_dbm_))

plt.figure(1)
plt.plot(loop_gains, num_iterations)
plt.xlabel("Loop Gain")
plt.ylabel("Iterations to Converge")
plt.title("Iterations to Convergence vs Loop Gain")

plt.figure(2)
plt.plot(loop_gains, errors)
plt.xlabel("Loop Gain")
plt.ylabel("Power Erors")
plt.title("Iterations to Convergence vs Loop Gain @ pwr = {} dBm".format(attrib.tx_.tx_pwr_dbm_))
plt.grid(True)
plt.show()