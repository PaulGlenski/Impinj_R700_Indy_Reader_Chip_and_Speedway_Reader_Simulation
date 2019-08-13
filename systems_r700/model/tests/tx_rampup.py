'''
Created on Oct 2, 2017

@author: zchen
'''

import numpy as np
import matplotlib.pyplot as plt
import collections as c
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes
import systems_r700.model.src.simulation_engine.simulation_engine
from scipy.interpolate import interp1d


def open_loop_calibration():
    '''
    Test method to calibrate tx values
    '''
    attrib = ReaderAttributes()
    tx_pwr_array = []
    adc_out_array = []
    for tx_pwr in np.arange(0, 34.0, 1):
        print(tx_pwr)
        attrib.tx_.tx_pwr_dbm_ = tx_pwr
        attrib.tx_.tx_ramp_up_loop_gain = 2000
        sim_eng = systems_r700.model.src.simulation_engine.simulation_engine.SimulationEngine(attrib)
        for tx_gain in np.arange(0, 50000, 10000):
            adc_out = sim_eng.process_tx_calibration(tx_gain)
            tx_pwr = sim_eng.reader_.rfa_.tx_pwr[-1]
            print("adc_out = {}, tx_pwr = {}".format(adc_out, tx_pwr))
            adc_out_array.append(adc_out)
            tx_pwr_array.append(tx_pwr)
    cal_table = generate_cal_table(tx_pwr_array, adc_out_array)
    print_cal_table_yaml(cal_table)

def generate_cal_table(tx_pwr_array, adc_out_array):
    print(tx_pwr_array)
    adc_of_tx_pwr = interp1d(tx_pwr_array, adc_out_array, assume_sorted=False, fill_value='extrapolate')
    pwr_dict = c.OrderedDict()
    for tx_pwr in np.arange(0, 35, 3):
        pwr_dict[tx_pwr] = int(adc_of_tx_pwr(tx_pwr))
    return pwr_dict
            

def closed_loop_rampup():
    #Setup for closed loop rampup
    attrib = ReaderAttributes()
    tx_pwr = 30.0
    attrib.tx_.tx_pwr_dbm_ = tx_pwr
    attrib.tx_.tx_ramp_up_loop_gain = 768
    attrib.tx_.code_delay_us = 4.2
    sim_eng = systems_r700.model.src.simulation_engine.simulation_engine.SimulationEngine(attrib)
    plt.close('all')
    sim_eng.process_tx_rampup()
    sim_eng.process_tx_rampdown()
    plt.figure(1)
    tx_pwr = np.array(sim_eng.reader_.rfa_.tx_pwr)
    ts = 1/attrib.rx_.fs_
    t = np.arange(0, len(tx_pwr))*ts
    plt.plot(t, tx_pwr)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Tx Power dBm")
    plt.title("Tx Power vs Time @ Pwr = {}".format(attrib.tx_.tx_pwr_dbm_))
    plt.grid(True)
    
    plt.figure(2)
    
    tx_pwr_sampled = np.array(sim_eng.reader_.rfa_.tx_pwr_sampled)
    plt.plot(tx_pwr_sampled)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Tx Power dBm")
    plt.title("Tx Power vs Microblaze Iterations @ Pwr = {}".format(attrib.tx_.tx_pwr_dbm_))
    plt.grid(True)
    plt.hold(True)
    plt.show()

def print_cal_table_yaml(cal_table):
    for k in (cal_table):
        print( "{} : {}".format(k, cal_table[k]))

def main():
    closed_loop_rampup()
#     open_loop_calibration()
    


if __name__ == "__main__":
    main()