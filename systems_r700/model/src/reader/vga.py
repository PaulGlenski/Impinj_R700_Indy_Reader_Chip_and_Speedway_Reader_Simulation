from scipy.interpolate import interp1d, interp2d
import numpy as np
import pandas as pd
from systems_r700.model.src.common import utils as ut


class Vga(object):
    """ VGA Component class. Generates Vout vs GC curves at specific input voltages
        Params:
        - gain_control_voltages: Array of voltage values used for gain control
        - output_voltages: Modulator output voltage for a corresponding gain control voltage
        - input_vref: Input voltage reference for the output_voltage vs gain_control_voltage curve
        - sautration_gc: 1dB compresson point of output voltage
    """
    def __init__(self, tx_):
        try:
            df = tx_.vga_characteristics
            y = pd.unique(df.mag)
            x = pd.unique(df.gain_control)
            z = []
            
            # Measurements were taken post_vga
            post_vga_gain = ut.db2lin(tx_.pa_driver_amp_gain_db + \
                            tx_.saw_filter_gain_db + \
                            tx_.pa_gain_db + \
                            tx_.bi_d_coupler_gain_db)
            for mag in y:
                f_y = []
                f_xy = interp1d(df[df.mag == mag]['gain_control'].get_values(), 
                                                  df[df.mag == mag]['output_voltage'].get_values(),
                                                  fill_value='extrapolate')
                for gc in x:
                    val = f_xy(gc)/post_vga_gain
                    f_y.append(val)
                z.append(f_y) 
        except:
            print("Error initializing VGA")
            x = [0]
            y = [0]
            z = [0]
        self.vga_function = interp2d(x, y, z)

    
    def process(self, input_voltage, gc_voltage):
        """ Gets the RF Modulator output Vrms given an input voltage and a gain control voltage
            Params:
            - input_voltage: Input into VGA from MOD_GAIN lookup table
            - gc_voltage: Gain control voltage passed in from the digital error block
        """
        input_volt_real = np.real(input_voltage)
        input_volt_imag = np.imag(input_voltage)
        return self.vga_function(gc_voltage, input_volt_real) + 1j*self.vga_function(gc_voltage, input_volt_imag)
