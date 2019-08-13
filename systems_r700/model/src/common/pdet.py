import numpy as np

class Pdet(object):
    """ RFDET_Component. Child of Component Class. Linear model of voltage_out vs p_in 
        Takes in the following parameters:
        - in_intercept_dBm: This is the x intercept of the voltage_out vs p_in curve, (i.e out_slope_v_per_dB*(p_in - in_intercept_dBm) = 0)
        - out_slope_v_per_dB: The slope of the voltage_out vs p_in curve, in volts per dB
        - input_gain_dB: Input attenuation (voltage divider)
    """
    def __init__(self, in_intercept_dBm=-57, out_slope_v_per_dB=.0255, input_gain_dB=-36.6, input_impedance=50):
        self.input_gain_dB = float(input_gain_dB)
        self.in_intercept_dBm = float(in_intercept_dBm)
        self.out_slope_v_per_dB = float(out_slope_v_per_dB)
        self.input_impedance = float(input_impedance)
    
    #Return output in Volts
    def process(self, input_V):
        input_V = float(input_V)
        input_pwr_dBm = 10.0*np.log10(input_V**2/self.input_impedance) + 30 + self.input_gain_dB
        if input_pwr_dBm < self.in_intercept_dBm:
            output = 0
        else:
            output = (input_pwr_dBm - self.in_intercept_dBm)*self.out_slope_v_per_dB
        return output