##
#  This class models an ADC

import numpy as np

class Adc(object):

    # constructor
    def __init__(self, v_analog_range=1.0, num_bits=12, bipolar=True):
        # set ranges and steps
        if bipolar:
            self.v_min_ = -v_analog_range/2.0
            self.v_max_ =  v_analog_range/2.0
        else:
            self.v_min_ = 0.0
            self.v_max_ = v_analog_range
        num_steps = (2 ** num_bits)
        self.v_step_ = (self.v_max_ - self.v_min_)/num_steps

        # initialize clipped state
        self.clipped_ = False

    # main processing
    def \
            process(self, v_in):
        self.clipped_ = False

        if isinstance(v_in, np.ndarray) == True:
            output_list = []
            for value in v_in:
            #clip the input range
                if value > (self.v_max_ - self.v_step_):
                    value = self.v_max_ - self.v_step_
                    self.clipped_ = True
                else:
                    if value < self.v_min_:
                        value = self.v_min_
                        self.clipped_ = True
                adc_out_single = int(round(value/self.v_step_))
                output_list.append(adc_out_single)
            adc_out = output_list

        else:
        # clip the input range
            if v_in > (self.v_max_ - self.v_step_):
                v_in = self.v_max_ - self.v_step_
                self.clipped_ = True
            else:
                if v_in < self.v_min_:
                    v_in = self.v_min_
                    self.clipped_ = True
            adc_out = int(round(v_in / self.v_step_))

        return adc_out
