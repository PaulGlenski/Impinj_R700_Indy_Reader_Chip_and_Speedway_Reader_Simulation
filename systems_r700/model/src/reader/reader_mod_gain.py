'''
Created on Jan 8, 2018

@author: zchen
'''

from collections import OrderedDict
import numpy as np

class Mod_Gain(object):
    
    tx_rate_decim_dict = \
    {1: 0x57B,
     3: 0x9BD,
     5: 0x703,
     7: 0x728}
    
    def __init__(self, unity_pwr_dbm=30, max_pwr_dbm=30, unity_gain=0x800,
                 pwr_dbm=25, min_pwr_dbm=25, step_size=.25):
        '''
        Mod_gain is short for modulation gain and is a digital attenuation.
        The goal of this digital attenuation is to keep the tx-gain constant
        at the VGA, so that the device only needs one predistortion table for
        pwrs in the non-linear portion of the PA's pout/pin curve. 

        @param unity_pwr_dbm: Power corresponding to unity digital gain
        @param max_pwr_dbm: The highest value in the mod-gain table
        @param unity_gain: Digital value corresponding to unity gain
        @param pwr_dbm: Default reference target power
        @param min_pwr_dbm: The lowest value in the mod-gain table
        @param step_size: The step size of the mod_gain table
        '''
        self.unity_gain = unity_gain
        self.max_pwr_dbm = max_pwr_dbm
        self.unity_pwr_dbm = unity_pwr_dbm
        self.min_pwr_dbm = min_pwr_dbm
        self.pwr = 0
        self.tx_pwr_dbm = pwr_dbm
        self.dec_filt_rate = 1
        self.step_size = step_size
        self.mod_gain_table = OrderedDict()
        self.mod_gain_table_index = OrderedDict()
        self._generate_mod_gain_table()

    def _generate_mod_gain_table(self):
        '''
        Generates mod_gain vs pwr_dbm table. Mod_gain is short for modulation gain
        and is a digital attenuation
        @param span_db: Span in dB of the table
        @param span_step_size: Step size of the table
        '''
        for idx, pwr in enumerate(np.arange(self.min_pwr_dbm, self.max_pwr_dbm + self.step_size, self.step_size)):
            self.mod_gain_table[pwr] = int((10.0**((pwr - self.unity_pwr_dbm)/20.0))*(self.unity_gain))
            self.mod_gain_table_index[pwr] = idx
        
    def _round_pwr_to_nearest_quarterdb(self, pwr):
        """ 
        Given an input pwr, rounds it to nearest quarter dBm 
        for input powers between 25 and 32.5 dBm
        @params:
        - pwr: input pwr
        """
        pwr_floor = float(np.floor(pwr))
        pwr_ceil = float(np.ceil(pwr))
        if pwr_floor == pwr_ceil:
            return pwr
        else:
            pwr_quartiles = np.arange(pwr_floor, pwr_ceil + self.step_size, self.step_size)
            min_delta = 1
            pwr_min = pwr
            for p in pwr_quartiles:
                if np.abs(pwr - p) <= min_delta:
                    min_delta = np.abs(pwr-p)
                    pwr_min = p
            return pwr_min

    def get_mod_gain_table_index(self):
        """
        Returns the current mod-gain table index for the power the mod gain controller is set to
        """
        return self.mod_gain_table_index[self.tx_pwr_dbm]

    def get_mod_gain_val(self):
        """
        Gest the mod gain value for the current tx power
        :return:
        """
        return self.mod_gain_table[self.tx_pwr_dbm];

    def get_mod_gain_output(self, mod_gain_input):
        """ 
        Given an input pwr and the output of the CIC filters, gets the mod_gain_output 
        @params:
        - mod_gain_input: output of the CIC filters
        Returns:
        - mod_gain_output for the given power
        """

        output = ((( (mod_gain_input*self.tx_rate_decim_dict[self.dec_filt_rate]) & 0x7FFFFF) >> 11)*\
        (self.mod_gain_table[self.tx_pwr_dbm])) >> 11
        return output
    
    def get_mod_gain_table(self):
        """
        Returns the mod gain table of target power vs digital attenuation
        """
        return self.mod_gain_table
    
    def get_mod_gain_table_subset(self, nth_item):
        """
        Returns a subset of every nth item in the mod gain table
        """
        key_subset = self.mod_gain_table.keys()[0::nth_item]
        mod_gain_subset = OrderedDict()
        for k in key_subset:
            mod_gain_subset[k] = self.mod_gain_table[k]
        if self.unity_pwr_dbm not in mod_gain_subset:
            mod_gain_subset[self.unity_pwr_dbm] = self.mod_gain_table[self.unity_pwr_dbm]
        return mod_gain_subset
    
    def get_mod_gain_pwrs(self):
        """
        Returns the valid powers in the mod_gain table
        """
        return self.mod_gain_table.keys()

    @property
    def tx_pwr_dbm(self):
        return self.pwr
    @tx_pwr_dbm.setter
    def tx_pwr_dbm(self, pwr_dbm):
        self.pwr = self._round_pwr_to_nearest_quarterdb(pwr_dbm)
        if self.pwr > self.max_pwr_dbm:
            self.pwr = self.max_pwr_dbm
        if self.pwr < self.min_pwr_dbm:
            self.pwr = self.min_pwr_dbm

    
    