import numpy as np

from python_tools.systems_utils.decode_3b4b import Decode3b4bClass
from python_tools.systems_utils.bd_decode import BDDecodeClass
from python_tools.systems_utils.decode_bits import DecodeBitsClass

class DecodeSymbolClass(object):
    def __init__(self,
                 x=0,
                 orig=0,
                 config=0,
                 # mod_type=0,
                 last_word=0
                 ):

        self.x=x
        self.y=None
        self.orig=orig
        self.mod_type=config.link_dict['mod_type']
        self.hard_decision=config.decode_dict['hard_decision_decoding']
        self.last_word=last_word
        self.ber = 0
        self.decode_symbol=None
        self.bit_err_loc = None

    def execute(self):
        if '3b4b' in self.mod_type:
            self.decode_symbol = Decode3b4bClass(x = self.x,
                                                 mod_type=self.mod_type,
                                                 hard_decision=self.hard_decision
                                                 )
            self.decode_symbol.execute()
            self.y = self.decode_symbol.y
        elif 'fm0' in self.mod_type or 'miller' in self.mod_type:
            self.decode_symbol = BDDecodeClass(x = self.x,
                                               mod_type=self.mod_type)
            self.decode_symbol.execute()
            self.y = self.decode_symbol.y
        elif 'bpsk' in self.mod_type:
            self.y = self.x

        #self.bit_errs = self.find_ber()

    def find_ber(self):
        y = self.y
        if '3b4b' in self.mod_type:
            last_word = int(self.last_word * 3./4)
        else:
            last_word = self.last_word

        x = self.orig[:-last_word]

        self.bit_err_loc = np.zeros(len(x))

        if len(y) != len(x):
            raise ValueError("Incompatible input output lengths")

        self.bit_err_loc = np.abs(y-x)

        return np.sum(np.abs(y - x))