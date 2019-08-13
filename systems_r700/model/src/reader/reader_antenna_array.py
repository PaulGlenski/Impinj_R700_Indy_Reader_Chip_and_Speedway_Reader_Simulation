##
#  This class models the antenna array for the reader.
##

from systems_r700.model.src.reader.reader_antenna import ReaderAntenna
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes

import numpy as np


class ReaderAntennaArray:

    # constructor
    def __init__(self, attrib=ReaderAttributes()):
        self.selected_ = 0
        self.antenna_ = [ReaderAntenna(attrib.rx_.return_loss_db_[x], attrib.rx_.reflection_phase_[x])
                        for x in range(attrib.rx_.num_antennas_)]
        print("reflection phase", attrib.rx_.reflection_phase_[0])
    def select_antenna(self, sel):
        self.selected_ = sel

    def process(self, tx, antenna_in=0):
        y = self.antenna_[self.selected_].process(tx, antenna_in)
        return y

    def batch_process(self, tx, antenna_in=None):
        if antenna_in is None:
            antenna_in = np.zeros(len(tx))
        y = self.antenna_[self.selected_].batch_process(tx, antenna_in)
        return y