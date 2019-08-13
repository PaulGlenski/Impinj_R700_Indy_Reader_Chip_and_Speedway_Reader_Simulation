##
#  This class models one antenna element for the reader.
##

import cmath
import numpy as np
import systems_r700.model.src.common.utils as ut
from systems_r700.model.src.common.phase_noise import PhaseNoise
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes



class ReaderAntenna(object):

    # constructor
    def __init__(self, reflection_loss_db=-10.0, reflection_phase=0.0):
        # self.reflection_loss_db = reflection_loss_db = -10.0
        # self.reflection_phase = reflection_phase = 0.0
        print(reflection_phase)
        self.reflection_gain_ = ut.db2lin(reflection_loss_db) * cmath.exp(1j * reflection_phase)
        self.in_ = []
        self.out_ = []

    def cc_start_process(self, tx):
        self.in_.append(tx)
        y = self.reflection_gain_ * tx
        return y

    def process(self, tx, antenna_in):
        self.in_.append(tx)
        self.in_.append(antenna_in)
        y = self.reflection_gain_ * tx + antenna_in
        return y
        #returns an array of the tx self-jammer wave with its reflection gain and with the rx blf wave added

    def batch_process(self, tx, antenna_in):
        y = self.reflection_gain_ * tx + antenna_in
        return y



