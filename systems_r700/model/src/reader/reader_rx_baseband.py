##
#  This class models the reader receive baseband processing.
##

from systems_r700.model.src.reader.reader_attributes import ReaderAttributes


class ReaderRxBaseband(object):
    # constructor
    def __init__(self, attrib=ReaderAttributes()):
        # Placeholder for now
        self.cc_correction_ = 1 + 1j*0

    def cc_process(self, sample):
        return sample
