##
#  These classes encapsulate all the impairments encountered by the system.
##

class ReaderRxImpairments(object):

    # constructor
    def __init__(self):

        # Temperature in degrees Celsius
        # (used to model thermal noise incident on the antennas)
        self.temperature = 20


class ReaderTxImpairments(object):

    # constructor
    def __init__(self):
        # placeholder for now
        self.dummy = 0


class ReaderImpairments(object):

    # constructor
    def __init__(self):
        self.rx = ReaderRxImpairments()
        self.tx = ReaderTxImpairments()
