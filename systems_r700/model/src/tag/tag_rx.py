"""
This module provides a python golden model for the Mag block
"""

# Impinj imports
from python_tools.gmbase import GoldenModelBase


class CmTagRx(GoldenModelBase):
    """
    Tag Receiver (TAGRX)

    - Quadrature inputs
    - Block takes the magnitude of the inputs and slices the magnitude to
      convert the inputs to a single bit output.
    - 1 bit unsigned output

    """
    def __init__(self, block):
        # Initialize base class
        GoldenModelBase.__init__(self, block)

    def generate_verilog_files(self, path=''):
        """
        Generates the accompanying verilog test files for this block
        """
        self.cm_mag.generate_verilog_files(path=path)

    def execute(self, x_ic, x_qc, decay, float_calculation=False):
        """
        Execute the tagrx function
        """

        # Set inputs to new bit widths
        self.cm_mag.x_i.set_real_value(x_ic.get_real_value())
        self.cm_mag.x_q.set_real_value(x_qc.get_real_value())

        # Execute magnitude block
        y_mag = self.cm_mag.execute(self.cm_mag.x_i, self.cm_mag.x_q,
                                    float_calculation=float_calculation)

        # Execute slicer
        self.y = self.cm_slicer.execute(y_mag, decay,
                                        float_calculation=float_calculation)

        return self.y
