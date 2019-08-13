from systems_r700.model.src.tag.tag_waveforms import GenerateRevWaveform
from systems_r700.model.src.common.system_config import ConfigClass
from systems_r700.model.src.tag.python_tools.systems_utils.revlink_modes import revlink

class Tag(object):

    def __init__(self, link=revlink['fm0_640kHz_8LF'],
                       wvfm=GenerateRevWaveform(config=ConfigClass(link=revlink['fm0_640kHz_8LF']), EbNo_dB=100, debug=True)):
        self.link = link
        self.config = ConfigClass(link=self.link)
        self.wvfm = wvfm

    def generate_blf(self, bits=30, plot_input=False):

        final_wvfm = self.wvfm.generate_waveform(bits=bits,
                                                 plot=plot_input,
                                                 phase_rad=0.0,
                                                 phase_bits=None,
                                                 lpf_type='FIR')

        return final_wvfm

