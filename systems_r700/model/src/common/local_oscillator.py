import numpy as np
from systems_r700.model.src.common.phase_noise import PhaseNoise
from systems_r700.model.src.common.awgn import Awgn
import systems_r700.model.src.common.utils as ut
import matplotlib.pyplot as plt

class LocalOscillator(object):
    # constructor
    def __init__(self, random_init_phase=True, phase_noise_freq_hz=None, phase_noise_dbc=None, fs=1.0):
        self.phase_noise = PhaseNoise(phase_noise_freq_hz, phase_noise_dbc, fs)
        self.awgn = Awgn()
        self.random_init_phase = random_init_phase

        if random_init_phase:
            init_phase = np.random.uniform(0, 2 * np.pi)
        else:
            init_phase = 0.0
        self.init_phase = init_phase
        self.carrier_phasor = np.exp(1j * init_phase)

    # main processing
    def process(self):
        y = self.phase_noise.add_noise(self.carrier_phasor)

        return y

    def batch_process(self, num_samples):
        noise_garbage = self.phase_noise.gen_noise(num_samples)
        y = self.carrier_phasor * noise_garbage
        return y

