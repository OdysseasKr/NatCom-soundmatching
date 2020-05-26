import numpy as np
import synth
import random

sr = 44100
GENE_LABELS = ['osc_1',
               'amp_1',
               'phase_1',
               'osc_2',
               'amp_2',
               'cutoff'
               ]
GENE_VALUES = {
    'osc_1': list(synth.osc_1_options.keys()),
    'amp_1': np.arange(0.3, 0.8, 0.1),
    'phase_1': np.arange(0, 0.5, 0.1),
    'osc_2': list(synth.osc_2_options.keys()),
    'amp_2': np.arange(0.3, 0.8, 0.1),
    'cutoff': [2500, 5000, 7500, 10000]
}


class TargetGenerator(object):
    def __init__(self, seed=1337):
        super().__init__()
        self.synth = synth.Synth(sr=44100)
        self.r = random.Random(seed)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return self.get_random_values()

    def get_random_values(self):
        target_params = {}
        for k in GENE_VALUES:
            target_params[k] = self.r.choice(GENE_VALUES[k])

        target_params['amp_2'] = 1 - target_params['amp_1']

        self.synth.set_parameters(**target_params)
        target_sound = self.synth.get_sound_array()

        return target_params, target_sound
