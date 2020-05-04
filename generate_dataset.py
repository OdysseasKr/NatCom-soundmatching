from synth import Synth, osc_1_options, osc_2_options
import random
import sys
import numpy as np
import pandas as pd

# Range of values for each parameter of the synt
values = {
    'osc_1': list(osc_1_options.keys()),
    'osc_2': list(osc_2_options.keys()),
    'amp_1': np.arange(0.3, 0.8, 0.1),
    'phase_1': np.arange(0, 0.5, 0.1),
    'cutoff': [2500, 5000, 7500, 10000]
}

# The number of samples to generate
num_of_samples = 400

# Column names for the final CSV
columns = ['osc_1', 'osc_2', 'amp_1', 'amp_2', 'phase_1', 'cutoff']


# Gets a set of random parameters given the options
def get_random_values():
    params = {}
    for k in values:
        params[k] = random.choice(values[k])

    params['amp_2'] = 1 - params['amp_1']
    return params


def main():
    mysynth = Synth()
    labels = []  # Holds the parameter values
    sounds = []  # Holds the sound arrays

    # Get a number of random sounds
    i = 0
    while i < num_of_samples:
        print('{} of {}'.format(i, num_of_samples), end='\r')
        sys.stdout.flush()
        params = get_random_values()
        param_vector = [params[c] for c in columns]
        if param_vector in labels:
            continue

        mysynth.set_parameters(**params)
        sound = mysynth.get_sound_array()

        sounds.append(sound)
        labels.append(param_vector)
        i += 1

    # Save sounds and parameters
    np.save('sounds.npy', sounds)
    df = pd.DataFrame(labels, columns=columns)
    df.to_csv('params.csv', index=False)


if __name__ == '__main__':
    main()
