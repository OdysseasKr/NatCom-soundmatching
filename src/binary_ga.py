import random
import numpy as np
import synth
from target import GENE_LABELS, GENE_VALUES
from deap import creator, base, tools
import librosa

# Define experiment settings
sr = 44100


def int_to_bin(x):
    """
        Converts integer to list of binary digits
    """
    return list(str(bin(x))[2:])


def bin_to_int(x):
    """
        Converts a list of binary digits to an integer
    """
    x = ''.join(x)
    return int(x, 2)


def num_of_digits(x):
    """
        Calculates how many digits are necessary to represent x in binary
    """
    return len(int_to_bin(x))


# Calculate the individual size
individual_size = 0
for l in GENE_LABELS:
    digits = num_of_digits(len(GENE_VALUES[l])-1)
    individual_size += digits


def individual_to_params(individual):
    """
        Converts a binary individual to a dictionary of parameter values
    """
    start = 0
    params = {}
    for l in GENE_LABELS:
        digits = num_of_digits(len(GENE_VALUES[l])-1)
        part = individual[start: digits+start]
        pos = min(bin_to_int(part), len(GENE_VALUES[l])-1)
        params[l] = GENE_VALUES[l][pos]
        start += digits
    return params


def params_to_individual(params):
    """
        Converts the dictionary of parameters to an individual of binary genes
    """
    gene = []
    for l in GENE_LABELS:
        val = GENE_VALUES[l].index(params[l])
        gene.extend(int_to_bin(val))
    return gene


def extract_features(sound_array):
    """
        Extracts MFCC and spectral bandwidth, centroid, flatness, and roll-off
        It seems that only MFCC features already perform quite well
    """
    return librosa.feature.mfcc(sound_array, sr).flatten()


def fitness(individual, target_features):
    """
        Fitness function based on features
        Computes the mean squared error between the feature
        vectors of two signals
    """
    pred_synth = synth.Synth(sr=sr)
    pred_params = individual_to_params(individual)
    pred_synth = synth.Synth(sr=sr)
    pred_synth.set_parameters(**pred_params)
    soundarray = pred_synth.get_sound_array()
    f_vector = extract_features(soundarray)
    return np.mean(np.square(f_vector - target_features)),


def mate(individual1, individual2, k):
    """
        Mating function.
        Swaps k parameters between individuals (k-point crossover)
    """
    pos = random.sample(range(len(individual1)), k)
    for i in pos:
        individual1[i], individual2[i] = individual2[i], individual1[i]
    return individual1, individual2


def mutate(individual, k):
    """
        Mutation function.
        Randomly modifies k parameters
    """
    pos = random.sample(range(len(individual)), k)
    for i in pos:
        individual[i] = individual[i] = '1' if individual[i] == '0' else '0'
    return individual,


def get_toolbox(tournament_size):
    """
        Sets the parameters for the experiment and returns the toolbox
    """
    toolbox = base.Toolbox()

    # Create gene expression
    toolbox.register('attr_bin', lambda: random.choice(['0', '1']))
    toolbox.register('individual',
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.attr_bin,
                     individual_size)

    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # Define custom mate and mutate functions
    toolbox.register('mate', mate, k=5)
    toolbox.register('mutate', mutate, k=2)
    # Tournament selection
    toolbox.register('select', tools.selTournament, tournsize=tournament_size)

    return toolbox
