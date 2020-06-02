import random
import numpy as np
import synth
from target import GENE_LABELS, GENE_VALUES
from deap import creator, base, tools
import librosa
from cachetools import cached, RRCache
from cachetools.keys import hashkey

# Define experiment settings
sr = 44100
cache = RRCache(maxsize=100)


def individual_to_params(individual):
    """
        Converts an individual to a dictionary of parameter values
    """
    return dict(zip(GENE_LABELS, individual))


def extract_features(sound_array):
    """
        Extracts MFCC and spectral bandwidth, centroid, flatness, and roll-off
        It seems that only MFCC features already perform quite well
    """
    return librosa.feature.mfcc(sound_array, sr).flatten()


@cached(cache, key=lambda individual, target_features: hashkey(tuple(individual), tuple(target_features)))
def fitness(individual, target_features):
    """
        Fitness function based on features
        Computes the mean squared error between the feature
        vectors of two signals
    """
    pred_synth = synth.Synth(sr=sr)
    pred_params = individual_to_params(individual)
    pred_synth.set_parameters(**pred_params)

    pred_signal = pred_synth.get_sound_array()
    pred_features = extract_features(pred_signal)
    return np.mean(np.square(pred_features - target_features)),


def mate(individual1, individual2):
    """
        Mating function.
        Swaps one parameter setting between individuals (1-point crossover)
    """
    i = random.randint(0, len(individual1)-1)
    individual1[i], individual2[i] = individual2[i], individual1[i]
    return individual1, individual2


def mutate(individual):
    """
        Mutation function.
        Randomly modifies one of the parameters
    """
    i = random.randint(0, len(individual)-1)
    individual[i] = random.choice(GENE_VALUES[GENE_LABELS[i]])
    return individual,


def get_toolbox(tournament_size):
    """
        Sets the parameters for the experiment and returns the toolbox
    """
    toolbox = base.Toolbox()

    # Create gene expression
    toolbox.register('attr_osc_1', lambda: random.choice(GENE_VALUES['osc_1']))
    toolbox.register('attr_amp_1', lambda: random.choice(GENE_VALUES['amp_1']))
    toolbox.register('attr_phase_1',
                     lambda: random.choice(GENE_VALUES['phase_1']))
    toolbox.register('attr_osc_2', lambda: random.choice(GENE_VALUES['osc_2']))
    toolbox.register('attr_amp_2', lambda: random.choice(GENE_VALUES['amp_1']))
    toolbox.register('attr_cutoff',
                     lambda: random.choice(GENE_VALUES['cutoff']))

    attr_tuple = (toolbox.attr_osc_1,
                  toolbox.attr_amp_1,
                  toolbox.attr_phase_1,
                  toolbox.attr_osc_2,
                  toolbox.attr_amp_2,
                  toolbox.attr_cutoff
                  )

    # Creates an individual with a random value for each parameter
    toolbox.register('individual',
                     tools.initCycle,
                     creator.Individual,
                     attr_tuple,
                     1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # Define custom mate and mutate functions
    toolbox.register('mate', mate)
    toolbox.register('mutate', mutate)
    # Tournament selection
    toolbox.register('select', tools.selTournament, tournsize=tournament_size)

    return toolbox
