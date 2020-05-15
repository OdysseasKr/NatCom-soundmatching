""" Experiment using a binary gene representation

Works by mapping the representation (eg{'osc_1': 'Sine', 'osc_2': 'Square'})
to a list binary digits. This mapping is done in the following manner:

For each parameter, find the index of each value in the list of possible values.
The list of possible values specified in GENE_VALUES.
The index of the element is a integer. Convert this integer to its binary
representation and add it to the gene.

Example for {'osc_1': 'Sine', 'osc_2': 'Square'}:

'Sine' is first in the 'osc_1' list so the binary representation is 000
'Square' is third in the 'osc_2' list so the binary representation is 011

Therefore the gene for these parameters is 000011
"""

import random
import multiprocessing
import time
import numpy as np
import synth
from deap import creator, base, tools, algorithms
import librosa
import matplotlib.pyplot as plt

# Define experiment settings
SEED = 1

sr = 44100

GROUND_TRUTH = {
                'osc_1':'Sine',
                'amp_1':0.5,
                'phase_1':0.2,
                'osc_2':'Sawtooth',
                'amp_2':0.5,
                'cutoff':5000
                }
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
POP_SIZE = 100
GENERATIONS = 10
TOURNSIZE = 3
PARALLEL = False
# Set evaluation type: if 'features', the fitness is computed on the distance between feature vectors
#                      if 'distance', the fitness is computed on the distance between the raw signals
EVALUATION = 'distance'#'features' # It can be 'features' or 'distance'


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


# Calculate the necessary gene size
gene_size = 0
for l in GENE_LABELS:
    digits = num_of_digits(len(GENE_VALUES[l])-1)
    gene_size += digits


def gene_to_params(gene):
    """
        Converts a binary gene to a dictionary of parameter values
    """
    start = 0
    params = {}
    for l in GENE_LABELS:
        digits = num_of_digits(len(GENE_VALUES[l])-1)
        gene_part = gene[start: digits+start]
        ind = min(bin_to_int(gene_part), len(GENE_VALUES[l])-1)
        params[l] = GENE_VALUES[l][ind]
        start += digits
    return params


def params_to_gene(params):
    """
        Converts the dictionary of parameters to a binary gene
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
    # features = []
    # features.extend(librosa.feature.mfcc(sound_array, sr).flatten())        # MFCC
    # features.extend(librosa.feature.spectral_centroid(sound_array, sr)[0])  # Centroid
    # features.extend(librosa.feature.spectral_bandwidth(sound_array, sr)[0]) # Bandwidth
    # features.extend(librosa.feature.spectral_flatness(sound_array)[0])      # Flatness
    # features.extend(librosa.feature.spectral_rolloff(sound_array)[0])       # Rolloff
    return librosa.feature.mfcc(sound_array, sr).flatten()#np.array(features).flatten()


def fitness_features(gene, target):
    """
        Fitness function based on features
        Computes the mean squared error between the feature
        vectors of two signals
    """
    params = gene_to_params(gene)
    mysynth = synth.Synth(sr=sr)
    mysynth.set_parameters(**params)
    soundarray = mysynth.get_sound_array()
    f_vector = extract_features(soundarray)
    return np.sum(np.square(f_vector - target)),


def fitness_signals(gene, target):
    """
        Fitness function that computes the sum squared
        error between two raw signals
    """
    params = gene_to_params(gene)
    mysynth = synth.Synth(sr=sr)
    mysynth.set_parameters(**params)
    soundarray = mysynth.get_sound_array()
    return np.sum(np.square(soundarray[:110] - target[:110])),


def custom_mate(individual1, individual2, indp):
    """
        Mating function.
        Swaps one parameter setting between individuals
    """
    N = len(individual1)
    for i in range(N):
        if random.random() < indp:
            individual1[i], individual2[i] = individual2[i], individual1[i]

    return individual1, individual2


def custom_mutate(individual, indp):
    """
        Mutation function.
        Randomly modifies one of the parameters
    """
    N = len(individual)
    for i in range(N):
        if random.random() < indp:
            individual[i] = '1' if individual[i] == '0' else '0'
    return individual,


def get_toolbox(target, tournament_size, do_parallel=PARALLEL):
    """
        Sets the parameters for the experiment and returns the toolbox
    """
    toolbox = base.Toolbox()

    # Create gene expression
    toolbox.register('attr_bin', lambda: random.choice(['0', '1']))
    # Creates an individual with a random value for each parameter (based on the definition of GENE_VALUES)
    toolbox.register('individual',
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.attr_bin,
                     gene_size)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # If EVALUATION is set to distance, fitness is computed on raw signals
    # If EVALUATION is set to features, fitness is computed on feature vectors of signals
    if EVALUATION == 'distance':
        toolbox.register('evaluate', fitness_signals, target=target)
    else:
        toolbox.register('evaluate', fitness_features, target=extract_features(target))
    # Define custom mate and mutate functions
    toolbox.register('mate', custom_mate, indp=0.1)
    toolbox.register('mutate', custom_mutate, indp=0.1)
    # Tournament selection
    toolbox.register('select', tools.selTournament, tournsize=tournament_size)
    # Make it parallel AND FAST
    if do_parallel:
        toolbox.register('map', multiprocessing.Pool().map)
    return toolbox


def run_evolutionary_algorithm(target, n_generations=GENERATIONS, population_size=POP_SIZE,
                               tournament_size=TOURNSIZE, crossover_prob=0.5, mutation_prob=0.1):
    """
        Runs the evolutionary algorithm to approximate a single target signal
    """
    toolbox = get_toolbox(target, tournament_size)

    population = toolbox.population(n=population_size)
    fitness_vals = toolbox.map(toolbox.evaluate, population)
    print('Mean error of the initial population {:.3f}'.format(np.mean(list(fitness_vals))))

    start = time.time()
    for gen in range(1, n_generations+1):
        # Perform crossover (with probability cxpb) and mutation (with probability mutpb)
        offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob)
        # Calculate fitness on the offspring
        fitness_vals = list(toolbox.map(toolbox.evaluate, offspring))
        mean_fitness = 0
        for fit, ind in zip(fitness_vals, offspring):
            # Set each individuals fitness manually
            ind.fitness.values = fit
            mean_fitness += fit[0]/len(population)
        print('Generation', gen)
        print('\t{:20s} {:5.3f}'.format('Mean error:', np.mean(fitness_vals)))
        print('\t{:20s} {:5.3f}'.format('Error of best:', np.min(fitness_vals)))
        population = toolbox.select(offspring, k=population_size)
    print('Total runtime: {:.2f} seconds'.format(time.time()-start))
    return tools.selBest(population, k=1)


if __name__ == '__main__':
    # Set seed for reproducibility
    random.seed(SEED)

    # Set a target
    mysynth = synth.Synth(sr=sr)
    mysynth.set_parameters(**GROUND_TRUTH)
    target_sound = mysynth.get_sound_array()

    # Define fitness function objective (minimisation)
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    # Retrieve the best approximators of the target
    best_individual = run_evolutionary_algorithm(target_sound)[0]
    params = gene_to_params(best_individual)
    print('BEST', params)

    # Plot the ground truth
    plt.grid(True, zorder=0)
    plt.plot(target_sound[:300], label='Target', zorder=2)
    # Plot predicted signal
    mysynth = synth.Synth(sr=sr)
    mysynth.set_parameters(**params)
    soundarray = mysynth.get_sound_array()
    plt.plot(soundarray[:300], linewidth=1.2, label='Prediction', zorder=2)
    plt.legend()
    plt.show()
