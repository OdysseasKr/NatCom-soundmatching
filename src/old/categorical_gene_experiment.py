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
PARALLEL = True
# Set evaluation type: if 'features', the fitness is computed on the distance between feature vectors
#                      if 'distance', the fitness is computed on the distance between the raw signals 
EVALUATION = 'distance'#'features' # It can be 'features' or 'distance'


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


def fitness_features(params, target):
    """
        Fitness function based on features
        Computes the mean squared error between the feature
        vectors of two signals
    """
    mysynth = synth.Synth(sr=sr)
    params = dict(zip(GENE_LABELS, params))
    mysynth.set_parameters(**params)
    soundarray = mysynth.get_sound_array()
    f_vector = extract_features(soundarray)
    return np.mean(np.square(f_vector - target)),


def fitness_signals(params, target):
    """
        Fitness function that computes the sum squared
        error between two raw signals
    """
    mysynth = synth.Synth(sr=sr)
    params = dict(zip(GENE_LABELS, params))
    mysynth.set_parameters(**params)
    soundarray = mysynth.get_sound_array()
    return np.sum(np.square(soundarray[:110] - target[:110])),

def custom_mate(individual1, individual2):
    """
        Mating function.
        Swaps one parameter setting between individuals
    """
    i = random.randint(0, len(individual1)-1)
    individual1[i], individual2[i] = individual2[i], individual1[i]
    return individual1, individual2


def custom_mutate(individual):
    """
        Mutation function.
        Randomly modifies one of the parameters
    """
    i = random.randint(0, len(individual)-1)
    individual[i] = random.choice(GENE_VALUES[GENE_LABELS[i]])
    return individual,


def get_toolbox(target, tournament_size, do_parallel=PARALLEL):
    """
        Sets the parameters for the experiment and returns the toolbox
    """
    toolbox = base.Toolbox()

    # Create gene expression
    toolbox.register('attr_osc_1', lambda: random.choice(GENE_VALUES['osc_1']))
    toolbox.register('attr_amp', lambda: random.choice(GENE_VALUES['amp_1']))
    toolbox.register('attr_phase_1', lambda: random.choice(GENE_VALUES['phase_1']))
    toolbox.register('attr_osc_2', lambda: random.choice(GENE_VALUES['osc_2']))
    toolbox.register('attr_cutoff', lambda: random.choice(GENE_VALUES['cutoff']))
    attr_tuple = (toolbox.attr_osc_1,
                  toolbox.attr_amp,
                  toolbox.attr_phase_1,
                  toolbox.attr_osc_2,
                  toolbox.attr_amp,
                  toolbox.attr_cutoff
                  )
    # Creates an individual with a random value for each parameter (based on the definition of GENE_VALUES)
    toolbox.register('individual',
                     tools.initCycle,
                     creator.Individual,
                     attr_tuple,
                     1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # If EVALUATION is set to distance, fitness is computed on raw signals
    # If EVALUATION is set to features, fitness is computed on feature vectors of signals
    if EVALUATION == 'distance':
        toolbox.register('evaluate', fitness_signals, target=target)
    else:
        toolbox.register('evaluate', fitness_features, target=extract_features(target))
    # Define custom mate and mutate functions
    toolbox.register('mate', custom_mate)
    toolbox.register('mutate', custom_mutate)
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
    print('BEST', best_individual)

    # Plot the ground truth
    plt.grid(True, zorder=0)
    plt.plot(target_sound[:300], label='Target', zorder=2)
    # Plot predicted signal
    mysynth = synth.Synth(sr=sr)
    params = dict(zip(GENE_LABELS, best_individual))
    mysynth.set_parameters(**params)
    soundarray = mysynth.get_sound_array()
    plt.plot(soundarray[:300], linewidth=1.2, label='Prediction', zorder=2)
    plt.legend()
    plt.show()
