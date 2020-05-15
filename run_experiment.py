import random
import multiprocessing
import time
import numpy as np
import synth
from deap import creator, base, tools, algorithms
import librosa
import matplotlib.pyplot as plt
from utils import num_of_digits

# Define experiment settings
SEED = 1
EPSILON = 1

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


def calculate_gene_size():
    """
        Calculate the necessary gene size
    """
    gene_size = 0
    for l in GENE_LABELS:
        digits = num_of_digits(len(GENE_VALUES[l])-1)
        gene_size += digits
    return gene_size


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


def fitness_features(pred_params, target_features):
    """
        Fitness function based on features
        Computes the mean squared error between the feature
        vectors of two signals
    """
    pred_synth = synth.Synth(sr=sr)
    pred_params = dict(zip(GENE_LABELS, pred_params))
    pred_synth.set_parameters(**pred_params)
    
    pred_signal = pred_synth.get_sound_array()
    pred_features = extract_features(pred_signal)
    return np.mean(np.square(pred_features - target_features)),


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


def register_binary_individual(toolbox):
    toolbox.register('attr_bin', lambda: random.choice(['0', '1']))
    # Creates an individual with a random value for each parameter (based on the definition of GENE_VALUES)
    gene_size = calculate_gene_size()
    toolbox.register('individual',
                    tools.initRepeat,
                    creator.Individual,
                    toolbox.attr_bin,
                    gene_size)


def register_categorial_individual(toolbox):

    # Create gene expression
    toolbox.register('attr_osc_1', lambda: random.choice(GENE_VALUES['osc_1']))
    toolbox.register('attr_amp_1', lambda: random.choice(GENE_VALUES['amp_1']))
    toolbox.register('attr_phase_1', lambda: random.choice(GENE_VALUES['phase_1']))
    toolbox.register('attr_osc_2', lambda: random.choice(GENE_VALUES['osc_2']))
    toolbox.register('attr_amp_2', lambda: random.choice(GENE_VALUES['amp_1']))
    toolbox.register('attr_cutoff', lambda: random.choice(GENE_VALUES['cutoff']))

    attr_tuple = (toolbox.attr_osc_1,
                  toolbox.attr_amp_1,
                  toolbox.attr_phase_1,
                  toolbox.attr_osc_2,
                  toolbox.attr_amp_2,
                  toolbox.attr_cutoff
                  )

    # Creates an individual with a random value for each parameter (based on the definition of GENE_VALUES)
    toolbox.register('individual',
                    tools.initCycle,
                    creator.Individual,
                    attr_tuple,
                    1)

def get_toolbox(tournament_size, is_binary, do_parallel=PARALLEL):
    """
        Sets the parameters for the experiment and returns the toolbox
    """
    toolbox = base.Toolbox()

    # Create gene expression
    if is_binary:
        register_binary_individual(toolbox)
    else:
        register_categorial_individual(toolbox)
        
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # Define custom mate and mutate functions
    toolbox.register('mate', custom_mate)
    toolbox.register('mutate', custom_mutate)
    # Tournament selection
    toolbox.register('select', tools.selTournament, tournsize=tournament_size)
    # Make it parallel AND FAST
    if do_parallel:
        toolbox.register('map', multiprocessing.Pool().map)
    return toolbox


def run_evolutionary_algorithm(n_generations=GENERATIONS, population_size=POP_SIZE, 
                               tournament_size=TOURNSIZE, crossover_prob=0.5, mutation_prob=0.1):
    """
        Runs the evolutionary algorithm to approximate a single target signal
    """

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

        # If it converged
        if np.min(fitness_vals) < 0 + EPSILON:
            population = offspring
            break

        population = toolbox.select(offspring, k=population_size)
    print('Total runtime: {:.2f} seconds'.format(time.time()-start))
    return tools.selBest(population, k=1)


class TargetGenerator(object):
    def __init__(self, synth):
        super().__init__()
        self.synth = synth

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return self.get_random_values()

    def get_random_values(self):
        target_params = {}
        for k in GENE_VALUES:
            target_params[k] = random.choice(GENE_VALUES[k])

        target_params['amp_2'] = 1 - target_params['amp_1']

        target_synth.set_parameters(**target_params)
        target_sound = target_synth.get_sound_array()

        return target_params, target_sound

if __name__ == '__main__':
    # Set seed for reproducibility
    random.seed(SEED)
    N_TARGETS = 2

    # Define fitness function objective (minimisation)
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    target_synth = synth.Synth(sr=sr)
    target_generator = TargetGenerator(target_synth)

    toolbox = get_toolbox(TOURNSIZE, is_binary=False)

    target_sounds, best_individuals, best_fitnesses = [], [], []

    for i in range(N_TARGETS):
        print("Target "+str(i))
        target_params, target_sound = next(target_generator)
        target_sounds.append(target_sound)
        target_features = extract_features(target_sound)
        toolbox.register('evaluate', fitness_features, target_features=target_features)

        best_individual = run_evolutionary_algorithm()[0]

        best_individuals.append(best_individual)
        best_fitnesses.append(fitness_features(best_individual, target_features)[0])
    
    print(best_fitnesses)
    for i,best_individual in enumerate(best_individuals):
        plt.grid(True, zorder=0)
        plt.plot(target_sounds[i][:300], label='Target', zorder=2)
        # Plot predicted signal
        mysynth = synth.Synth(sr=sr)
        params = dict(zip(GENE_LABELS, best_individual))
        mysynth.set_parameters(**params)
        soundarray = mysynth.get_sound_array()
        plt.plot(soundarray[:300], linewidth=1.2, label='Prediction', zorder=2)
        plt.legend()
        plt.show()
