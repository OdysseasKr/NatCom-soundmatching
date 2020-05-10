import random
import multiprocessing
import time
import numpy as np
import synth
from deap import creator, base, tools, algorithms
import librosa
import matplotlib.pyplot as plt

# Define gene shape
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
GENE_SIZE = len(GENE_LABELS)
POP_SIZE = 100
GENERATIONS = 10
TOURNSIZE = 3
PARALLEL = True

# Define fitness function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def extract_features(sound_array):
    centroid = librosa.feature.spectral_centroid(sound_array, sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(sound_array, sr)[0]
    flatness = librosa.feature.spectral_flatness(sound_array)[0]
    rolloff = librosa.feature.spectral_rolloff(sound_array)[0]
    result = np.array([centroid, bandwidth, flatness, rolloff])
    return result.flatten()


def custom_evaluate(params, target):
    """Sum of squared error evaluation function"""
    mysynth = synth.Synth(sr=sr)
    params = dict(zip(GENE_LABELS, params))
    mysynth.set_parameters(**params)
    soundarray = mysynth.get_sound_array()
    f_vector = extract_features(soundarray)
    return np.sum(np.square(f_vector - target)),


def distance_evaluate(params, target):
    mysynth = synth.Synth(sr=sr)
    params = dict(zip(GENE_LABELS, params))
    mysynth.set_parameters(**params)
    soundarray = mysynth.get_sound_array()
    return np.sum(np.square(soundarray[:110] - target[:110])),

def custom_mate(ind1, ind2):
    """Mating function.
    Selects one of the parameters and swaps them.
    """
    i = random.randint(0, GENE_SIZE-1)
    ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


def custom_mutate(individual):
    """Mutation function.
    Randomly modifies one of the parameters
    """
    i = random.randint(0, GENE_SIZE-1)
    individual[i] = random.choice(GENE_VALUES[GENE_LABELS[i]])
    return individual,


def get_toolbox(target):
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
    toolbox.register('individual',
                     tools.initCycle,
                     creator.Individual,
                     attr_tuple,
                     1)

    # Setup custom operators
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', distance_evaluate, target=target)
    toolbox.register('mate', custom_mate)
    toolbox.register('mutate', custom_mutate)
    # Tournament selection
    toolbox.register('select', tools.selTournament, tournsize=TOURNSIZE)
    # Make it parallel AND FAST
    if PARALLEL:
        toolbox.register('map', multiprocessing.Pool().map)
    return toolbox


def run(target_sound):
    #target = extract_features(target_sound)
    target = target_sound
    toolbox = get_toolbox(target)

    population = toolbox.population(n=30)
    fits = toolbox.map(toolbox.evaluate, population)
    print('Mean error of the initial population', np.mean(list(fits)))

    start = time.time()
    for gen in range(1, GENERATIONS+1):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = list(toolbox.map(toolbox.evaluate, offspring))
        mean_fit = 0
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
            mean_fit += fit[0]/len(population)
        print('Generation', gen)
        print('   Mean error: \t\t', np.mean(fits))
        print('   Error of best: \t', np.min(fits))
        population = toolbox.select(offspring, k=len(population))
    print('Total runtime:', time.time()-start)
    return tools.selBest(population, k=3)


if __name__ == '__main__':
    mysynth = synth.Synth(sr=sr)
    mysynth.set_parameters(osc_1='Sine',
                           amp_1=0.5,
                           phase_1=0.2,
                           osc_2='Sawtooth',
                           amp_2=0.5,
                           cutoff=5000)
    target_sound = mysynth.get_sound_array()
    best = run(target_sound)
    print('BEST', best)
    # Plot results
    plt.plot(target_sound[:300])
    mysynth = synth.Synth(sr=sr)
    params = dict(zip(GENE_LABELS, best[0]))
    mysynth.set_parameters(**params)
    soundarray = mysynth.get_sound_array()
    plt.plot(soundarray[:300])
    plt.show()
