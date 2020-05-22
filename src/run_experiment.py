import os
import random
import multiprocessing
import time
import numpy as np
import pandas as pd
import synth
from deap import creator, base, tools, algorithms
import librosa
import matplotlib.pyplot as plt
from utils import num_of_digits
from logger import Logger

# Define experiment settings
SEED = 1
EPSILON = 1
sr = 44100
GENE = 'binary' # Can be 'categorical' or 'binary'
POP_SIZE = 10
GENERATIONS = 2
TOURNSIZE = 3
PARALLEL = True
N_TARGETS = 2

if GENE == 'categorical':
    import categorical_ga as ga
else:
    import binary_ga as ga

def run_evolutionary_algorithm(toolbox, n_generations=GENERATIONS, population_size=POP_SIZE,
                               tournament_size=TOURNSIZE, crossover_prob=0.5, mutation_prob=0.1):
    """
        Runs the evolutionary algorithm to approximate a single target signal
    """

    population = toolbox.population(n=population_size)
    fitness_vals = toolbox.map(toolbox.evaluate, population)
    logger.write(f'Initial population: {population}')
    logger.write('Mean error of the initial population = {:.3f}\n'.format(np.mean(list(fitness_vals))))

    start = time.time()
    for gen in range(1, n_generations+1):
        logger.write(f'Generation {gen}')
        # Perform crossover (with probability cxpb) and mutation (with probability mutpb)
        offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob)
        logger.write(f'\tOffspring (after crossover and mutation): {offspring}')
        # Calculate fitness on the offspring
        fitness_vals = list(toolbox.map(toolbox.evaluate, offspring))
        logger.write(f'\tFitness values: {[fit[0] for fit in fitness_vals]}')
        mean_fitness = 0
        for fit, ind in zip(fitness_vals, offspring):
            # Set each individuals fitness manually
            ind.fitness.values = fit
            mean_fitness += fit[0]/len(population)
        logger.write('\t{:20s} {:5.3f}'.format('Mean error:', np.mean(fitness_vals)))
        logger.write('\t{:20s} {:5.3f}'.format('Error of best:', np.min(fitness_vals)))

        # If converged
        if np.min(fitness_vals) < 0 + EPSILON:
            logger.write("Generation converged!")
            population = offspring
            break

        population = toolbox.select(offspring, k=population_size)
        logger.write(f'\tNew population after selection: {population}')

    logger.write('Total runtime: {:.2f} seconds'.format(time.time()-start))
    logger.flush()
    return tools.selBest(population, k=1)

class TargetGenerator(object):
    def __init__(self, folder):
        super().__init__()
        self.params = pd.read_csv(os.path.join(folder, 'params.csv'), index_col=False)
        self.sounds = np.load(os.path.join(folder, 'sounds.npy'))
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.sounds):
            return self.get_values()
        else:
            raise StopIteration

    def get_values(self):
        target_params = self.params.iloc[i].to_dict()
        target_sound = self.sounds[i]
        self.i += 1

        return target_params, target_sound

if __name__ == '__main__':
    # Set seed for reproducibility
    random.seed(SEED)
    # How many signals to approximate
    logger = Logger("../logs")

    # Define fitness function objective (minimisation)
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    # Create target signal generator and the toolbox
    target_generator = TargetGenerator("../dataset")
    toolbox = ga.get_toolbox(TOURNSIZE)

    # Enable parallel code
    pool = None
    if PARALLEL:
        pool = multiprocessing.Pool()
        toolbox.register('map', pool.map)

    # For logging and plotting
    target_params_list, target_sounds, best_individuals, best_fitnesses = [], [], [], []

    for i in range(N_TARGETS):
        # Generate target signal and its features
        target_params, target_sound = next(target_generator)
        target_features = ga.extract_features(target_sound)

        logger.write(f'TARGET {i+1}: {target_params}')
        target_params_list.append(list(target_params.values()))
        target_sounds.append(target_sound)

        # Register evaluation function (different for every target)
        toolbox.register('evaluate', ga.fitness, target_features=target_features)

        best_individual = run_evolutionary_algorithm(toolbox)[0]

        best_individuals.append(best_individual)
        best_fitnesses.append(ga.fitness(best_individual, target_features)[0])

    logger.write(f'\nAll targets:                    {target_params_list}')
    logger.write(f'Final predictions per target:   {best_individuals}')
    logger.write(f'Best fitness values per target: {best_fitnesses}')
    logger.close()

    # Plot all predictions vs. target (for debugging)
    for i,best_individual in enumerate(best_individuals):
        plt.grid(True, zorder=0)
        plt.plot(target_sounds[i][:300], label='Target', zorder=2)
        # Plot predicted signal
        mysynth = synth.Synth(sr=sr)
        params = ga.individual_to_params(best_individual)
        mysynth.set_parameters(**params)
        soundarray = mysynth.get_sound_array()
        plt.plot(soundarray[:300], linewidth=1.2, label='Prediction', zorder=2)
        plt.legend()
        plt.show()

    if PARALLEL:
        pool.close()
