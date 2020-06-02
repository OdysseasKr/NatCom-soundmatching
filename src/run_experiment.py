import tqdm
import random
import multiprocessing
import time
import numpy as np
from deap import creator, base, tools, algorithms
from logger import Logger
from target import TargetGenerator
from utils import plot_results

# Define experiment settings
SEED = 2
EPSILON = 1
sr = 44100
GENE = 'categorical'  # Can be 'categorical' or 'binary'
POP_SIZE = 25
GENERATIONS = 30
TOURNSIZE = 10
PARALLEL = True
N_TARGETS = 2
N_RUNS = 5
DESCRIPTION = 'A sample description for the log file'

if GENE == 'categorical':
    import categorical_ga as ga
else:
    import binary_ga as ga


def get_gen_stats(toolbox, population):
    """
        Returns the stats for a single generation
    """
    best = tools.selBest(population, k=1)
    best = [*toolbox.map(toolbox.evaluate, best)][0][0]
    worst = tools.selWorst(population, k=1)
    worst = [*toolbox.map(toolbox.evaluate, worst)][0][0]
    return {'best': float(best), 'worst': float(worst)}

def run_evolutionary_algorithm(toolbox, n_generations=GENERATIONS, population_size=POP_SIZE,
                               tournament_size=TOURNSIZE, crossover_prob=0.5, mutation_prob=0.1):
    """
        Runs the evolutionary algorithm to approximate a single target signal
    """

    population = toolbox.population(n=population_size)
    fitness_vals = toolbox.map(toolbox.evaluate, population)
    gen_stats = []

    start = time.time()
    for gen in range(1, n_generations+1):
        # Add stats of existing population
        gen_stats.append(get_gen_stats(toolbox, population))

        # Perform crossover (with probability cxpb) and mutation (with probability mutpb)
        offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob)
        # Calculate fitness on the offspring
        fitness_vals = list(toolbox.map(toolbox.evaluate, offspring))
        mean_fitness = 0
        for fit, ind in zip(fitness_vals, offspring):
            # Set each individuals fitness manually
            ind.fitness.values = fit
            mean_fitness += fit[0]/len(population)

        # If converged
        if np.min(fitness_vals) < 0 + EPSILON:
            population = offspring
            break

        population = toolbox.select(offspring, k=population_size)
    runtime = time.time() - start
    # Add stats of final population
    gen_stats.append(get_gen_stats(toolbox, population))

    return tools.selBest(population, k=1)[0], gen, runtime, gen_stats


if __name__ == '__main__':
    logger = Logger('../logs', DESCRIPTION)
    logger.set_header({
        'seed': SEED,
        'epsilon': EPSILON,
        'gene': GENE,
        'pop_size': POP_SIZE,
        'max_gen': GENERATIONS,
        'tourn_size': TOURNSIZE,
        'n_targets': N_TARGETS,
        'n_runs': N_RUNS
    })

    # Set seed for reproducibility
    random.seed(SEED)

    # Define fitness function objective (minimisation)
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    # Create target signal generator and the toolbox
    target_generator = TargetGenerator()
    toolbox = ga.get_toolbox(TOURNSIZE)

    # Enable parallel code
    pool = None
    if PARALLEL:
        pool = multiprocessing.Pool()
        toolbox.register('map', pool.map)

    # For logging and plotting

    for i in tqdm.tqdm(range(N_TARGETS), desc="#signals", ncols = 60):
        # Generate target signal and its features
        target_params, target_sound = next(target_generator)
        target_features = ga.extract_features(target_sound)

        logger.set_target(target_params)

        # Register evaluation function (different for every target)
        toolbox.register('evaluate', ga.fitness, target_features=target_features)
        for n in tqdm.tqdm(range(N_RUNS), desc="   #runs", leave=False, ncols=60):
            best_individual, gens, runtime, gen_stats = run_evolutionary_algorithm(toolbox)

            logger.add_run(best=ga.individual_to_params(best_individual),
                           best_fit=ga.fitness(best_individual, target_features)[0],
                           n_gens=gens,
                           early_stopping=(gens < GENERATIONS),
                           runtime=runtime,
                           gen_stats=gen_stats)
        
        logger.calculate_metrics(False)

    logger.close()



    if PARALLEL:
        pool.close()

    plot_results(logger.path)
