import argparse
import tqdm
import random
import multiprocessing
import time
import numpy as np
from deap import creator, base, tools, algorithms
from logger import Logger
from target import TargetGenerator

# Define experiment settings
SEED = 2
EPSILON = 10
sr = 44100
GENE = 'categorical'  # Can be 'categorical' or 'binary'
POP_SIZE = 50
GENERATIONS = 30
TOURNSIZE = 3
PARALLEL = True
N_TARGETS = 1
N_RUNS = 5
CROSSOVER_PROB = 0.5
MUTATION_PROB = 0.1
DESCRIPTION = 'Hyper Parameter optimizations 0.2'


### Argument parser ###
parser = argparse.ArgumentParser(description='Run a genetic algorithm for sound matching with specified parameters.')
parser.add_argument('-gene', nargs='+', dest="genes", default = [GENE], help="Gene representation (categorical/binary)")
parser.add_argument('-s', type=int, nargs='?', dest="seed",
                    default = SEED, help='Seed')
parser.add_argument('-e', type=int, nargs='?', dest="epsilon",
                    default = EPSILON, help='Error margin for convergence')
parser.add_argument('-p', type=int, nargs='?', dest="pop_size",
                    default = POP_SIZE, help="Population size")
parser.add_argument('-g', type=int, nargs='?', dest="generations",
                    default = GENERATIONS, help="Number of generations")
parser.add_argument('-t', type=int, nargs='?', dest='tournsize',
                    default = TOURNSIZE, help="Tournament size")
parser.add_argument('-ntargets', type=int, nargs='?', dest='n_targets',
                    default = N_TARGETS, help="Number of targets to approximate")
parser.add_argument('-nruns', type=int, nargs='?', dest='n_runs',
                    default = N_RUNS, help="Number of runs for each target")
parser.add_argument('-cp', type=float, nargs='+', dest='crossover_probs',
                    default = [CROSSOVER_PROB], help="Crossover probabilities to test")
parser.add_argument('-mp', type=float, nargs='+', dest='mutation_probs',
                    default = [MUTATION_PROB], help="Mutation probabilities to test")


if GENE == 'categorical':
    import categorical_ga as ga
else:
    import binary_ga as ga


def get_gen_stats(toolbox, population):
    """
        Returns the stats for a single generation
    """
    best = tools.selBest(population, k=1)[0].fitness.values[0]
    worst = tools.selWorst(population, k=1)[0].fitness.values[0]
    return {'best': float(best), 'worst': float(worst)}

def run_evolutionary_algorithm(toolbox,
                               n_generations=GENERATIONS,
                               population_size=POP_SIZE,
                               tournament_size=TOURNSIZE,
                               crossover_prob=CROSSOVER_PROB,
                               mutation_prob=MUTATION_PROB):
    """
        Runs the evolutionary algorithm to approximate a single target signal
    """

    population = toolbox.population(n=population_size)
    fitness_vals = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitness_vals):
        ind.fitness.values = fit
    gen_stats = []

    start = time.time()
    for gen in range(1, n_generations+1):
        # Add stats of existing population
        gen_stats.append(get_gen_stats(toolbox, population))

        # Perform crossover (with probability cxpb) and mutation (with probability mutpb)
        offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob)
        # Calculate fitness on the offspring
        fitness_vals = list(toolbox.map(toolbox.evaluate, offspring))
        for fit, ind in zip(fitness_vals, offspring):
            # Set each individuals fitness manually
            ind.fitness.values = fit

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
    args = parser.parse_args()

    print(args.genes, args.crossover_probs, args.mutation_probs)

    for gene in args.genes:
        for crossover_prob in args.crossover_probs:
            for mutation_prob in args.mutation_probs:
                logger = Logger('../logs', DESCRIPTION)
                logger.set_header({
                    'seed': args.seed,
                    'epsilon': args.epsilon,
                    'gene': gene,
                    'crossover-prob': crossover_prob,
                    'mutation-prob': mutation_prob,
                    'pop_size': args.pop_size,
                    'max_gen': args.generations,
                    'tourn_size': args.tournsize,
                    'n_targets': args.n_targets,
                    'n_runs': args.n_runs
                })

                # Set seed for reproducibility
                random.seed(args.seed)

                # Define fitness function objective (minimisation)
                creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
                creator.create('Individual', list, fitness=creator.FitnessMin)

                # Create target signal generator and the toolbox
                target_generator = TargetGenerator()
                toolbox = ga.get_toolbox(args.tournsize)

                # Enable parallel code
                pool = None
                if PARALLEL:
                    pool = multiprocessing.Pool()
                    toolbox.register('map', pool.map)

                # For logging and plotting

                for i in tqdm.tqdm(range(args.n_targets), desc="#signals", ncols = 60):
                    # Generate target signal and its features
                    target_params, target_sound = next(target_generator)
                    target_features = ga.extract_features(target_sound)

                    logger.set_target(target_params)

                    # Register evaluation function (different for every target)
                    toolbox.register('evaluate', ga.fitness, target_features=target_features)
                    for n in tqdm.tqdm(range(args.n_runs), desc="   #runs", leave=False, ncols=60):
                        best_individual, gens, runtime, gen_stats = run_evolutionary_algorithm(toolbox)

                        logger.add_run(best=ga.individual_to_params(best_individual),
                                    best_fit=ga.fitness(best_individual, target_features)[0],
                                    n_gens=gens,
                                    early_stopping=(gens < args.generations),
                                    runtime=runtime,
                                    gen_stats=gen_stats)

                    logger.calculate_metrics(False)

                logger.close()

                if PARALLEL:
                    pool.close()

