from deap import base
from deap import creator
from deap import tools

import random
import array

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tsp
import elitism
from elitism import createHybridOrder, createHeuristicTour

# set the random seed for repeatable results
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Problem instances and their respective optimal tour lengths:
TSP_PROBLEMS = ["berlin52", "eil101"
                # , "tsp225", "pcb442", "pr1002"
                ]
TSP_OPTIMALS = [7542, 629
                # , 3916, 50778, 259045
                ]

# Genetic Algorithm constants:
POPULATION_SIZE = 300
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 30
P_CROSSOVER = 0.9     # probability for crossover
P_MUTATION = 0.1      # probability for mutating an individual
HEURISTIC_PROB = 0  # probability of using heuristic initialization per individual
HEURISTIC_K = 3       # top-k nearest neighbors for randomized heuristic tour

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list of integers:
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)


def setup_toolbox(current_tsp):
    """Configure and return a DEAP toolbox for the given TSP instance."""
    toolbox = base.Toolbox()

    # create an operator that generates hybrid (heuristic + random) indices:
    toolbox.register("hybridOrder", createHybridOrder, current_tsp,
                     heuristic_prob=HEURISTIC_PROB, k=HEURISTIC_K)

    # create the individual creation operator to fill up an Individual instance with hybrid indices:
    toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.hybridOrder)

    # create the population creation operator to generate a list of individuals:
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    # fitness calculation - compute the total distance of the list of cities represented by indices:
    def tpsDistance(individual):
        return current_tsp.getTotalDistance(individual),  # return a tuple

    toolbox.register("evaluate", tpsDistance)

    # Genetic operators:
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0 / len(current_tsp))

    return toolbox


# Genetic Algorithm flow:
def main():
    sns.set_style("whitegrid")
    best_ever_fitness = {}
    errors = {}
    pct_errors = {}
    fig_num = 1

    for problem_name, opt_val in zip(TSP_PROBLEMS, TSP_OPTIMALS):
        # reset the random seed for repeatable results per problem:
        random.seed(RANDOM_SEED)

        # create the desired traveling salesman problem instance:
        current_tsp = tsp.TravelingSalesmanProblem(problem_name)
        toolbox = setup_toolbox(current_tsp)

        # create initial population (generation 0):
        population = toolbox.populationCreator(n=POPULATION_SIZE)

        # prepare the statistics object:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        # define the hall-of-fame object:
        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

        # perform the Genetic Algorithm flow with hof feature added:
        population, logbook = elitism.eaSimpleWithElitism(
            population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
            ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=False
        )

        # extract best individual and compute error metrics:
        best = hof.items[0]
        best_fitness = best.fitness.values[0]
        error = best_fitness - opt_val
        pct_error = (error / opt_val) * 100.0

        errors[problem_name] = error
        pct_errors[problem_name] = pct_error
        best_ever_fitness[problem_name] = best_fitness

        # # print best individual info for this problem:
        # print(f"=== Problem: {problem_name} ===")
        # print("-- Best Ever Individual = ", best)
        # print(f"-- Best Ever Fitness    =  {best_fitness:.4f} (rounded: {round(best_fitness)})")
        # print(f"-- Optimal Tour Length  =  {opt_val}")
        # print(f"-- Error (Best - Opt)   =  {error:.4f} ({pct_error:.2f}%)")
        # print("-" * 60)

        # # plot best solution:
        # plt.figure(fig_num, figsize=(10, 6))
        # current_tsp.plotData(best)
        # plt.title(f"{problem_name} - Best GA Tour: {best_fitness:.2f} | Opt: {opt_val} | Error: {error:.2f} ({pct_error:.2f}%)")
        # plt.tight_layout()
        # fig_num += 1

        # # plot statistics:
        # minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
        # plt.figure(fig_num, figsize=(10, 6))
        # plt.plot(minFitnessValues, color='red', label='Min Fitness')
        # plt.plot(meanFitnessValues, color='green', label='Avg Fitness')
        # plt.axhline(y=opt_val, color='blue', linestyle='--', label=f'Optimal ({opt_val})')
        # plt.xlabel('Generation')
        # plt.ylabel('Min / Average Fitness')
        # plt.title(f'{problem_name} - Min and Average Fitness over Generations')
        # plt.legend()
        # plt.tight_layout()
        # fig_num += 1

    # print summary of all errors and their total sum:
    total_error_sum = sum(errors.values())
    total_pct_error_sum = sum(pct_errors.values())
    avg_pct_error = total_pct_error_sum / len(TSP_PROBLEMS)

    print("\n" + "=" * 60)
    print("SUMMARY OF ERRORS ACROSS ALL TSP PROBLEMS")
    print("=" * 60)
    for problem_name, opt_val in zip(TSP_PROBLEMS, TSP_OPTIMALS):
        print(f"{problem_name:<10} | Opt: {opt_val:<8} | Best: {best_ever_fitness[problem_name]:>12.4f} | Error: {errors[problem_name]:>12.4f} | Deviation: {pct_errors[problem_name]:>7.2f}%")
    print("-" * 60)
    print(f"Total Sum of Errors (Distance) : {total_error_sum:.4f}")
    print(f"Total Sum of Deviation (%)     : {total_pct_error_sum:.2f}%")
    print(f"Average Deviation (%)          : {avg_pct_error:.2f}%")
    print("=" * 60)

    # # show all plots simultaneously:
    # plt.show()


if __name__ == "__main__":
    main()
