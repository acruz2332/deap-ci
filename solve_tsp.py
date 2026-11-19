from deap import base
from deap import creator
from deap import tools

import random
import array
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tsp
from elitism import (
    createHybridOrder, mutInversion, selRank,
    eaSimpleWithElitism, epWithElitism, epStandard, esMuPlusLambda,
    evSteadyState, evBatch, evSteadyStateReplaceWorst, eaMuPlusLambdaUnique
)

# set the random seed for repeatable results
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Problem instances and their respective optimal tour lengths:
TSP_PROBLEMS = ["berlin52", "eil101"
                , "tsp225", "pcb442", "pr1002"
                ]
TSP_OPTIMALS = [7542, 629
                , 3916, 50778, 259045
                ]

# Shared Population & Initialization constants:
POPULATION_SIZE = 300
MAX_EVALS = 50000     # strict maximum fitness function evaluations per run (assignment limit)
MAX_GENERATIONS = None  # set to None to run purely until MAX_EVALS is reached
HEURISTIC_PROB = 1.0  # probability of using heuristic initialization per individual
HEURISTIC_K = 1       # top-k nearest neighbors for randomized heuristic tour
TOURN_SIZE = 2        # tournament size for parent selection (used by EA and GENITOR)

# EA (Genetic Algorithm) specific constants:
EA_HALL_OF_FAME_SIZE = 20  # number of elites preserved in HallOfFame in EA
EA_P_CROSSOVER = 0.9       # probability for crossover in EA
EA_P_MUTATION = 0.1        # probability for mutating an individual in EA

# EP (Evolutionary Programming) specific constants:
EP_P_MUTATION = 1.0   # probability for mutating each offspring in EP (no crossover in EP)
EP_Q_TOURNAMENT = 10  # number of random opponents in standard textbook EP round-robin tournament

# ES (Evolution Strategy: (mu + lambda)-ES) specific constants:
ES_MU = 300            # number of parents (mu) kept each generation (set to 1 for (1 + lambda)-ES)
ES_LAMBDA = 600       # number of offspring (lambda) generated each generation (lambda > mu)
ES_P_CROSSOVER = 0.9  # crossover probability in ES (varOr requires ES_P_CROSSOVER + ES_P_MUTATION <= 1.0)
ES_P_MUTATION = 0.1   # mutation probability in ES (1.0 = pure mutation ES)

# EV (Steady-State EV(m, 1) & Batch EV(m, n) from lecture slides 3-4) specific constants:
EV_P_MUTATION = 1.0   # probability of mutating the cloned parent in EV
EV_BATCH_SIZE = 300   # batch size n of offspring produced before competition in EV(m, n)

# EV Steady-State Replace-Worst Variant (GENITOR: Tournament + Crossover + Mutation + Replace-Worst):
EV_RW_P_CROSSOVER = 0.9  # crossover probability in steady-state replace-worst
EV_RW_P_MUTATION = 0.1   # mutation probability in steady-state replace-worst

# Proposed Diversity-Preserving (mu + lambda)-EA specific constants (tuned optimal values):
EA_MPL_MU = 300          # number of UNIQUE parents kept each generation
EA_MPL_LAMBDA = 300      # number of offspring generated each generation
EA_MPL_P_CROSSOVER = 0.5 # tuned optimal crossover probability
EA_MPL_P_MUTATION = 0.3  # tuned optimal mutation probability
EA_MPL_RANK_PRESSURE = 10.0  # tuned optimal non-linear rank selection pressure (weight = rank ** pressure)

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

    # Genetic operators (tuned optimal operators: Linear Rank + Order Crossover + Inversion Mutation):
    toolbox.register("select", selRank, pressure=EA_MPL_RANK_PRESSURE)
    # toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", mutInversion)
    # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0 / len(current_tsp))

    return toolbox


# Genetic Algorithm flow:
def main():
    sns.set_style("whitegrid")
    best_ever_fitness = {}
    errors = {}
    pct_errors = {}
    eval_counts = {}
    times = {}
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

        # define the hall-of-fame object (used only by EA):
        hof = tools.HallOfFame(EA_HALL_OF_FAME_SIZE)

        """
        This is where you can choose which evolutionary algorithm to run by uncommenting the corresponding block of code. You can choose between:
        1. EA (Genetic Algorithm with Elitism)
        2. EP Deterministic (EV(m, n) with m=n)
        3. EP Standard Textbook (Stochastic q-Tournament Survivor Selection)
        4. ES ((mu + lambda)-ES by Schwefel, 1975)
        5. EV Steady-State (EV(m, 1) - immediate 1-on-1 competition)
        6. EV Batch (EV(m, n) - batch of n children competing 1-on-1)
        7. EV Steady-State Replace-Worst Variant (GENITOR - Tournament + Crossover + Mutation + Replace-Worst)
        8. Proposed Diversity-Preserving (mu + lambda)-EA (Unique-Survivor Elitist GA)
        Each algorithm will evolve the population and return the best individual found, along with its fitness and error metrics compared to the known optimal solution. The results will be printed and can also be plotted for
        """

        #import time module and record the start time
        start_time = time.time()

        # perform the Evolutionary flow (uncomment the algorithm you want to run):
        # 1. EA (Genetic Algorithm with Elitism):
        # population, logbook = eaSimpleWithElitism(
        #     population, toolbox, cxpb=EA_P_CROSSOVER, mutpb=EA_P_MUTATION,
        #     ngen=MAX_GENERATIONS, max_evals=MAX_EVALS, stats=stats, halloffame=hof, verbose=False
        # )

        # 2. EP Deterministic (EV(m, n) with m=n from lecture slides):
        # population, logbook = epWithElitism(
        #     population, toolbox, mutpb=EP_P_MUTATION,
        #     ngen=MAX_GENERATIONS, max_evals=MAX_EVALS, stats=stats, verbose=False
        # )

        # 3. EP Standard Textbook (Stochastic q-Tournament Survivor Selection):
        # population, logbook = epStandard(
        #     population, toolbox, mutpb=EP_P_MUTATION, q_tournament=EP_Q_TOURNAMENT,
        #     ngen=MAX_GENERATIONS, max_evals=MAX_EVALS, stats=stats, verbose=False
        # )

        # 4. ES ((mu + lambda)-ES by Schwefel, 1975):
        # population, logbook = esMuPlusLambda(
        #     population, toolbox, mu=ES_MU, lambda_=ES_LAMBDA,
        #     cxpb=ES_P_CROSSOVER, mutpb=ES_P_MUTATION,
        #     ngen=MAX_GENERATIONS, max_evals=MAX_EVALS, stats=stats, verbose=False
        # )

        # 5. EV Steady-State (EV(m, 1) from lecture slide 3):
        # population, logbook = evSteadyState(
        #     population, toolbox, mutpb=EV_P_MUTATION,
        #     ngen=MAX_GENERATIONS, max_evals=MAX_EVALS, stats=stats, verbose=False
        # )

        # 6. EV Batch / Generational (EV(m, n) from lecture slide 4):
        # population, logbook = evBatch(
        #     population, toolbox, batch_size=EV_BATCH_SIZE, mutpb=EV_P_MUTATION,
        #     ngen=MAX_GENERATIONS, max_evals=MAX_EVALS, stats=stats, verbose=False
        # )

        # 7. EV Steady-State Replace-Worst Variant (GENITOR with No-Duplicates Policy):
        # population, logbook = evSteadyStateReplaceWorst(
        #     population, toolbox, cxpb=EV_RW_P_CROSSOVER, mutpb=EV_RW_P_MUTATION,
        #     ngen=MAX_GENERATIONS, max_evals=MAX_EVALS, stats=stats, verbose=False
        # )

        # 8. Proposed Diversity-Preserving (mu + lambda)-EA (Unique-Survivor Elitist GA):
        population, logbook = eaMuPlusLambdaUnique(
            population, toolbox, mu=EA_MPL_MU, lambda_=EA_MPL_LAMBDA,
            cxpb=EA_MPL_P_CROSSOVER, mutpb=EA_MPL_P_MUTATION,
            ngen=MAX_GENERATIONS, max_evals=MAX_EVALS, stats=stats, verbose=False
        )

        # extract best individual and compute error metrics (works for both EA and EP):
        best = tools.selBest(population, 1)[0]
        best_fitness = best.fitness.values[0]
        error = best_fitness - opt_val
        end_time = time.time()
        execution_time = end_time - start_time
        times[problem_name] = execution_time
        pct_error = (error / opt_val) * 100.0
        total_evals = sum(logbook.select("nevals"))

        errors[problem_name] = error
        pct_errors[problem_name] = pct_error
        best_ever_fitness[problem_name] = best_fitness
        eval_counts[problem_name] = total_evals

        # # print best individual info for this problem:
        # print(f"=== Problem: {problem_name} ===")
        # print("-- Best Ever Individual = ", best)
        print(f"-- Best Ever Fitness    =  {best_fitness:.4f} (rounded: {round(best_fitness)})")
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
        print(f"{problem_name:<10} | Opt: {opt_val:<8} | Best: {best_ever_fitness[problem_name]:>12.4f} | Error: {errors[problem_name]:>12.4f} | Deviation: {pct_errors[problem_name]:>7.2f}% | Execution Time: {times[problem_name]:>7.2f}s | Evals: {eval_counts[problem_name]}")
    print("-" * 60)
    print(f"Total Sum of Errors (Distance) : {total_error_sum:.4f}")
    print(f"Total Sum of Deviation (%)     : {total_pct_error_sum:.2f}%")
    print(f"Total time taken for all problems  (s) : {sum(times.values()):.2f}s")
    print(f"Average Deviation (%)          : {avg_pct_error:.2f}%")
    print(f"Average Execution Time (s)     : {sum(times.values()) / len(times):.2f}s")
    print("=" * 60)

    # # show all plots simultaneously:
    # plt.show()


if __name__ == "__main__":
    main()
