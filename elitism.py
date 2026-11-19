import random
from deap import tools
from deap import algorithms


def createHeuristicTour(current_tsp, k=3):
    """Generates a TSP tour using Randomized Nearest Neighbor (top-k closest unvisited cities)."""
    num_cities = len(current_tsp)
    unvisited = set(range(num_cities))

    # Pick a random starting city:
    current_city = random.choice(list(unvisited))
    tour = [current_city]
    unvisited.remove(current_city)

    while unvisited:
        # Find the top-k closest unvisited cities to current_city:
        nearest_candidates = sorted(
            unvisited,
            key=lambda city: current_tsp.distances[current_city][city]
        )
        next_city = random.choice(nearest_candidates[:k])

        tour.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city

    return tour


def createHybridOrder(current_tsp, heuristic_prob=0.2, k=3):
    """Hybrid initialization: generates a heuristic tour with probability `heuristic_prob`,
    otherwise generates a purely random permutation.
    """
    if random.random() < heuristic_prob:
        return createHeuristicTour(current_tsp, k=k)
    else:
        return random.sample(range(len(current_tsp)), len(current_tsp))


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen=None, stats=None,
             halloffame=None, verbose=__debug__, max_evals=50000):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    Stops when `max_evals` fitness evaluations or `ngen` generations is reached.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    total_evals = len(invalid_ind)

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    gen = 0
    while True:
        if ngen is not None and gen >= ngen:
            break
        if max_evals is not None and total_evals >= max_evals:
            break
        gen += 1

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if max_evals is not None and total_evals + len(invalid_ind) > max_evals:
            break

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        total_evals += len(invalid_ind)

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def epWithElitism(population, toolbox, mutpb=1.0, ngen=None, stats=None,
                        verbose=__debug__, max_evals=50000):
    ## this implementation is based on ppt slides from the course
    """Evolutionary Programming (EP) algorithm with strict (mu + mu) Elitist Survivor Selection.
    Unlike GA/EA, EP does not use crossover or a separate halloffame archive. Instead:
    1. Every parent in the population (mu) produces a mutated offspring (mu).
    2. Offspring are evaluated.
    3. Parents and offspring are combined into a pool of 2*mu (e.g., 600 individuals),
       and strictly the top mu (e.g., 300) best individuals are selected to survive.
    Stops when `max_evals` fitness evaluations or `ngen` generations is reached.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness (Generation 0)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    total_evals = len(invalid_ind)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the Evolutionary Programming generational process
    gen = 0
    while True:
        if ngen is not None and gen >= ngen:
            break
        if max_evals is not None and total_evals >= max_evals:
            break
        gen += 1

        # 1. Reproduction & Mutation only (No parent selection, no crossover):
        # Each parent is cloned and mutated with probability `mutpb` (typically 1.0 in EP)
        offspring = [toolbox.clone(ind) for ind in population]
        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # 2. Evaluate the mutated offspring with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if max_evals is not None and total_evals + len(invalid_ind) > max_evals:
            break

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        total_evals += len(invalid_ind)

        # 3. Survivor Selection: Strict (mu + mu) Elitism
        # Combine mu parents + mu offspring (600) and pick strictly the top mu (300) best:
        combined_pool = population + offspring
        population[:] = tools.selBest(combined_pool, len(population))

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def epStandard(population, toolbox, mutpb=1.0, q_tournament=10, ngen=None, stats=None,
               verbose=__debug__, max_evals=50000):
    """Standard Textbook Evolutionary Programming (EP) with Stochastic q-Tournament Survivor Selection.
    Based on classical EP (Fogel / Eiben & Smith):
    1. Every parent in the population (mu) produces 1 mutated offspring (mu) via asexual reproduction.
    2. Offspring are evaluated.
    3. Parents and offspring are combined into a pool of 2*mu (e.g., 600 individuals).
    4. Each individual competes in a round-robin tournament against `q_tournament` randomly chosen
       opponents from the combined pool and scores 1 win for each opponent it beats.
    5. The top mu (e.g., 300) individuals with the most tournament wins survive to the next generation.
    Stops when `max_evals` fitness evaluations or `ngen` generations is reached.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness (Generation 0)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    total_evals = len(invalid_ind)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the Standard EP generational process
    gen = 0
    while True:
        if ngen is not None and gen >= ngen:
            break
        if max_evals is not None and total_evals >= max_evals:
            break
        gen += 1

        # 1. Reproduction & Mutation only (No parent selection, no crossover):
        offspring = [toolbox.clone(ind) for ind in population]
        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # 2. Evaluate the mutated offspring with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if max_evals is not None and total_evals + len(invalid_ind) > max_evals:
            break

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        total_evals += len(invalid_ind)

        # 3. Survivor Selection: Stochastic q-Tournament (Round-Robin) on combined pool (2*mu)
        combined_pool = population + offspring
        pool_size = len(combined_pool)
        q = min(q_tournament, pool_size - 1)

        scores = []
        for idx, ind in enumerate(combined_pool):
            # Sample q random opponents (excluding self)
            wins = 0
            for _ in range(q):
                opp_idx = random.randrange(pool_size - 1)
                if opp_idx >= idx:
                    opp_idx += 1
                # DEAP Fitness comparison automatically respects weights=(-1.0,) (shorter tour is greater)
                if ind.fitness > combined_pool[opp_idx].fitness:
                    wins += 1
            scores.append((wins, ind))

        # Sort by tournament wins descending (ties broken by random order) and keep top mu
        random.shuffle(scores)
        scores.sort(key=lambda item: item[0], reverse=True)
        population[:] = [ind for _, ind in scores[:len(population)]]

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def esMuPlusLambda(population, toolbox, mu, lambda_, cxpb=0.0, mutpb=1.0, ngen=None,
                   stats=None, verbose=__debug__, max_evals=50000):
    """Evolution Strategy (mu + lambda)-ES (Schwefel, 1975) with strict Elitist Survivor Selection.
    1. Maintains a parent population of size `mu` (if initial population > mu, keeps top mu after Gen 0).
    2. Generates `lambda_` offspring (where lambda_ > mu, e.g., mu=50, lambda_=300) using DEAP's varOr
       (recombination with probability `cxpb`, mutation with probability `mutpb`; cxpb + mutpb <= 1.0).
    3. Combines `mu` parents + `lambda_` offspring and deterministically selects the top `mu` best
       individuals (`tools.selBest`) to survive to the next generation.
    Stops when `max_evals` fitness evaluations or `ngen` generations is reached.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness (Generation 0)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    total_evals = len(invalid_ind)

    # Trim initial population to top `mu` parents if initial population size > mu
    if len(population) > mu:
        population[:] = tools.selBest(population, mu)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the (mu + lambda)-ES generational process
    gen = 0
    while True:
        if ngen is not None and gen >= ngen:
            break
        if max_evals is not None and total_evals >= max_evals:
            break
        gen += 1

        # 1. Generate `lambda_` offspring from `mu` parents (varOr requires cxpb + mutpb <= 1.0)
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # 2. Evaluate the offspring with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if max_evals is not None and total_evals + len(invalid_ind) > max_evals:
            break

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        total_evals += len(invalid_ind)

        # 3. Survivor Selection: Strict (mu + lambda) Elitism
        # Combine `mu` parents + `lambda_` offspring and keep strictly the top `mu` best:
        population[:] = tools.selBest(population + offspring, mu)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def evSteadyState(population, toolbox, mutpb=1.0, ngen=None, stats=None,
                  verbose=__debug__, max_evals=50000):
    """Steady-State Evolutionary Algorithm EV(m, 1) (from lecture slide 3):
    1. Uniformly at random select 1 parent from the current population of size m.
    2. Clone the parent and probabilistically mutate it (with probability `mutpb`) to produce 1 child.
    3. Evaluate the child's fitness.
    4. Uniformly at random select 1 candidate from the current population for deletion, and keep
       whichever has higher fitness between the candidate and the child.
    Stops when `max_evals` fitness evaluations or `ngen` generation-equivalents is reached.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate initial population of size m (Generation 0)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    total_evals = len(invalid_ind)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    pop_size = len(population)
    gen = 0
    evals_since_log = 0

    # Steady-state loop (1 child at a time)
    while True:
        if max_evals is not None and total_evals >= max_evals:
            break
        if ngen is not None and gen >= ngen:
            break

        # 1. Uniformly select 1 parent and clone it
        parent = random.choice(population)
        child = toolbox.clone(parent)

        # 2. Probabilistically mutate to produce 1 offspring
        if random.random() < mutpb:
            child, = toolbox.mutate(child)
            del child.fitness.values

        # 3. Compute the fitness of the offspring (if mutated)
        if not child.fitness.valid:
            if max_evals is not None and total_evals + 1 > max_evals:
                break
            child.fitness.values = toolbox.evaluate(child)
            total_evals += 1
            evals_since_log += 1

        # 4. Immediate 1-on-1 competition against a uniformly random candidate in the population
        cand_idx = random.randrange(pop_size)
        if child.fitness > population[cand_idx].fitness:
            population[cand_idx] = child

        # Log stats once every `pop_size` evaluations (1 generation-equivalent) for speed & plotting
        if evals_since_log >= pop_size:
            gen += 1
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=evals_since_log, **record)
            if verbose:
                print(logbook.stream)
            evals_since_log = 0

    # Log any leftover evaluations from the final partial generation
    if evals_since_log > 0:
        gen += 1
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=evals_since_log, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def evBatch(population, toolbox, batch_size=100, mutpb=1.0, ngen=None, stats=None,
            verbose=__debug__, max_evals=50000):
    """Batch / Generational Evolutionary Algorithm EV(m, n) (from lecture slide 4):
    1. Produce a batch of `n` (`batch_size`) offspring by uniformly selecting a parent from
       the parent population `n` times, cloning, and probabilistically mutating (`mutpb`).
    2. Evaluate the batch of offspring.
    3. For each of the `n` offspring, force it to compete 1-on-1 against a uniformly random
       member of the current population, keeping whichever has higher fitness.
    Stops when `max_evals` fitness evaluations or `ngen` generations is reached.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate initial population of size m (Generation 0)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    total_evals = len(invalid_ind)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    pop_size = len(population)
    gen = 0

    # Batch generational loop (n children per batch)
    while True:
        if ngen is not None and gen >= ngen:
            break
        if max_evals is not None and total_evals >= max_evals:
            break
        gen += 1

        # 1. For I = 1 to n: uniformly select a parent, clone, and mutate to build offspring batch
        offspring = []
        for _ in range(batch_size):
            child = toolbox.clone(random.choice(population))
            if random.random() < mutpb:
                child, = toolbox.mutate(child)
                del child.fitness.values
            offspring.append(child)

        # 2. Evaluate the batch of offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if max_evals is not None and total_evals + len(invalid_ind) > max_evals:
            break

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        total_evals += len(invalid_ind)

        # 3. For I = 1 to n: force each offspring to compete 1-on-1 against a random parent member
        for child in offspring:
            cand_idx = random.randrange(pop_size)
            if child.fitness > population[cand_idx].fitness:
                population[cand_idx] = child

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def evSteadyStateReplaceWorst(population, toolbox, cxpb=0.9, mutpb=0.2, ngen=None, stats=None,
                              verbose=__debug__, max_evals=50000):
    """Steady-State Variant (GENITOR / Replace-Worst Survivor Selection).
    Extends the standard Steady-State EV(m, 1) model with:
    1. Tournament parent selection (`toolbox.select`) to pick 2 parents.
    2. Crossover (`cxpb`) + Mutation (`mutpb`) to generate 2 offspring.
    3. Immediate Replace-Worst survivor selection: each offspring replaces the WORST member
       of the population if and only if the offspring is fitter (naturally 100% elitist).
    Stops when `max_evals` fitness evaluations or `ngen` generation-equivalents is reached.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate initial population of size m (Generation 0)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    total_evals = len(invalid_ind)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    pop_size = len(population)
    gen = 0
    evals_since_log = 0

    # Track fitness occurrences to prevent duplicate clones from taking over (GENITOR no-duplicates policy)
    fit_counts = {}
    for ind in population:
        fk = round(ind.fitness.values[0], 4)
        fit_counts[fk] = fit_counts.get(fk, 0) + 1

    # Steady-state GENITOR loop
    while True:
        if max_evals is not None and total_evals >= max_evals:
            break
        if ngen is not None and gen >= ngen:
            break

        # 1. Select 2 parents via tournament selection and clone them
        p1, p2 = toolbox.select(population, 2)
        child1, child2 = toolbox.clone(p1), toolbox.clone(p2)

        # 2. Apply crossover with probability `cxpb`
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        # 3. Apply mutation to each child with probability `mutpb`
        for child in (child1, child2):
            if random.random() < mutpb:
                toolbox.mutate(child)
                del child.fitness.values

        # 4. Evaluate each modified child and apply No-Duplicate Replace-Worst survivor selection
        for child in (child1, child2):
            if not child.fitness.valid:
                if max_evals is not None and total_evals + 1 > max_evals:
                    break
                child.fitness.values = toolbox.evaluate(child)
                total_evals += 1
                evals_since_log += 1

                child_fk = round(child.fitness.values[0], 4)
                # Reject duplicate tours so clones never take over the population
                if fit_counts.get(child_fk, 0) > 0:
                    continue

                # If duplicates exist in population (e.g., from Gen 0), replace the worst duplicate first;
                # otherwise replace the overall worst individual only if child is strictly better.
                dup_indices = [i for i in range(pop_size) if fit_counts[round(population[i].fitness.values[0], 4)] > 1]
                if dup_indices:
                    target_idx = min(dup_indices, key=lambda i: population[i].fitness)
                    should_replace = True
                else:
                    target_idx = min(range(pop_size), key=lambda i: population[i].fitness)
                    should_replace = child.fitness > population[target_idx].fitness

                if should_replace:
                    old_fk = round(population[target_idx].fitness.values[0], 4)
                    fit_counts[old_fk] -= 1
                    if fit_counts[old_fk] == 0:
                        del fit_counts[old_fk]
                    population[target_idx] = child
                    fit_counts[child_fk] = 1

        # Log stats once every `pop_size` evaluations (1 generation-equivalent)
        if evals_since_log >= pop_size:
            gen += 1
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=evals_since_log, **record)
            if verbose:
                print(logbook.stream)
            evals_since_log = 0

    # Log any leftover evaluations from the final partial generation
    if evals_since_log > 0:
        gen += 1
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=evals_since_log, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def _selectUniqueBest(individuals, k):
    """Selects up to `k` best individuals prioritizing distinct tours (by rounded fitness).
    If fewer than `k` distinct tours exist, fills remaining slots with the next-best individuals.
    """
    sorted_inds = tools.selBest(individuals, len(individuals))
    selected = []
    duplicates = []
    seen = set()

    for ind in sorted_inds:
        fk = round(ind.fitness.values[0], 4)
        if fk not in seen:
            seen.add(fk)
            selected.append(ind)
            if len(selected) == k:
                return selected
        else:
            duplicates.append(ind)

    # Fill any remaining slots if fewer than k unique individuals exist
    needed = k - len(selected)
    selected.extend(duplicates[:needed])
    return selected


def eaMuPlusLambdaUnique(population, toolbox, mu=100, lambda_=100, cxpb=0.9, mutpb=0.2,
                         ngen=None, stats=None, verbose=__debug__, max_evals=50000):
    """Proposed Diversity-Preserving (mu + lambda) Evolutionary Algorithm.
    Combines the strengths of GA, (mu + lambda)-ES, and Duplicate Elimination:
    1. Evaluates the initial population and retains the top `mu` DISTINCT tours.
    2. Each generation selects `lambda_` parents via Tournament Selection (`toolbox.select`)
       and applies BOTH Crossover (`cxpb`) and Mutation (`mutpb`) via `algorithms.varAnd`.
    3. Survivor Selection merges parents + offspring (`mu + lambda_`) and keeps strictly the
       top `mu` UNIQUE individuals (`_selectUniqueBest`), preventing duplicate clone takeover
       while guaranteeing full elitism across all `mu` slots and more generations within 50k evals.
    Stops when `max_evals` fitness evaluations or `ngen` generations is reached.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate initial population (Generation 0)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    total_evals = len(invalid_ind)

    # Retain top `mu` unique parents from Generation 0
    population[:] = _selectUniqueBest(population, mu)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    gen = 0
    while True:
        if ngen is not None and gen >= ngen:
            break
        if max_evals is not None and total_evals >= max_evals:
            break
        gen += 1

        # 1. Parent Selection (Tournament) + Variation (Crossover AND Mutation via varAnd)
        offspring = toolbox.select(population, lambda_)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # 2. Evaluate modified offspring within the strict evaluation budget
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if max_evals is not None and total_evals + len(invalid_ind) > max_evals:
            break

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        total_evals += len(invalid_ind)

        # 3. Diversity-Preserving (mu + lambda) Survivor Selection: keep top `mu` UNIQUE tours
        population[:] = _selectUniqueBest(population + offspring, mu)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook







