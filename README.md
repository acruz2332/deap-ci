# CCS516 Assignment 01 — Solving Traveling Salesman Problem (TSP) Using DEAP (`main`)

**Course:** CCS516 Computational Intelligence (Semester 1, 2026/2027)  
**Institution:** School of Computer Sciences, Universiti Sains Malaysia (USM)  
**Branch:** `main` (Provided Base Source Codes / Foundation)

---

## Overview

This repository contains the codebase for **CCS516 Assignment 01: Solving Traveling Salesman Problem (TSP) Using Distributed Evolutionary Algorithms in Python (DEAP)**.

The `main` branch holds the **original base source files** provided for the assignment (`tsp.py`, `solve_tsp.py`, and `elitism.py`) along with the TSPLIB benchmark data files in `tsp-data/`. These three source files serve as the foundation for designing and evaluating our Evolutionary Algorithm (EA) across the five target TSP instances:
- **`berlin52`** (52 cities)
- **`eil101`** (101 cities)
- **`tsp225`** (225 cities)
- **`pcb442`** (442 cities)
- **`pr1002`** (1002 cities)

---

## Project Structure

```text
DEAP-source-codes/
├── tsp-data/                  # TSPLIB problem coordinate (.tsp) and optimal tour (.opt.tour) files
│   ├── berlin52.tsp
│   ├── berlin52.opt.tour
│   ├── eil101.tsp
│   ├── eil101.opt.tour
│   ├── tsp225.tsp
│   ├── tsp225.opt.tour
│   ├── pcb442.tsp
│   ├── pcb442.opt.tour
│   ├── pr1002.tsp
│   └── pr1002.opt.tour
├── tsp.py                     # Base TravelingSalesmanProblem class (data loading, distance calculation, plotting)
├── solve_tsp.py               # Base DEAP Genetic Algorithm script for TSP
├── elitism.py                 # Elitism evolutionary loop (eaSimpleWithElitism)
└── README.md                  # Documentation for the base codelines (main branch)
```

---

## Base Source Files Description

### 1. [`tsp.py`](./tsp.py) — TSP Problem Encapsulation
Encapsulates a TSPLIB Traveling Salesman Problem instance via the `TravelingSalesmanProblem` class:
- **Data Loading & Caching (`__initData`, `__createData`):** Reads 2D city coordinates from `tsp-data/{name}.tsp`, computes the $N \times N$ pairwise Euclidean distance matrix, and caches both to disk as `.pickle` files (`{name}-loc.pickle` and `{name}-dist.pickle`).
- **Tour Evaluation (`getTotalDistance`):** Computes the closed-loop total Euclidean distance of a tour represented by a sequence of 0-based city indices.
- **Visualization (`plotData`):** Plots city coordinates as red dots and connects the tour path with blue lines using `matplotlib`.
- *(Note: In the base code, `main()` is configured with a hardcoded demo instance `bayg29`.)*

### 2. [`solve_tsp.py`](./solve_tsp.py) — Base DEAP Genetic Algorithm
Implements a baseline Genetic Algorithm using the **DEAP** (`deap.base`, `deap.creator`, `deap.tools`) framework:
- **Representation:** Permutation of 0-based city indices stored in `array.array('i')`.
- **Objective (`FitnessMin`):** Single-objective minimization (`weights=(-1.0,)`) of total tour distance (`tsp.getTotalDistance`).
- **Initialization:** Uniform random permutation via `random.sample(range(len(tsp)), len(tsp))`.
- **Selection:** Tournament Selection (`tools.selTournament`, `tournsize=2`).
- **Crossover:** Ordered Crossover (`tools.cxOrdered`, `P_CROSSOVER = 0.9`).
- **Mutation:** Shuffle Indexes Mutation (`tools.mutShuffleIndexes`, `P_MUTATION = 0.1`, `indpb = 1.0 / len(tsp)`).
- **Population & Generations (Base Constants):** `POPULATION_SIZE = 300`, `MAX_GENERATIONS = 200`, `HALL_OF_FAME_SIZE = 30`, `RANDOM_SEED = 42`.

### 3. [`elitism.py`](./elitism.py) — Generational Loop with Elitism
Provides `eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats, halloffame, verbose)`:
- Extends DEAP's standard `eaSimple` generational loop by preserving the top `hof_size` individuals stored in `halloffame`.
- Selects `len(population) - hof_size` parents for crossover and mutation (`algorithms.varAnd`), then directly injects the `halloffame.items` elites into the next generation unchanged.

---

## Assignment Tasks Overview

1. **Task 1:** Use `tsp.py` to generate plots for the optimal tours specified in `berlin52.opt.tour`, `eil101.opt.tour`, `tsp225.opt.tour`, `pcb442.opt.tour`, and `pr1002.opt.tour`.
2. **Task 2:** Modify `tsp.py` to compute the optimal tour length (rounded to the nearest integer) for `berlin52`, `eil101`, `tsp225`, `pcb442`, and `pr1002`.
3. **Task 3:** Modify `solve_tsp.py` and `elitism.py` to design an Evolutionary Algorithm (EA) to solve all five TSPs (`berlin52`, `eil101`, `tsp225`, `pcb442`, and `pr1002`), covering initialization, parent/survivor selection, variation operators & probabilities, population management, random seeds across 5 runs, and a termination condition of at most **50,000 fitness evaluations per run**.

---

## Getting Started

### Prerequisites
Install Python 3 and the required libraries:
```bash
pip install deap numpy matplotlib seaborn
```

### Running the Base Scripts
```bash
# Run the TSP class test script
python tsp.py

# Run the base DEAP Genetic Algorithm script
python solve_tsp.py
```
