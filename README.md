# CCS516 Assignment 01 — Solving Traveling Salesman Problem (TSP) Using DEAP

**Course:** CCS516 Computational Intelligence (Semester 1, 2026/2027)  
**Lecturer:** Dr. Wong Li Pei  
**Institution:** School of Computer Sciences, Universiti Sains Malaysia (USM)  
**Group 3:** Akbar Fajar Ramadhan (`25203025`) & Teoh Poey Guan (`25102708`)  
**Repository:** https://github.com/acruz2332/deap-ci

---

## Overview

This repository contains the complete implementation and reproducibility pipeline for **CCS516 Assignment 01: Solving Traveling Salesman Problem (TSP) Using Distributed Evolutionary Algorithms in Python (DEAP)** across five TSPLIB benchmark instances:
- **`berlin52`** (52 cities — TSPLIB Opt: `7542`)
- **`eil101`** (101 cities — TSPLIB Opt: `629`)
- **`tsp225`** (225 cities — TSPLIB Opt: `3916`)
- **`pcb442`** (442 cities — TSPLIB Opt: `50778`)
- **`pr1002`** (1002 cities — TSPLIB Opt: `259045`)

This repository (`main` branch) combines Tasks 1, 2, and 3 in one unified pipeline, implementing our proposed hybrid Evolutionary Algorithm **`eaMuPlusLambdaUnique`** (Diversity-Preserving $(\mu + \lambda)$-EA with Nearest Neighbor heuristic initialization, Tournament parent selection, Order Crossover, Shuffle Indexes Mutation, and duplicate-eliminating $(\mu + \lambda)$ elitist survivor selection) under a strict **50,000 fitness evaluation limit** per run.

---

## Prerequisites & Setup

1. **Clone the repository** (or extract the submitted zip archive):
   ```bash
   git clone https://github.com/acruz2332/deap-ci.git
   cd deap-ci
   ```

2. **Install Python 3 dependencies** (virtual environment recommended):
   ```bash
   pip install deap numpy matplotlib seaborn
   ```

---

## Step-by-Step Reproduction Guide

### 1. Reproducing Task 1 & Task 2 (Table 1 & Figure 3)
Runs `tsp.py` directly (using `float64` coordinates and TSPLIB `EUC_2D` per-edge nearest-integer rounding `int(distance + 0.5)`) to print exact optimal tour lengths (`7542`, `629`, `3916`, `50778`, `259045`) and plot all 5 optimal tours simultaneously:
```bash
python tsp.py
```

---

### 2. Reproducing Section 3.1.2 — 8-Algorithm Empirical Comparison (Table 2 & Figure 6)
All 8 evolutionary algorithms are implemented in `elitism.py` and selectable directly inside `solve_tsp.py`.
To reproduce Table 2 under the shared baseline configuration (`POPULATION_SIZE = 300`, `MAX_EVALS = 50000`, `HEURISTIC_PROB = 1.0`, `HEURISTIC_K = 1`, `RANDOM_SEED = 42`, `mutShuffleIndexes`, `cxOrdered`, `selTournament(tournsize=2)`, and for `#8`: `EA_MPL_MU = 300`, `EA_MPL_LAMBDA = 300`, `EA_MPL_P_CROSSOVER = 0.9`, `EA_MPL_P_MUTATION = 0.1`):
1. Open `solve_tsp.py` and set `EA_MPL_TOURN_SIZE = 2` (and for `#8`, set `EA_MPL_LAMBDA = 300` and `EA_MPL_P_MUTATION = 0.1`).
2. **Uncomment only the algorithm block (`#1` through `#8`)** you want to benchmark, then run:
```bash
python solve_tsp.py
```

| Block | Function in `elitism.py` | Description |
| :---: | :--- | :--- |
| `#1` | `eaSimpleWithElitism` | Generational GA with HallOfFame Elitism (`HOF = 20`) |
| `#2` | `epWithElitism` | Deterministic EP ($EV(m, n)$ with $m = n$, strict top-$m$ survival) |
| `#3` | `epStandard` | Standard Textbook EP (Stochastic $q$-Tournament, $q = 10$) |
| `#4` | `esMuPlusLambda` | Schwefel $(\mu + \lambda)$-ES ($\mu = 300$, $\lambda = 600$) |
| `#5` | `evSteadyState` | Steady-State $EV(m, 1)$ (immediate 1-on-1 random-opponent replacement) |
| `#6` | `evBatch` | Batch $EV(m, n)$ (batch of $n = 300$ children competing 1-on-1) |
| `#7` | `evSteadyStateReplaceWorst` | Steady-State GENITOR variant (No-Duplicate Replace-Worst) |
| `#8` | `eaMuPlusLambdaUnique` | Proposed Diversity-Preserving $(\mu + \lambda)$-EA |

---

### 3. Reproducing Section 3.2 — Final Proposed Algorithm & 5-Run Results (Tables 5–6 & Figure 7)

Run `solve_tsp.py` directly out of the box (default `RANDOM_SEED = 42` reproduces Run 1 and displays the Figure 7 tour and convergence plots simultaneously):
```bash
python solve_tsp.py
```

#### Final Tuned Configuration (`solve_tsp.py`)
| Component | Setting |
| :--- | :--- |
| **Algorithm** | `eaMuPlusLambdaUnique` (`elitism.py`) |
| **Initialization** | Heuristic Nearest Neighbor (`HEURISTIC_PROB = 1.0`, `HEURISTIC_K = 1`) |
| **Population Management** | $\mu = 300$ unique parents, $\lambda = 150$ offspring (`POPULATION_SIZE = 300`) |
| **Parent Selection** | Tournament Selection (`tools.selTournament`, `EA_MPL_TOURN_SIZE = 5`) |
| **Crossover** | Order Crossover (`tools.cxOrdered`, `EA_MPL_P_CROSSOVER = 0.9`) |
| **Mutation** | Shuffle Indexes Mutation (`tools.mutShuffleIndexes`, `EA_MPL_P_MUTATION = 0.3`) |
| **Survivor Selection** | Diversity-Preserving $(\mu + \lambda)$ Elitism (`_selectUniqueBest`, 4-decimal fitness deduplication) |
| **Termination Condition** | `MAX_EVALS = 50000` (`MAX_GENERATIONS = None`) |

#### Reproducing All 5 Independent Runs (Table 6)
In `solve_tsp.py` (line 21), set `RANDOM_SEED` to each of the **5 seeds** used in the report and run `python solve_tsp.py` for each seed:
1. **Run 1:** `RANDOM_SEED = 42` $\rightarrow$ `python solve_tsp.py`
2. **Run 2:** `RANDOM_SEED = 100` $\rightarrow$ `python solve_tsp.py`
3. **Run 3:** `RANDOM_SEED = 2026` $\rightarrow$ `python solve_tsp.py`
4. **Run 4:** `RANDOM_SEED = 7` $\rightarrow$ `python solve_tsp.py`
5. **Run 5:** `RANDOM_SEED = 99` $\rightarrow$ `python solve_tsp.py`

#### Reproducing the Visualizations (Figure 7)
Tour plots and Min/Average fitness convergence curves for all 5 TSPs are enabled by default in `solve_tsp.py` (lines 226–244 and `plt.show()` on line 265). Running `python solve_tsp.py` with `RANDOM_SEED = 42` reproduces Figure 7 directly.

---

## Expected Final Results (Averaged Over 5 Runs — TSPLIB `EUC_2D`)

| Problem | TSPLIB Optimal | Run 1 (`42`) | Run 2 (`100`) | Run 3 (`2026`) | Run 4 (`7`) | Run 5 (`99`) | **Average Tour Length** | **Deviation (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`berlin52`** | 7,542 | 7,542 | 7,542 | 7,542 | 7,596 | 7,542 | **7,552.80** | **0.14%** |
| **`eil101`** | 629 | 691 | 692 | 688 | 707 | 684 | **692.40** | **10.08%** |
| **`tsp225`** | 3,916 | 4,519 | 4,543 | 4,520 | 4,512 | 4,468 | **4,512.40** | **15.23%** |
| **`pcb442`** | 50,778 | 57,124 | 57,255 | 57,544 | 56,882 | 58,168 | **57,394.60** | **13.03%** |
| **`pr1002`** | 259,045 | 310,665 | 309,991 | 309,237 | 309,084 | 308,345 | **309,464.40** | **19.46%** |
| **Overall Avg** | — | 11.54% | 11.69% | 11.50% | 11.93% | 11.29% | — | **11.59%** |

---

## Project Structure

```text
DEAP-source-codes/
├── tsp-data/        # TSPLIB coordinate (.tsp), optimal tour (.opt.tour), and pickle cache files
├── tsp.py           # TravelingSalesmanProblem class (Euclidean distance matrix & plotting)
├── elitism.py       # Custom initialization, operators (mutInversion, selRank), and all 8 EA/EP/ES/EV algorithms
├── solve_tsp.py     # Experiment pipeline, tuned parameters, metrics summary, and visualization
└── README.md        # Reproduction guide and project documentation
```
