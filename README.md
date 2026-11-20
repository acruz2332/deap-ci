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

On the **`task3`** branch, we implement our proposed hybrid Evolutionary Algorithm **`eaMuPlusLambdaUnique`** (Diversity-Preserving $(\mu + \lambda)$-EA with Nearest Neighbor heuristic initialization, non-linear rank parent selection, Order Crossover, Inversion Mutation, and duplicate-eliminating $(\mu + \lambda)$ elitist survivor selection) under a strict **50,000 fitness evaluation limit** per run.

---

## Branches & Tags Map (Reproducibility Index)

Every experiment and table in the assignment report can be reproduced directly using the Git branches and tags below:

| Git Ref (Branch / Tag) | Purpose | Report Reference |
| :--- | :--- | :--- |
| `main` *(branch)* | Original provided starter files (`tsp.py`, `solve_tsp.py`, `elitism.py`) | Foundation |
| `Task1andTask2` *(tag)* | Optimal tour parser, exact & rounded tour lengths, and simultaneous plotting for all 5 TSPs | **Task 1 & 2** (Table 1, Figures 1–3) |
| `AlgorithmEmpiricalTestingCodes` *(tag)* | Empirical comparison of all 8 algorithms (`EA`, `EP`, `EP Standard`, `ES`, `EV Steady-State`, `EV Batch`, `GENITOR`, `eaMuPlusLambdaUnique`) | **Section 3.1.2** (Table 2, Figure 6) |
| `Task3SoftRelease` *(tag)* / `task3` *(branch)* | Final tuned **`eaMuPlusLambdaUnique`** algorithm and parameters | **Section 3.2** (Tables 5–6, Figure 7) |

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
Checks out the Task 1 & 2 tag and runs `tsp.py` to print exact/rounded optimal tour lengths and plot all 5 optimal tours simultaneously:
```bash
git checkout Task1andTask2
python tsp.py
```

---

### 2. Reproducing Section 3.1.2 — 8-Algorithm Empirical Comparison (Table 2 & Figure 6)
Checks out the empirical testing tag containing all 8 evolutionary algorithms under the shared baseline configuration (`POPULATION_SIZE = 300`, `MAX_EVALS = 50000`, `HEURISTIC_PROB = 1.0`, `HEURISTIC_K = 1`, `RANDOM_SEED = 42`, `mutShuffleIndexes`, `cxOrdered`):
```bash
git checkout AlgorithmEmpiricalTestingCodes
```
Open `solve_tsp.py` and **uncomment only the algorithm block (`#1` through `#8`)** you want to benchmark, then run:
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

Switch to the `task3` branch (or `Task3SoftRelease` tag):
```bash
git checkout task3
# or: git checkout Task3SoftRelease
```

#### Final Tuned Configuration (`solve_tsp.py`)
| Component | Setting |
| :--- | :--- |
| **Algorithm** | `eaMuPlusLambdaUnique` (`elitism.py`) |
| **Initialization** | Heuristic Nearest Neighbor (`HEURISTIC_PROB = 1.0`, `HEURISTIC_K = 1`) |
| **Population Management** | $\mu = 300$ unique parents, $\lambda = 300$ offspring (`POPULATION_SIZE = 300`) |
| **Parent Selection** | Non-Linear Rank Selection (`selRank`, `EA_MPL_RANK_PRESSURE = 10.0`) |
| **Crossover** | Order Crossover (`tools.cxOrdered`, `EA_MPL_P_CROSSOVER = 0.5`) |
| **Mutation** | Inversion Mutation (`mutInversion`, `EA_MPL_P_MUTATION = 0.3`) |
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
To display the best tour plots and Min/Average fitness convergence curves for all 5 TSPs simultaneously:
1. In `solve_tsp.py`, uncomment the plotting blocks on **lines 226–244** and `plt.show()` on **line 265**.
2. Run `python solve_tsp.py` (Figure 7 in the report was generated with `RANDOM_SEED = 42`).

---

## Expected Final Results (Averaged Over 5 Runs)

| Problem | TSPLIB Optimal | Run 1 (`42`) | Run 2 (`100`) | Run 3 (`2026`) | Run 4 (`7`) | Run 5 (`99`) | **Average Tour Length** | **Deviation (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`berlin52`** | 7,542 | 7,544.3652 | 7,544.3652 | 7,544.3652 | 7,544.3652 | 7,544.3652 | **7,544.3652** | **0.03%** |
| **`eil101`** | 629 | 657.4204 | 671.4535 | 663.9196 | 676.9794 | 692.3359 | **672.4218** | **6.90%** |
| **`tsp225`** | 3,916 | 4,196.6807 | 4,158.8599 | 4,215.6406 | 4,177.1655 | 4,285.9375 | **4,206.8568** | **7.43%** |
| **`pcb442`** | 50,778 | 56,835.5820 | 57,172.2656 | 57,168.5469 | 57,624.6367 | 57,576.6328 | **57,275.5328** | **12.80%** |
| **`pr1002`** | 259,045 | 307,832.8750 | 307,271.3125 | 306,584.9688 | 307,939.3750 | 308,415.6875 | **307,608.8438** | **18.75%** |
| **Overall Avg** | — | 8.50% | 8.84% | 8.83% | 9.34% | 10.40% | — | **9.18%** |

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
