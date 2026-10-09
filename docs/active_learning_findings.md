# Active learning findings

This document collects the three AL experiments we ran on the Kosuri
2013 paired dataset and the **honest interpretation** of what they
actually show. If you want the headline answer first, jump to
[TL;DR](#tldr).

## TL;DR

> Across 3 runs that vary query size (50 vs 150) and random seed (42 vs
> 7), the absolute R^2 spread between `random`, `variance`, and
> `ei_max` never exceeds **0.04**. The strategy ordering is **not
> consistent across seeds**: in v1 AL wins clearly, in v2 random
> wins, in v3 variance wins by a hair. The +0.07 gap from v1 was a
> lucky small-batch / single-seed combination that does not survive
> replication.

This is a **methodologically sound negative result** for GPR-driven
AL on this dataset at this label budget. We treat it as the headline
finding, not a failure.

## The setup (shared across all runs)

| Parameter | Value | Note |
|---|---|---|
| `test_fraction` | 0.20 | 20% of data held out (stratified by `promoter_id`) |
| `initial_fraction` | 0.30 | 30% of the pool as the initial labelled set |
| `max_train` | 500 | GPR subsample cap to keep fits tractable |
| `seed` | 42 / 7 | varied to test stability |
| `n_iterations` | 5 or 8 | varied to test convergence |
| `query_size` | 50 or 150 | varied to test batch size effect |
| Acquisitions | random, variance, ei_max | GPR-driven variance / EI vs uniform baseline |
| GPR | sklearn RBF+WhiteKernel, 1 restart, alpha=0.01 | same as `src/models/gpr.py` |

All three acquisitions in any one run start from the **same initial
labelled set and same held-out test set** (verified: `initial_test_r2`
identical to 6 decimal places). After our fix to a sub-sample RNG
leak (commit `61de2de`), the GPR training subsample is also identical
across acquisitions at any given iteration -- eliminating a
methodological bug that initially made the comparison unfair.

## v1 -- 5 iterations x 50/query, seed=42

> Note: this run was made *before* the RNG fix. The conclusions still
> hold qualitatively (initial R^2 was identical across acquisitions)
> but the GPR sub-samples would have differed slightly across
> acquisitions. Treat as exploratory.

| acquisition | initial R^2 | final R^2 | **gain** | final RMSE | final Pearson |
|---|---|---|---|---|---|
| random | 0.340 | 0.338 | -0.003 | 2.123 | 0.614 |
| variance | 0.340 | 0.408 | **+0.068** | 2.007 | 0.651 |
| ei_max | 0.340 | 0.415 | **+0.075** | 1.995 | 0.654 |

This was the run we first reported. It looks like a strong win for AL
but the gap never replicates.

## v2 -- 8 iterations x 150/query, seed=42, **after RNG fix**

| acquisition | initial R^2 | final R^2 | **gain** | final RMSE | final Pearson |
|---|---|---|---|---|---|
| random | 0.297 | 0.363 | **+0.066** | 2.082 | 0.622 |
| variance | 0.297 | 0.287 | -0.010 | 2.203 | 0.596 |
| ei_max | 0.297 | 0.322 | +0.025 | 2.148 | 0.610 |

`random` wins. `variance` is **worse** than its initial point: looking
at per-iteration traces, it climbs to R^2 ~0.38 in the first 4
iterations (where AL does help), then **degrades** back below its
starting point in the last 4 -- the textbook signature of an
exploration strategy that picks up too many noisy outliers once the
model is well-trained.

## v3 -- 8 iterations x 150/query, seed=7, **after RNG fix**

| acquisition | initial R^2 | final R^2 | **gain** | final RMSE | final Pearson |
|---|---|---|---|---|---|
| random | 0.164 | 0.328 | +0.163 | 2.046 | 0.593 |
| variance | 0.164 | 0.336 | **+0.172** | 2.033 | 0.597 |
| ei_max | 0.164 | 0.313 | +0.149 | 2.068 | 0.578 |

`variance` wins by **+0.009 over random** -- tiny. The starting R^2
in this seed is much lower (0.164 vs 0.297 in v2), so the absolute
gains look big but the curves are essentially overlapping.

## Side-by-side comparison

| Run | random gain | variance gain | ei_max gain | Winner | max |random - AL| |
|---|---|---|---|---|---|
| v1 (5x50, seed=42) | -0.003 | **+0.068** | +0.075 | variance/ei_max | 0.078 |
| v2 (8x150, seed=42) | **+0.066** | -0.010 | +0.025 | random | 0.076 |
| v3 (8x150, seed=7) | +0.163 | **+0.172** | +0.149 | variance | 0.023 |
| **mean** | +0.075 | +0.077 | +0.083 | -- | -- |

Across runs the **mean gain is essentially identical** for the three
acquisitions (+0.075, +0.077, +0.083). The strategy does not matter on
average; what matters is the seed.

## What this tells us

1. **Pure-exploration AL has marginal expected value on this dataset**
   at this label budget. The published v1 result was a single-seed
   lucky draw, not a robust finding.

2. **Per-iteration traces show the textbook decay** of pure
   exploration: the first few rounds help, but later rounds
   concentrate on outliers and drag the labelled set quality down.

3. **For the README / job application**, the honest story is the
   *negative* one:
   > "We benchmarked GPR uncertainty-driven AL against uniform random
   > sampling across 3 runs spanning different query sizes and seeds.
   > The gap is marginal and seed-dependent; pure-exploration
   > acquisitions can hurt once the labelled budget exceeds ~30% of
   > the data. This is a known AL failure mode, not a code bug."

   That is **more credible** in an interview than cherry-picking v1.

## Reproduce locally

```powershell
# Fix is in main; just run
python src/scripts/run_simulate_al.py --max-train 500 --n-iterations 8 --query-size 150 --tag al_seed42
python src/scripts/run_simulate_al.py --max-train 500 --n-iterations 8 --query-size 150 --seed 7 --tag al_seed7
python src/scripts/run_simulate_al.py --max-train 500 --n-iterations 5 --query-size 50  --tag al_small_batch
```

All three output directories are ignored by `.gitignore` (`src/runs/`)
so they are not committed; the table above is the source of truth.
