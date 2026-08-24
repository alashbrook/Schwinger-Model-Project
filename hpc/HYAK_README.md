# Running the relaxation sweep on HYAK (klone)

The notebook measures `t_F(L)` — the physical time to reach fidelity `F` with the
sector Gibbs state — for **L = 2, 4, 6, 8, 10**, then *extrapolates* a power-law fit to
fill in **L = 12, 14, 16**. Those rows print with a `*` in the projection table.

This directory replaces the `*` with measurements. It does **not** run the notebook on
the cluster; it runs the one cell that is too expensive to run locally, and hands the
numbers back to the notebook.

---

## Why not just run the notebook itself?

You could (`papermill`/`jupyter nbconvert --execute` both work on a compute node), but
it would be the wrong shape for this job:

* Only one section is expensive. Everything else — the circuit builds, the `dt` sweep,
  the CNOT accounting — takes seconds and wants to stay interactive.
* The three sizes differ in cost by a factor of ~1000. They belong in separate jobs
  with separate wall-clock limits, not in one linear notebook.
* L = 14 and L = 16 run for days. A notebook has nowhere to put a checkpoint, so any
  preemption or timeout throws the whole run away.

So the expensive cell is extracted into a script. To keep the script honest,
`thermal_hpc.py` is **generated from the notebook's own cells**, verbatim — see
`build_thermal_hpc.py`. Nothing is retyped or paraphrased.

```
build_thermal_hpc.py   notebook cells  ->  thermal_hpc.py     (regenerate after editing the notebook)
run_relax.py           one L, checkpointed  ->  results/tF_L{L}.json
backends.py            scipy | mkl | gpu -- who multiplies the matrices
aggregate_relax.py     results/*.json  ->  tF_measured.json + summary.csv + the comparison table
relax.slurm            one array task per size
submit.sh              submits each size with resources matched to its cost
setup_env.sh           one-time conda env on the login node
```

---

## What it costs

Timed by running one RK4 step at each size on a laptop core. `d_sec = C(L, L/2)` is the
size of the matrix actually evolved — the charge sector, not the full `2^L`.

| L | `d_sec` | dense ρ | RK4 step | fidelity eval | RK4 steps | **1 core** |
|---|---|---|---|---|---|---|
| 10 | 252 | 1 MiB | 0.12 s | 0.21 s | 684 | 3.6 min (measured) |
| 12 | 924 | 13 MiB | 2.1 s | 3.0 s | ~1,300 | **~1 h** |
| 14 | 3,432 | 180 MiB | 49 s | 85 s | ~2,400 | **~35 h** |
| 16 | 12,870 | 2.6 GiB | ~880 s\* | ~4,450 s\* | ~3,500 | **~36 days**\* |

\* extrapolated from the L = 12→14 scaling; everything else is measured. The step
count is set by `t_end / dt_RK4`, where `dt_RK4 = 0.9/‖H‖∞` shrinks with L *and*
`t_end` (the time to hit the `1e-4` infidelity floor) grows with it — so each size
costs roughly 20× the one before.

The L = 10 row was run through this exact pipeline as a check and returned
6.2940835 / 21.7416885 / 47.2559707, matching the notebook's 6.29 / 21.74 / 47.26.

**L = 12 and L = 14 are ordinary CPU jobs.** L = 16 is not: a month of single-core time
is more than a wall-clock limit allows, and the cost is in `scipy`'s sparse×dense
product, which is single-threaded. Three ways out, in order of preference:

1. **Stop at L = 14.** Measuring 12 and 14 already tests the extrapolation and refits
   `t_F(L)` on 7 sizes instead of 5, which tightens the L = 16 *prediction* a lot. This
   is the cheapest real answer and is what `submit.sh` does by default.

   There is already a reason to expect the `*` rows to be **low**. Holding out L = 10
   and fitting only L = 2…8 — the same power law, one size shorter — under-predicts the
   measured L = 10 by 6.0% / 2.9% / 7.8% at F = 0.9 / 0.99 / 0.999. The fit
   systematically undershoots as it extrapolates, so treat the notebook's 12/14/16
   cycle counts as a floor, not a central estimate. `aggregate_relax.py` prints this
   same held-out comparison for whichever sizes you measure.
2. **GPU** — `--backend gpu` (cupy/cuSPARSE). Needs ~25 GiB of card memory for the RK4
   temporaries, so an A40/L40 (48 GB) or A100. Expect well under a day.
3. **Threaded CPU** — `--backend mkl` (`pip install sparse-dot-mkl`) on a 40-core node.
   Roughly 10× if the sparse kernel threads well, so ~3–4 days across several requeued
   jobs. The checkpointing is what makes that survivable.

Memory is never the binding constraint: the largest object is a 2.6 GiB density matrix
and RK4 keeps about eight of them, so ~40 GB covers L = 16.

---

## Setup (once)

```bash
ssh <uwnetid>@klone.hyak.uw.edu          # UW 2FA

hyakalloc                                # <-- your account name(s) and partitions
```

Work out of `/gscratch`, not `$HOME` — home directories are small and this writes
checkpoints of a few GB.

```bash
cd /gscratch/<your-group>/<uwnetid>
git clone <this repo> Thermal-State-Project      # or: rsync from your laptop
cd Thermal-State-Project

bash hpc/setup_env.sh                    # ON THE LOGIN NODE -- compute nodes have no internet
```

`setup_env.sh` installs miniforge, creates the `thermal` env from the pinned
`requirements.txt`, and finishes by running

```bash
python hpc/run_relax.py --selftest
```

which recomputes L = 6 and L = 8 and checks them against the numbers the notebook
printed (3.36 / 11.15 / 23.26 and 4.82 / 16.23 / 35.07). **If the selftest does not say
PASSED, stop** — the environment is computing something other than what the notebook
computed, and no production result from it is worth having.

If you already keep a conda somewhere else, skip the install and point at it:
`CONDA_ROOT=/gscratch/<group>/miniconda3 ENV_NAME=myenv bash hpc/setup_env.sh`.

---

## Submitting

```bash
cd /gscratch/<your-group>/<uwnetid>/Thermal-State-Project
ACCOUNT=<your-account> PARTITION=compute bash hpc/submit.sh
```

That submits L = 12 and L = 14 with their own time/memory, plus a dependent
aggregation job that runs when they land. To include L = 16:

```bash
ACCOUNT=<acct> PARTITION=gpu-a40 BACKEND=gpu SIZES="16" bash hpc/submit.sh
# or, CPU:
ACCOUNT=<acct> PARTITION=compute BACKEND=mkl SIZES="16" bash hpc/submit.sh
```

Verify a non-default backend before spending days on it — it is a different matrix
kernel, so make it prove itself first:

```bash
python hpc/run_relax.py --selftest --backend mkl
python hpc/run_relax.py --selftest --backend gpu
```

Monitoring:

```bash
squeue -u $USER
tail -f logs/relax_*.out
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem
```

### Checkpointing and preemption

`run_relax.py` writes `results/ckpt_L{L}.npz` every 30 minutes and resumes from it
automatically. Resubmitting the identical command after a timeout or preemption picks
up where it stopped — it does not restart. This is what makes the free, preemptible
`ckpt` partition usable for the long sizes:

```bash
ACCOUNT=<acct> PARTITION=ckpt bash hpc/submit.sh
```

`relax.slurm` sets `--requeue`, so SLURM puts a preempted job back in the queue itself.
The checkpoint is deleted once the size finishes.

---

## Getting the answer back into the notebook

```bash
python hpc/aggregate_relax.py --results hpc/results
```

prints the merged `t_F(L)` table, the refit, and the comparison that is the point:

```
  Extrapolation vs measurement -- was the * row right?
      L   target    predicted     measured    error
  -----------------------------------------------
     12      0.9         ...          ...      ...
```

and writes `hpc/results/tF_measured.json`. Copy that back to your laptop:

```bash
# from your laptop
scp <uwnetid>@klone.hyak.uw.edu:/gscratch/<group>/<uwnetid>/Thermal-State-Project/hpc/results/tF_measured.json \
    hpc/results/
```

Then paste the cell in `notebook_merge_cell.py` into the notebook **immediately after
the relaxation sweep cell** (`id: 56ce1835`, the one that prints the `L / d_sec /
dt_RK4 / t(F=…)` table) and re-run from there. It folds the measured sizes into
`relax`, refits, and the projection table's `*` markers disappear for whichever sizes
came back — the downstream cells need no edits.

---

## Gotchas

* **Regenerate after editing the notebook.** `thermal_hpc.py` is generated. If you
  change `sector_ops`, `exact_fidelity_curve`, or the Hamiltonian in the notebook, run
  `python hpc/build_thermal_hpc.py` again or the cluster keeps computing the old thing.
  `build_thermal_hpc.py` addresses cells by **id**, so inserting cells is safe, but
  deleting or renumbering one makes it fail loudly rather than silently pick up the
  wrong code.
* **`sample_dt` widens automatically with `d_sec`** (`0.25·√(d/252)`), because each
  recorded point costs an `eigh` plus an `svd` of a `d×d` matrix — at L = 16 that is
  over an hour per sample. `t_at_fidelity` interpolates in `log(1−F)`, so the threshold
  times stay far better resolved than the sampling interval. Pass `--sample-dt` to
  override.
* **The three sizes are independent jobs.** If L = 16 never finishes, L = 12 and L = 14
  still aggregate — the dependency is `afterany`, not `afterok`.
* **The `*` rows in the notebook are a *fit*, not a bound.** The extrapolation rests on
  `t_F ~ L^1.24` (R² = 0.9935 over 5 sizes) and, separately, on the `c(L)` infidelity
  floor fitted through only **two** calibration sizes (L = 4 and 6). This sweep fixes
  the first. It does not touch the second — the feasibility column ("dt too coarse")
  still rests on a two-point extrapolation, and measuring it would need a
  density-matrix *circuit* simulation at L ≥ 8, which is a different and much more
  expensive job (`2^(L+n_aux)` amplitudes, not `d_sec`).
