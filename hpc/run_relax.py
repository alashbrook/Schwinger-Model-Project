#!/usr/bin/env python3
"""One system size L of the exact sector-Lindblad relaxation sweep.

This is the notebook's "How does relaxation time depend on L?" cell, cut down to a
single L so SLURM can run the sizes as an array, and with checkpointing so a size
that outlives its wall-clock limit resumes instead of restarting.

    python run_relax.py --L 12 --out results
    python run_relax.py --selftest          # reproduce the notebook at L = 6, 8

Writes results/tF_L{L}.json: t_F for each fidelity target, the whole F(t) curve,
the RK4 step, the ceiling, and every parameter used. aggregate_relax.py turns a
directory of those into the notebook's table with the * rows filled in.

WHY THIS IS AFFORDABLE WHERE A CIRCUIT SIMULATION IS NOT. H and every L(n) commute
with the total charge, so the dynamics never leaves the initial state's sector and
the matrix evolved is d_sec = C(L, L/2), not 2^L. At L = 16 that is 12,870 instead
of 65,536 -- a dense rho of 2.6 GiB instead of 64 GiB. The Liouvillian is never
assembled; it is only ever applied to a state.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from backends import get_backend  # noqa: E402
from thermal_hpc import (  # noqa: E402
    sector_ops, staggered_bitstring, gibbs_sector, rk4_dt, basis_rho,
    t_at_fidelity, _rk4_stepper,
)

# The notebook's own published numbers, used by --selftest. If a change to the
# notebook moves these, the cluster run is no longer the same measurement.
NOTEBOOK = {
    6: {0.9: 3.36, 0.99: 11.15, 0.999: 23.26},
    8: {0.9: 4.82, 0.99: 16.23, 0.999: 35.07},
}


def curve_with_checkpoint(H_sec, L_sec, init_state, idx, T, sample_dt, infid_floor,
                          t_cap, ckpt_path=None, ckpt_every=900.0, log_every=20.0,
                          dt=None, backend=None):
    """exact_fidelity_curve() with the trajectory checkpointed to disk.

    Structurally identical to the notebook's function -- same RK4 step, same
    sampling interval, same two stopping rules (infidelity floor reached, or the
    infidelity stopped falling) -- with the state written out every `ckpt_every`
    seconds so a job that hits its wall-clock limit can be requeued and resume.
    `--selftest` checks this path against the notebook's numbers.
    """
    bk = backend if backend is not None else get_backend("scipy")
    d = H_sec.shape[0]
    dt = rk4_dt(H_sec) if dt is None else dt
    step = _rk4_stepper(bk.rhs(H_sec, L_sec), dt)
    fine = max(1, int(round(sample_dt / dt)))
    F = bk.fidelity_to(gibbs_sector(H_sec, T))

    rho, t, ts, Fs = None, 0.0, None, None
    if ckpt_path is not None and pathlib.Path(ckpt_path).exists():
        z = np.load(ckpt_path)
        if int(z["d"]) == d and abs(float(z["dt"]) - dt) < 1e-15:
            rho, t = bk.array(z["rho"]), float(z["t"])
            ts, Fs = list(z["ts"]), list(z["Fs"])
            print("  resumed from {} at t = {:.2f} ({} samples)".format(
                ckpt_path, t, len(ts)), flush=True)
        else:
            print("  ignoring {}: built for a different run".format(ckpt_path),
                  flush=True)
    if rho is None:
        rho = bk.array(basis_rho(init_state, idx, d))
        ts, Fs = [0.0], [F(rho)]

    def save():
        if ckpt_path is None:
            return
        tmp = str(ckpt_path) + ".tmp.npz"     # atomic: never leave a torn checkpoint
        np.savez(tmp, rho=bk.to_numpy(rho), t=t, ts=np.array(ts), Fs=np.array(Fs),
                 d=d, dt=dt)
        os.replace(tmp, ckpt_path)

    t_ckpt = t_log = time.time()
    while t < t_cap:
        for _ in range(fine):
            rho = step(rho)
        t += fine * dt
        ts.append(t)
        Fs.append(F(rho))
        infid = 1.0 - Fs[-1]

        now = time.time()
        if now - t_log > log_every:
            print("  t = {:8.2f}   1 - F = {:.3e}   ({} samples)".format(
                t, infid, len(ts)), flush=True)
            t_log = now
        if ckpt_every > 0 and now - t_ckpt > ckpt_every:
            save()
            t_ckpt = now

        if infid <= infid_floor:
            break
        if t > 2.0 and len(Fs) > 6 and (Fs[-1] - Fs[-6]) <= 1e-3 * infid:
            break

    save()
    Fs = np.array(Fs)
    return np.array(ts), Fs, dt, float(1.0 - Fs.max())


def run_one(L, args, ckpt_path=None, backend=None):
    t0 = time.time()
    print("building sector operators for L = {} ...".format(L), flush=True)
    H_sec, L_sec, idx = sector_ops(
        L, args.m, args.e, args.a, args.T, D_kind=args.D_kind, D0=args.D0,
        D_sigma=(args.sigma if args.D_kind == "gaussian" else None), Q0=args.Q0)
    d = H_sec.shape[0]
    print("  d_sec = {}   nnz(H) = {}   dense rho = {:.2f} GiB   ({:.1f}s)".format(
        d, H_sec.nnz, d * d * 16 / 2 ** 30, time.time() - t0), flush=True)

    init = staggered_bitstring(L)
    # The sampling interval only sets the resolution of the threshold times, which
    # t_at_fidelity interpolates anyway; each sample costs an eigh + an svd of a
    # d x d matrix, so at large d it is worth widening.
    sample_dt = args.sample_dt or max(0.25, 0.25 * (d / 252.0) ** 0.5)

    ts, Fs, dt_rk, ceiling = curve_with_checkpoint(
        H_sec, L_sec, init, idx, args.T, sample_dt, args.infid_floor, args.t_cap,
        ckpt_path=ckpt_path, ckpt_every=args.ckpt_every, backend=backend)

    tF = {f: t_at_fidelity(ts, Fs, f) for f in args.targets}
    return dict(
        L=L, d_sec=int(d), dt_RK4=float(dt_rk), sample_dt=float(sample_dt),
        backend=(backend.name if backend is not None else "scipy"),
        init=init, ceiling=ceiling, wall_sec=time.time() - t0,
        t_end=float(ts[-1]), n_samples=int(len(ts)),
        tF={str(f): (None if v is None else float(v)) for f, v in tF.items()},
        curve=dict(ts=[float(x) for x in ts], Fs=[float(x) for x in Fs]),
        params=dict(m=args.m, e=args.e, a=args.a, T=args.T, D_kind=args.D_kind,
                    D0=args.D0, sigma=args.sigma, Q0=args.Q0,
                    infid_floor=args.infid_floor, t_cap=args.t_cap),
        measured=True,
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=int, help="system size (fermion sites)")
    p.add_argument("--L-list", type=int, nargs="+", default=[12, 14, 16],
                   help="sizes an array job indexes into (default: the * rows)")
    p.add_argument("--index", type=int, default=None,
                   help="pick --L-list[index]; defaults to $SLURM_ARRAY_TASK_ID")
    p.add_argument("--out", default="results")
    p.add_argument("--selftest", action="store_true",
                   help="reproduce the notebook at L = 6 and 8, then exit")
    # physics -- defaults are the notebook's fixed-e control panel
    p.add_argument("--m", type=float, default=0.5)
    p.add_argument("--e", type=float, default=0.71)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--T", type=float, default=5.0)
    p.add_argument("--D-kind", default="delta", choices=["delta", "gaussian", "const"])
    p.add_argument("--D0", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--Q0", type=int, default=0)
    # numerics
    p.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.99, 0.999])
    p.add_argument("--sample-dt", type=float, default=None,
                   help="spacing of recorded points (default: scales with d_sec)")
    p.add_argument("--infid-floor", type=float, default=1e-4)
    p.add_argument("--t-cap", type=float, default=4000.0)
    p.add_argument("--ckpt-every", type=float, default=900.0,
                   help="seconds between checkpoints (0 disables)")
    p.add_argument("--ckpt-dir", default=None,
                   help="where to put the resume file (default: --out)")
    p.add_argument("--backend", default="scipy", choices=["scipy", "reference", "mkl", "gpu"],
                   help="who multiplies the matrices (see backends.py)")
    args = p.parse_args()

    bk = get_backend(args.backend)
    print("backend: {} -- {}".format(bk.name, bk.info()), flush=True)

    if args.selftest:
        ok = True
        for L, ref in sorted(NOTEBOOK.items()):
            r = run_one(L, args, ckpt_path=None, backend=bk)
            for f, want in sorted(ref.items()):
                got = r["tF"].get(str(f))
                bad = got is None or abs(got - want) > 0.02
                ok = ok and not bad
                print("  L={}  F={:<6g} notebook {:>6.2f}   here {:>8}   {}".format(
                    L, f, want, "None" if got is None else "{:.2f}".format(got),
                    "MISMATCH" if bad else "ok"))
        print("\nselftest " + ("PASSED" if ok else "FAILED"))
        return 0 if ok else 1

    L = args.L
    if L is None:
        i = args.index
        if i is None:
            i = os.environ.get("SLURM_ARRAY_TASK_ID")
            if i is None:
                p.error("give --L, or --index, or run inside a SLURM array")
            i = int(i)
        L = args.L_list[i]

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = None
    if args.ckpt_every > 0:
        cdir = pathlib.Path(args.ckpt_dir or args.out)
        cdir.mkdir(parents=True, exist_ok=True)
        ckpt = cdir / "ckpt_L{}.npz".format(L)

    print("=== L = {} ===".format(L), flush=True)
    res = run_one(L, args, ckpt_path=ckpt, backend=bk)

    dest = outdir / "tF_L{}.json".format(L)
    dest.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n  " + "  ".join(
        "t(F={:g}) = {}".format(f, res["tF"][str(f)]) for f in args.targets))
    print("  ceiling = {:.1e}   wall = {:.2f} h".format(
        res["ceiling"], res["wall_sec"] / 3600))
    print("  wrote {}".format(dest))
    if ckpt is not None and ckpt.exists():
        ckpt.unlink()          # finished: the resume file is now just dead weight
    return 0


if __name__ == "__main__":
    sys.exit(main())
