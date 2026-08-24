#!/usr/bin/env python3
"""Collect the per-size JSON files into the notebook's table, with the * rows measured.

    python hpc/aggregate_relax.py --results hpc/results

Reads every tF_L*.json in --results, merges them with the sizes the notebook already
measured (passed with --known, or the notebook's published L = 2..10 by default),
refits t_F(L), and writes

    <results>/tF_measured.json    what the notebook cell reads back in
    <results>/summary.csv         one row per size

and prints the comparison that is the point of the whole exercise: what the power-law
fit through L = 2..10 PREDICTED at 12, 14, 16 versus what the cluster actually measured.
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from thermal_hpc import fit_scaling  # noqa: E402

# What the notebook measured in-session (cell 57), so the fit here is the same fit.
NOTEBOOK = {
    2: {"0.9": 1.06, "0.99": 2.96, "0.999": 6.53},
    4: {"0.9": 2.05, "0.99": 6.93, "0.999": 13.19},
    6: {"0.9": 3.36, "0.99": 11.15, "0.999": 23.26},
    8: {"0.9": 4.82, "0.99": 16.23, "0.999": 35.07},
    10: {"0.9": 6.29, "0.99": 21.74, "0.999": 47.26},
}
D_SEC = {2: 2, 4: 6, 6: 20, 8: 70, 10: 252, 12: 924, 14: 3432, 16: 12870}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="hpc/results")
    p.add_argument("--targets", nargs="+", default=["0.9", "0.99", "0.999"])
    args = p.parse_args()

    rdir = pathlib.Path(args.results)
    runs = {}
    for f in sorted(rdir.glob("tF_L*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        runs[int(r["L"])] = r
    if not runs:
        sys.exit("no tF_L*.json in {} -- did the jobs finish?".format(rdir))

    # The fit the notebook extrapolated with. Any size the cluster re-measured is
    # dropped from it, so "predicted vs measured" below is always OUT of sample --
    # otherwise re-running an existing size would compare a fit against a point it
    # was fitted to, and flatter itself.
    base = sorted(set(NOTEBOOK) - set(runs))
    if len(base) < 2:
        sys.exit("need at least two un-re-measured notebook sizes to fit against")
    old_fits = {t: fit_scaling(base, [NOTEBOOK[L][t] for L in base])
                for t in args.targets}

    merged = {str(L): {t: v for t, v in NOTEBOOK[L].items()} for L in base}
    for L, r in runs.items():
        merged[str(L)] = {t: r["tF"].get(t) for t in args.targets}

    print("\n  t_F(L), lattice units    (rows from the cluster are marked NEW)")
    hdr = "{:>5}{:>9}".format("L", "d_sec") + "".join(
        "{:>13}".format("t(F=" + t + ")") for t in args.targets)
    print(hdr + "{:>10}{:>9}".format("ceiling", "wall h"))
    print("  " + "-" * (len(hdr) + 17))
    for L in sorted(int(k) for k in merged):
        r = runs.get(L)
        row = "{:>5}{:>9}".format(L, D_SEC.get(L, r["d_sec"] if r else "?"))
        for t in args.targets:
            v = merged[str(L)].get(t)
            row += "{:>13}".format("never" if v is None else "{:.2f}".format(v))
        row += "{:>10}{:>9}".format(
            "{:.1e}".format(r["ceiling"]) if r else "-",
            "{:.2f}".format(r["wall_sec"] / 3600) if r else "-")
        print("  " + row + ("   NEW" if r else ""))

    print("\n  Extrapolation vs measurement -- was the * row right?")
    print("  (fit through L = {}, none of which was re-measured here)".format(
        ", ".join(str(L) for L in base)))
    print("  {:>5}{:>9}{:>13}{:>13}{:>9}".format(
        "L", "target", "predicted", "measured", "error"))
    print("  " + "-" * 49)
    for L in sorted(runs):
        for t in args.targets:
            got = merged[str(L)].get(t)
            if got is None:
                continue
            pw, A, r2p = old_fits[t]["power"]
            c, B, r2e = old_fits[t]["exponential"]
            pred = A * L ** pw if r2p >= r2e else B * np.exp(c * L)
            print("  {:>5}{:>9}{:>13.2f}{:>13.2f}{:>9}".format(
                L, t, pred, got, "{:+.1f}%".format(100 * (pred - got) / got)))

    print("\n  Refit including the new sizes:")
    print("  {:>9}{:>10}{:>9}{:>13}{:>9}   better".format(
        "target", "power p", "R^2", "exp rate c", "R^2"))
    Ls_all = sorted(int(k) for k in merged
                    if all(merged[k].get(t) is not None for t in args.targets))
    new_fits = {}
    for t in args.targets:
        ys = [merged[str(L)][t] for L in Ls_all]
        fit = fit_scaling(Ls_all, ys)
        new_fits[t] = fit
        (pw, _, r2p), (c, _, r2e) = fit["power"], fit["exponential"]
        print("  {:>9}{:>10.2f}{:>9.4f}{:>13.3f}{:>9.4f}   {}".format(
            t, pw, r2p, c, r2e, "exponential" if r2e > r2p else "power law"))
        old_p = old_fits[t]["power"][0]
        print("        (the notebook's L = 2..10 fit gave p = {:.2f}; "
              "{} sizes now)".format(old_p, len(Ls_all)))

    out = rdir / "tF_measured.json"
    out.write_text(json.dumps(
        {str(L): {"tF": merged[str(L)],
                  "d_sec": D_SEC.get(L),
                  "ceiling": runs[L]["ceiling"] if L in runs else None,
                  "dt_RK4": runs[L]["dt_RK4"] if L in runs else None,
                  "curve": runs[L]["curve"] if L in runs else None,
                  "init": runs[L]["init"] if L in runs else None,
                  "from_cluster": L in runs}
         for L in sorted(int(k) for k in merged)}, indent=1), encoding="utf-8")
    print("\n  wrote {}  (load this from the notebook)".format(out))

    csv_path = rdir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["L", "d_sec", "from_cluster"]
                   + ["tF_" + t for t in args.targets] + ["ceiling", "wall_sec"])
        for L in sorted(int(k) for k in merged):
            r = runs.get(L)
            w.writerow([L, D_SEC.get(L), bool(r)]
                       + [merged[str(L)].get(t) for t in args.targets]
                       + [r["ceiling"] if r else "", r["wall_sec"] if r else ""])
    print("  wrote {}".format(csv_path))


if __name__ == "__main__":
    main()
