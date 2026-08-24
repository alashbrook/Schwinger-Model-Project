# ============================================================
#  Fold the cluster's measured sizes into the sweep above
# ============================================================
# Paste this cell into the notebook DIRECTLY AFTER the relaxation sweep cell
# (id 56ce1835 -- the one printing the L / d_sec / dt_RK4 / t(F=...) table) and re-run
# from here down.  It reads hpc/results/tF_measured.json, written by
# hpc/aggregate_relax.py from the HYAK jobs, and merges those sizes into `relax`.
#
# The sizes it adds were computed by hpc/thermal_hpc.py, which is GENERATED from this
# notebook's own cells -- same sector_ops, same exact_fidelity_curve, same RK4 step --
# so they are the same measurement as the rows above, just run somewhere with more
# time.  hpc/run_relax.py --selftest re-derives L = 6 and 8 on the cluster and checks
# them against this notebook's printed values before any production size is trusted.
#
# Everything downstream (the projection table, the plots) picks the change up with no
# edits: t_F_of_L() finds these sizes in `relax` and stops marking them `*`.
import json
import pathlib

MEASURED = pathlib.Path("hpc/results/tF_measured.json")

if not MEASURED.exists():
    print(f"  no {MEASURED} -- the projection below stays extrapolated.")
    print("  See hpc/HYAK_README.md to produce it.")
else:
    blob = json.loads(MEASURED.read_text(encoding="utf-8"))
    added = []
    for L_str, rec in blob.items():
        L_new = int(L_str)
        if not rec.get("from_cluster"):
            continue                      # a size this notebook already measured
        tF_new = {f: rec["tF"].get(str(f)) for f in F_TARGETS}
        curve = rec.get("curve") or {"ts": [], "Fs": []}
        relax[L_new] = dict(
            d_sec=rec["d_sec"],
            ts=np.array(curve["ts"]), F=np.array(curve["Fs"]),
            tF=tF_new, dt_rk=rec["dt_RK4"], ceiling=rec["ceiling"],
            init=rec["init"], secs=float("nan"), from_cluster=True)
        added.append(L_new)

    if not added:
        print(f"  {MEASURED} holds no cluster sizes yet.")
    else:
        L_relax = sorted(set(L_relax) | set(added))

        # Recomputed exactly as the sweep cell does, now over the longer list.
        Ls = [L for L in L_relax if L in relax]
        got = lambda f: np.array([relax[L]["tF"][f] for L in Ls], dtype=float)
        ok = [f for f in F_TARGETS if all(relax[L]["tF"][f] is not None for L in Ls)]

        print(f"  Merged L = {added} from the cluster; the sweep now covers {Ls}.")
        print(f"\n{'L':>7}{'d_sec':>9}" + "".join(f"{'t(F=' + f'{f:g}' + ')':>13}"
                                                  for f in F_TARGETS)
              + f"{'ceiling':>11}   source")
        print("  " + "-" * (16 + 13 * len(F_TARGETS) + 22))
        for L in Ls:
            r = relax[L]
            row = f"{L:>7}{r['d_sec']:>9}"
            for f in F_TARGETS:
                v = r["tF"][f]
                row += f"{v:>13.2f}" if v is not None else f"{'never':>13}"
            src = "HYAK" if r.get("from_cluster") else "this notebook"
            print("  " + row + f"{r['ceiling']:>11.1e}   {src}")

        # --- what the extrapolation had predicted, before it was measured ---------
        if fits:
            print("\n  Was the extrapolation right?   (fit through the pre-cluster sizes)")
            print(f"  {'L':>5}{'target':>9}{'predicted':>12}{'measured':>12}{'error':>9}")
            print("  " + "-" * 47)
            for L in added:
                for f in ok:
                    v = relax[L]["tF"][f]
                    if v is None:
                        continue
                    p, A, r2p = fits[f]["power"]
                    c, B, r2e = fits[f]["exponential"]
                    pred = A * L ** p if r2p >= r2e else B * np.exp(c * L)
                    print(f"  {L:>5}{f:>9g}{pred:>12.2f}{v:>12.2f}"
                          f"{100 * (pred - v) / v:>8.1f}%")

        # --- refit on the longer lever arm ---------------------------------------
        fits = {f: fit_scaling(Ls, got(f)) for f in ok} if len(Ls) >= 2 else {}
        print(f"\n  Refit over {len(Ls)} sizes (L = {Ls[0]}..{Ls[-1]}):")
        print(f"{'target':>10}{'power p':>10}{'R^2':>8}{'exp rate c':>13}{'R^2':>8}"
              f"   better")
        for f in ok:
            p, _, r2p = fits[f]["power"]
            c, _, r2e = fits[f]["exponential"]
            print(f"{f:>10g}{p:>10.2f}{r2p:>8.4f}{c:>13.3f}{r2e:>8.4f}   "
                  f"{'exponential' if r2e > r2p else 'power law'}")
        print("  The projection table below now MEASURES these sizes rather than "
              "extrapolating\n  to them, so their `*` markers are gone. The rows past "
              f"L = {Ls[-1]} are still fits --\n  and note the feasibility column "
              "('dt too coarse') rests on the SEPARATE two-point\n  c(L) calibration, "
              "which this sweep does not touch.")
