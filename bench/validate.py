#!/usr/bin/env python3
"""Phase 4 gate check: does the model predict, and does it rank?

Joins the measured sweep with model predictions and reports two different things,
because they are different and a model can pass one while failing the other.

ABSOLUTE ACCURACY is what the project promised: mean absolute percentage error
with the full distribution, not just the mean.

RANKING QUALITY is what the model is actually for. An autotuner does not consume
TFLOP/s, it consumes an ordering. A uniform 40% under-prediction ranks perfectly.
An 8% error that varies by config can rank badly. See the Phase 4 gate in
PLAN.md.

TIES ARE REPORTED AS RANGES, NOT COLLAPSED. PLAN.md Finding 11 records that the
current model predicts only four distinct values per shape, with a fourteen-way
tie among the 128x128 configs. When the top prediction is a tie, "the model's top
pick" is not a single config, and reporting one number for recall or regret would
be inventing a decision the model did not make. So every ranking metric that can
be affected by ties is reported as [pessimistic, optimistic]: what an autotuner
would get if it broke ties in the worst possible order, and in the best. A wide
range is itself the finding.

Predictions come from `ridge-predict --batch`, never from a Python
reimplementation of the model. A second copy of the math would drift silently and
the first symptom would be a validation result describing a model nobody ships.

No third-party dependencies. The statistics are short enough to write out, and a
gate that fails because numpy is missing is a bad gate.
"""

import argparse
import collections
import csv
import pathlib
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Statistics, written out rather than imported
# ---------------------------------------------------------------------------


def average_ranks(xs):
    """Ranks with ties averaged, which is what Spearman requires.

    Using ordinal ranks instead would silently invent an ordering inside every
    tie group and inflate the correlation, which matters enormously here because
    the current model produces very large tie groups.
    """
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    if da == 0.0 or db == 0.0:
        # No variance on one side. With the current model this happens whenever
        # every config in a shape predicts the same value, and it is a real
        # result rather than a numerical accident, so it is reported as such.
        return float("nan")
    return num / (da * db)


def spearman(a, b):
    return pearson(average_ranks(a), average_ranks(b))


def percentile(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_registered(path):
    shapes = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split(",")
        shapes.append((int(p[0]), int(p[1]), int(p[2])))
    return shapes


def load_variants(path):
    src = open(path).read()
    return [
        tuple(map(int, v))
        for v in re.findall(
            r"RIDGE_VARIANT\((\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)\)",
            src,
        )
    ]


def load_csv_rows(path):
    return list(csv.DictReader(r for r in open(path) if not r.startswith("#")))


def read_meta(path):
    """Pulls the '# key,value' provenance lines measure.cu writes."""
    meta = {}
    for line in open(path):
        if not line.startswith("#"):
            continue
        parts = line[1:].strip().split(",", 1)
        if len(parts) == 2:
            meta[parts[0].strip()] = parts[1].strip()
    return meta


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def check_sweep_matches_registration(rows, registered, variants):
    """The measured set must be exactly the registered cross product.

    A sweep that quietly lost its inconvenient configs would otherwise produce a
    better error number and pass. See PLAN.md anti-pattern 8.
    """
    expected = set()
    for M, N, K in registered:
        for BM, BN, BK, WM, WN, S, G, _ in variants:
            expected.add((M, N, K, BM, BN, BK, WM, WN, S, G))
    got = set()
    for r in rows:
        got.add(
            (
                int(r["M"]), int(r["N"]), int(r["K"]),
                int(r["BM"]), int(r["BN"]), int(r["BK"]),
                int(r["WM"]), int(r["WN"]), int(r["stages"]), int(r["groupM"]),
            )
        )
    missing, extra = expected - got, got - expected
    return missing, extra


def rank_bounds(rows, key_pred, key_meas):
    """Where the measured-best config sits in the predicted ordering.

    Returns (best_rank, worst_rank). They differ when the measured-best config is
    inside a predicted tie group, because an autotuner breaking that tie could
    place it anywhere in the group.
    """
    best = max(rows, key=lambda r: float(r[key_meas]))
    p = float(best[key_pred])
    strictly_better = sum(1 for r in rows if float(r[key_pred]) > p)
    tied = sum(1 for r in rows if float(r[key_pred]) == p)
    return strictly_better + 1, strictly_better + tied


def regret_bounds(rows, key_pred, key_meas):
    """Throughput lost by taking the model's top pick, as a fraction.

    With ties at the top, the model has not chosen a single config, so this
    returns the range over the tie group rather than a single number.
    """
    measured_best = max(float(r[key_meas]) for r in rows)
    top_pred = max(float(r[key_pred]) for r in rows)
    tied = [float(r[key_meas]) for r in rows if float(r[key_pred]) == top_pred]
    best_case = (measured_best - max(tied)) / measured_best
    worst_case = (measured_best - min(tied)) / measured_best
    return worst_case, best_case, len(tied)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", default="data/measured/a100.csv")
    ap.add_argument("--registered", default="data/sweep/phase4-registered.csv")
    ap.add_argument("--variants", default="bench/kernels/gemm-mma.cu")
    ap.add_argument("--hw", default="data/hardware/a100-sxm4-40gb.json")
    ap.add_argument("--predict-bin", default="build/ridge-predict")
    ap.add_argument("--ncu", default=None,
                    help="machine-readable ncu export for bottleneck validation")
    ap.add_argument("--max-mape", type=float, default=10.0)
    ap.add_argument("--fail-mape", type=float, default=15.0)
    args = ap.parse_args()

    for p in (args.measured, args.registered, args.variants, args.hw):
        if not pathlib.Path(p).exists():
            sys.exit(f"missing {p}")

    meta = read_meta(args.measured)
    registered = load_registered(args.registered)
    variants = load_variants(args.variants)

    print("=" * 74)
    print("Ridge Phase 4 validation")
    print("=" * 74)
    for k in ("gpu", "sms", "nominal_clock_mhz", "warmup_seconds",
              "canary_drift_fraction"):
        if k in meta:
            print(f"  {k:24} {meta[k]}")
    print(f"  {'hardware model':24} {args.hw}")
    print()

    failures = []

    # --- predictions, from the C++ model ---------------------------------
    with tempfile.TemporaryDirectory() as td:
        pred_path = str(pathlib.Path(td) / "pred.csv")
        proc = subprocess.run(
            [args.predict_bin, "--hw", args.hw,
             "--batch", args.measured, "--out", pred_path],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            sys.exit(f"ridge-predict failed:\n{proc.stderr}")
        sys.stderr.write(proc.stderr)
        rows = load_csv_rows(pred_path)

    if not rows:
        sys.exit("no rows after prediction")

    # --- gate: the sweep is the registered sweep -------------------------
    missing, extra = check_sweep_matches_registration(rows, registered, variants)
    print(f"registered cross product: {len(registered)} shapes x {len(variants)} "
          f"variants = {len(registered) * len(variants)}")
    print(f"measured rows:            {len(rows)}")
    if missing or extra:
        print(f"  MISSING from measured: {len(missing)}")
        for m in sorted(missing)[:5]:
            print(f"    {m}")
        print(f"  EXTRA in measured:     {len(extra)}")
        failures.append(
            "measured set does not match the registered sweep. A narrowed sweep "
            "produces a better number and is exactly what anti-pattern 8 forbids."
        )
    else:
        print("  matches the registered sweep exactly")
    print()

    # --- absolute accuracy ------------------------------------------------
    errs = []
    for r in rows:
        m, p = float(r["tflops"]), float(r["pred_tflops"])
        r["_abs_pct_err"] = abs(p - m) / m * 100.0
        r["_signed_pct_err"] = (p - m) / m * 100.0
        errs.append(r["_abs_pct_err"])

    mape = sum(errs) / len(errs)
    signed = [r["_signed_pct_err"] for r in rows]
    print("absolute accuracy")
    print(f"  {'MAPE':22} {mape:8.2f} %   (gate: <= {args.max_mape:.0f}%)")
    print(f"  {'median abs error':22} {percentile(errs, 50):8.2f} %")
    print(f"  {'p90 abs error':22} {percentile(errs, 90):8.2f} %")
    print(f"  {'max abs error':22} {max(errs):8.2f} %")
    print(f"  {'mean signed error':22} {sum(signed)/len(signed):+8.2f} %   "
          f"(negative = model under-predicts)")
    print()
    print("  worst 8 configs by absolute error:")
    print("    %-22s %-18s %9s %9s %8s" %
          ("shape", "tile/warp/stages", "measured", "predicted", "err%"))
    for r in sorted(rows, key=lambda r: -r["_abs_pct_err"])[:8]:
        cfg = f"{r['BM']}x{r['BN']}x{r['BK']} w{r['WM']}x{r['WN']} s{r['stages']}"
        print("    %-22s %-18s %9.1f %9.1f %+8.1f" %
              (f"{r['M']}x{r['N']}x{r['K']}", cfg,
               float(r["tflops"]), float(r["pred_tflops"]), r["_signed_pct_err"]))
    print()

    if mape > args.fail_mape:
        failures.append(
            f"MAPE {mape:.1f}% is above the {args.fail_mape:.0f}% ceiling. The "
            f"model is wrong, iterate rather than ship."
        )
    elif mape > args.max_mape:
        failures.append(f"MAPE {mape:.1f}% exceeds the {args.max_mape:.0f}% gate.")

    # --- ranking, per shape ----------------------------------------------
    by_shape = collections.OrderedDict()
    for r in rows:
        by_shape.setdefault((r["M"], r["N"], r["K"], r.get("tag", "")), []).append(r)

    print("ranking quality, per problem shape")
    print("  an autotuner selects for one shape at a time, so these are not pooled")
    print()
    print("  %-22s %8s %7s %13s %13s %16s" %
          ("shape", "spearman", "ties", "rank of best", "recall@3", "regret@1 %"))

    spearmans, worst_regrets, tie_sizes = [], [], []
    for (M, N, K, tag), rs in by_shape.items():
        meas = [float(r["tflops"]) for r in rs]
        pred = [float(r["pred_tflops"]) for r in rs]
        rho = spearman(meas, pred)
        best_rank, worst_rank = rank_bounds(rs, "pred_tflops", "tflops")
        worst_reg, best_reg, ntied = regret_bounds(rs, "pred_tflops", "tflops")
        distinct = len(set(round(p, 6) for p in pred))
        largest_tie = max(collections.Counter(round(p, 6) for p in pred).values())

        spearmans.append(rho)
        worst_regrets.append(worst_reg)
        tie_sizes.append(largest_tie)

        rho_s = "  n/a  " if rho != rho else f"{rho:7.3f}"
        print("  %-22s %8s %7d %13s %13s %16s" % (
            f"{tag or ''} {M}x{N}x{K}"[:22],
            rho_s,
            largest_tie,
            f"{best_rank}-{worst_rank}" if best_rank != worst_rank else f"{best_rank}",
            f"{'yes' if worst_rank <= 3 else ('maybe' if best_rank <= 3 else 'no')}",
            f"{worst_reg*100:.1f}-{best_reg*100:.1f}" if ntied > 1
            else f"{worst_reg*100:.1f}",
        ))

    valid_rho = [r for r in spearmans if r == r]
    print()
    print(f"  {'mean Spearman':26} "
          f"{(sum(valid_rho)/len(valid_rho)) if valid_rho else float('nan'):7.3f}"
          f"   ({len(spearmans)-len(valid_rho)} shapes had no prediction variance)")
    print(f"  {'worst-case regret@1':26} {max(worst_regrets)*100:7.2f} %")
    print(f"  {'largest tie group seen':26} {max(tie_sizes):7d} configs")
    print()

    if max(tie_sizes) > 1:
        print("  NOTE: the model predicts identical values for groups of configs, so")
        print("  'the model's top pick' is not a single config. Ranking figures are")
        print("  ranges over how a tie could break. See PLAN.md Finding 11.")
        print()

    # --- bottleneck labels ------------------------------------------------
    labels = collections.Counter(r["pred_bottleneck"] for r in rows)
    print("bottleneck attribution")
    print(f"  predicted label distribution: {dict(labels)}")
    if len(labels) == 1:
        print("  NOTE: every config received the same label, so the attribution")
        print("  carries no information on this sweep.")
    if args.ncu:
        print(f"  ncu export: {args.ncu}")
        print("  (comparison not implemented yet, see below)")
        failures.append("ncu comparison requested but not implemented")
    else:
        print("  NOT VALIDATED. The Phase 4 gate requires bottleneck-label accuracy")
        print("  cross-checked against Nsight Compute on at least 10 configs, using")
        print("  the mapping pre-registered in data/sweep/ncu-label-mapping.md.")
        failures.append(
            "bottleneck-label accuracy not validated against ncu. The gate cannot "
            "pass without it. Run ncu and pass --ncu."
        )
    print()

    # --- verdict -----------------------------------------------------------
    print("=" * 74)
    if failures:
        print("PHASE 4 GATE: FAIL")
        for f in failures:
            print(f"  - {f}")
        print()
        print("Report the honest number. A well-explained 18% beats a gamed 10%,")
        print("and narrowing the sweep to reach a threshold is anti-pattern 8.")
        return 1
    print("PHASE 4 GATE: PASS")
    print(f"  MAPE {mape:.2f}%, mean Spearman "
          f"{(sum(valid_rho)/len(valid_rho)) if valid_rho else float('nan'):.3f}, "
          f"worst regret@1 {max(worst_regrets)*100:.2f}%")
    print()
    print("Next: human-review checkpoint 3 in PLAN.md. The 2 to 3 case studies")
    print("explaining a surprising measured result are deliberately not part of")
    print("this machine gate, because judging what is surprising is human work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
