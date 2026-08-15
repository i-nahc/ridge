"""Model-pruned empirical autotuner.

The model prunes, the hardware decides. For a given problem shape it ranks the
candidate tile configs by predicted throughput, keeps the top k, and then
*measures* those k and takes the fastest. The model never picks the config.

That split is deliberate and it is the whole design. Ridge's ranking is Spearman
about 0.65 with worst-case regret near 47%, so letting it choose directly gives a
weak answer. Pruning asks something much easier of it: not "which config is
fastest" but "is the fastest config somewhere in your top k". A model can be
mediocre at ordering and still excellent at that, and the empirical step then
recovers the exact optimum.

Predictions come from `build/ridge-predict --batch`, the same binary
`bench/validate.py` uses. The model is never reimplemented here.

Evaluation runs entirely against an existing measured sweep. Every config's real
throughput is already known, so "benchmark the top k" is simulated exactly by
looking up those k measurements, and the exhaustive best is known too. No GPU is
needed and no new measurement is taken.

Usage:
    python3 tools/ridge-autotune.py
    python3 tools/ridge-autotune.py --measured data/measured/a100-fresh.csv
"""

import argparse
import csv
import os
import subprocess
import statistics
import sys
import tempfile


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", default="data/measured/a100-fresh.csv")
    ap.add_argument("--hw", default="data/hardware/a100-fresh.json")
    ap.add_argument("--predict-bin", default="build/ridge-predict")
    ap.add_argument("--k", default="1,2,3,5,10,20",
                    help="shortlist sizes to evaluate, per shape")
    args = ap.parse_args()

    if not os.path.exists(args.measured):
        sys.exit("missing %s" % args.measured)

    with tempfile.TemporaryDirectory() as tmp:
        pred_path = os.path.join(tmp, "pred.csv")
        cmd = [args.predict_bin, "--batch", args.measured, "--out", pred_path]
        if os.path.exists(args.hw):
            cmd = [args.predict_bin, "--hw", args.hw,
                   "--batch", args.measured, "--out", pred_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit("ridge-predict failed:\n" + proc.stderr)
        pred = load_rows(pred_path)

    meas = load_rows(args.measured)
    if len(pred) != len(meas):
        sys.exit("prediction/measurement row count mismatch: %d vs %d"
                 % (len(pred), len(meas)))

    # Join positionally: --batch preserves input order.
    shapes = {}
    for m, p in zip(meas, pred):
        key = m["tag"]
        pt = float(p.get("pred_tflops") or p.get("predictedTFLOPS") or 0.0)
        shapes.setdefault(key, []).append((pt, float(m["tflops"])))

    ks = [int(x) for x in args.k.split(",")]
    total_configs = len(meas)
    per_shape = len(meas) // len(shapes)

    print("Model-pruned empirical autotuner")
    print("  measured sweep   %s" % args.measured)
    print("  %d configs over %d shapes (%d per shape)"
          % (total_configs, len(shapes), per_shape))
    print()
    print("  The model ranks, the hardware decides. Regret is measured against")
    print("  the exhaustive best, which is the fastest config actually measured")
    print("  for that shape.")
    print()
    print("  %5s %12s %12s %11s %11s %10s" % (
        "top_k", "benchmarks", "vs exhaustive", "mean regret",
        "max regret", "exact hits"))

    best_k = None
    for k in ks:
        if k > per_shape:
            continue
        regrets = []
        hits = 0
        for _tag, rows in shapes.items():
            best = max(r[1] for r in rows)
            shortlist = sorted(rows, key=lambda r: -r[0])[:k]
            picked = max(r[1] for r in shortlist)
            regrets.append((best - picked) / best * 100.0)
            hits += picked == best
        benches = k * len(shapes)
        mean_r = statistics.mean(regrets)
        max_r = max(regrets)
        print("  %5d %12d %11.1fx %10.2f%% %10.2f%% %6d/%-3d" % (
            k, benches, total_configs / benches, mean_r, max_r,
            hits, len(shapes)))
        if best_k is None and mean_r <= 3.0:
            best_k = (k, mean_r, max_r, benches, total_configs / benches, hits)

    if best_k:
        k, mean_r, max_r, benches, save, hits = best_k
        print()
        print("  Smallest k within 3%% of exhaustive: k=%d" % k)
        print("    %d benchmarks instead of %d, a %.1fx reduction"
              % (benches, total_configs, save))
        print("    mean regret %.2f%%, worst %.2f%%, exact optimum on %d/%d shapes"
              % (mean_r, max_r, hits, len(shapes)))
    else:
        print()
        print("  No k reached within 3% of exhaustive. Report that as the result:")
        print("  the model's ranking is not strong enough to prune this space.")

    print()
    print("  Per-shape detail at each k is deterministic from the sweep, so this")
    print("  analysis is reproducible offline with no GPU.")


if __name__ == "__main__":
    main()
