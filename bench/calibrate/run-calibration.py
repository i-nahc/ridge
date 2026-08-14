#!/usr/bin/env python3
"""Runs the calibration microbenchmarks and writes data/hardware/<gpu>.json.

This is the Phase 3 gate check. With --check it exits nonzero if any measured
constant falls outside its sanity band, and it refuses to write the json in that
case, so a bad constant cannot reach the model by default.

What the bands are, and what they are not. The constants themselves come from
our own cal-*.cu microbenchmarks. The bands are independent cross-checks against
public NVIDIA datasheet figures, used to catch a broken microbenchmark rather
than to supply a value. No published paper is load-bearing here, see PLAN.md
anti-pattern 7.

The bands are deliberately uneven. PLAN.md Finding 7 says the memory constants
are expected to land well below their theoretical figures, because a shared
memory number like "32 banks x 4 bytes" is a bank capacity rather than an
achievable load/store rate, while the tensor core peak tracks its datasheet
closely. So the compute band is tight and the memory bands are wide, and a
shared memory reading far under theoretical is an expected result rather than a
failure.

Specs are selected from the detected GPU, not hardcoded. PLAN.md Finding 10
records why: the project carried the A100 80GB HBM figure while measuring on a
40GB card, and a hardcoded band would have rejected a correct measurement.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

# Public datasheet figures, per card. Cross-checks only, never sources of a
# calibrated value.
SPECS = {
    "A100-SXM4-40GB": dict(peak_tflops=312.0, hbm_bytes_per_sec=1.555e12, sms=108),
    "A100-PCIE-40GB": dict(peak_tflops=312.0, hbm_bytes_per_sec=1.555e12, sms=108),
    "A100-SXM4-80GB": dict(peak_tflops=312.0, hbm_bytes_per_sec=2.039e12, sms=108),
    "A100-PCIE-80GB": dict(peak_tflops=312.0, hbm_bytes_per_sec=1.935e12, sms=108),
}

# Theoretical shared memory rate, 32 banks times 4 bytes. A capacity, not a rate.
SMEM_THEORETICAL_BYTES_PER_CYCLE = 128.0

BENCHMARKS = ["cal-mma", "cal-smem-bw", "cal-hbm-bw", "cal-latency"]


def detect_gpu(text):
    """Pulls the GPU name out of a benchmark's banner line."""
    m = re.search(r"GPU:\s*(?:NVIDIA\s+)?([A-Za-z0-9\-]+)", text)
    return m.group(1) if m else None


def run_benchmark(binary_dir, name):
    path = pathlib.Path(binary_dir) / name
    if not path.exists():
        sys.exit(
            f"{path} not found.\n"
            f"Build it first:\n"
            f"  nvcc -arch=sm_80 -O3 -std=c++17 -Ibench "
            f"bench/calibrate/{name}.cu -o {path}"
        )
    print(f"--- {name} " + "-" * (60 - len(name)))
    proc = subprocess.run([str(path)], capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    results = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"CAL_RESULT\s+(\S+)\s+(\S+)", line)
        if m:
            results[m.group(1)] = float(m.group(2))
    return results, proc.stdout, proc.returncode


def check(label, value, low, high, unit, note=""):
    ok = low <= value <= high
    status = "ok" if ok else "OUT OF BAND"
    print(
        f"  {label:<22} {value:>12.4g} {unit:<14} "
        f"band [{low:.4g}, {high:.4g}]  {status}"
    )
    if note:
        print(f"  {'':22} {note}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary-dir", default="build")
    ap.add_argument("--out", default=None, help="defaults to data/hardware/<gpu>.json")
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if any constant is out of band, and do not write the json",
    )
    args = ap.parse_args()

    all_results = {}
    banner = ""
    any_benchmark_failed = False
    for name in BENCHMARKS:
        res, out, rc = run_benchmark(args.binary_dir, name)
        all_results.update(res)
        banner += out
        if rc != 0:
            print(f"  {name} exited {rc}, its constants are unstable")
            any_benchmark_failed = True
        print()

    gpu = detect_gpu(banner)
    if gpu is None:
        sys.exit("could not detect the GPU from the benchmark output")
    if gpu not in SPECS:
        sys.exit(
            f"no datasheet entry for {gpu}.\n"
            f"Add one to SPECS rather than reusing another card's numbers. "
            f"See PLAN.md Finding 10 for what happens otherwise."
        )
    spec = SPECS[gpu]
    print(f"detected {gpu}, cross-checking against its datasheet figures")
    print(f"  peak {spec['peak_tflops']:.0f} TFLOP/s, "
          f"HBM {spec['hbm_bytes_per_sec']/1e12:.3f} TB/s\n")

    required = [
        "mmaCyclesPerInst",
        "impliedPeakTflops",
        "smemBytesPerCycle",
        "hbmBytesPerSec",
        "warpsNeededToHide",
    ]
    missing = [k for k in required if k not in all_results]
    if missing:
        sys.exit(f"benchmarks did not report: {', '.join(missing)}")

    print("sanity bands")
    ok = True

    # Compute band is tight. Finding 7: sustained MMA issue rate tracks the
    # datasheet closely, so a large deviation means the microbenchmark is wrong
    # rather than the hardware being surprising.
    ok &= check(
        "impliedPeakTflops",
        all_results["impliedPeakTflops"],
        spec["peak_tflops"] * 0.85,
        spec["peak_tflops"] * 1.05,
        "TFLOP/s",
        "tight band, the datasheet is expected to be close here",
    )

    # Memory bands are wide and asymmetric. A shared memory figure far under the
    # theoretical 128 is the expected outcome, not a fault.
    ok &= check(
        "smemBytesPerCycle",
        all_results["smemBytesPerCycle"],
        SMEM_THEORETICAL_BYTES_PER_CYCLE * 0.25,
        SMEM_THEORETICAL_BYTES_PER_CYCLE * 1.10,
        "B/cycle/SM",
        "well under 128 is expected, see Finding 7",
    )

    ok &= check(
        "hbmBytesPerSec",
        all_results["hbmBytesPerSec"] / 1e12,
        spec["hbm_bytes_per_sec"] * 0.70 / 1e12,
        spec["hbm_bytes_per_sec"] * 1.02 / 1e12,
        "TB/s",
        "above spec means the buffer fit in L2 and the number is not HBM",
    )

    ok &= check(
        "warpsNeededToHide", all_results["warpsNeededToHide"], 1, 32, "warps/SM"
    )

    if any_benchmark_failed:
        ok = False
        print("\n  at least one benchmark reported unstable repeats")

    print()
    if not ok:
        print("FAIL: at least one constant is outside its band.")
        print("A constant outside its band means the microbenchmark is wrong, not")
        print("that the hardware is surprising. Fix the benchmark. Do not widen the")
        print("band to make it pass, and do not import a value from another GPU.")
        print("See PLAN.md anti-patterns 7 and 8.")
        return 1

    if args.check:
        print("PASS: all constants inside their bands. Re-run without --check to write.")
        return 0

    out_path = pathlib.Path(args.out or f"data/hardware/{gpu.lower()}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": gpu,
        "note": (
            "measured by bench/calibrate/run-calibration.py. Constants come from "
            "our own microbenchmarks, bands cross-check public datasheets. Clocks "
            "must have been locked during the run, see PLAN.md Finding 9."
        ),
        "numSMs": spec["sms"],
        "clockHz": 1.41e9,
        "mmaCyclesPerInst": all_results["mmaCyclesPerInst"],
        "smemBytesPerCycle": all_results["smemBytesPerCycle"],
        "hbmBytesPerSec": all_results["hbmBytesPerSec"],
        "regsPerSM": 65536,
        "smemBytesPerSM": 167936,
        "maxWarpsPerSM": 64,
        "maxCtasPerSM": 32,
        "warpsNeededToHide": all_results["warpsNeededToHide"],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"PASS: wrote {out_path}")
    print("\nNext: human-review checkpoint 2 in PLAN.md reviews these constants")
    print("before any validation runs against them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
