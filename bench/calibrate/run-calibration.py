#!/usr/bin/env python3
"""Runs the calibration microbenchmarks and writes data/hardware/<gpu>.json.

With --check it exits nonzero if any measured constant falls outside its sanity
band and refuses to write the json, so a bad constant cannot reach the model.

The constants come from the cal-*.cu microbenchmarks. The bands are independent
cross-checks against public datasheet figures, used to catch a broken benchmark
rather than to supply a value.

The bands are deliberately uneven. A shared memory figure like "32 banks x 4
bytes" is a capacity rather than an achievable rate, so the memory constants are
expected to land well below theoretical, while the tensor core peak tracks its
datasheet closely. Hence a tight compute band and wide memory bands.

Specs are selected from the detected GPU rather than hardcoded, because a
hardcoded band once carried the A100 80GB HBM figure while measuring a 40GB card
and would have rejected a correct measurement.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

# Public datasheet figures per card. Cross-checks only, never sources of a value.
SPECS = {
    "A100-SXM4-40GB": dict(peak_tflops=312.0, hbm_bytes_per_sec=1.555e12, sms=108),
    "A100-PCIE-40GB": dict(peak_tflops=312.0, hbm_bytes_per_sec=1.555e12, sms=108),
    "A100-SXM4-80GB": dict(peak_tflops=312.0, hbm_bytes_per_sec=2.039e12, sms=108),
    "A100-PCIE-80GB": dict(peak_tflops=312.0, hbm_bytes_per_sec=1.935e12, sms=108),
}

# 32 banks times 4 bytes. A capacity, not a rate.
SMEM_THEORETICAL_BYTES_PER_CYCLE = 128.0

BENCHMARKS = ["cal-mma", "cal-smem-bw", "cal-hbm-bw", "cal-latency"]


def detect_gpu(text):
    m = re.search(r"GPU:\s*(?:NVIDIA\s+)?([A-Za-z0-9\-]+)", text)
    return m.group(1) if m else None


def run_benchmark(binary_dir, name):
    path = pathlib.Path(binary_dir) / name
    if not path.exists():
        sys.exit(f"{path} not found. Build it with:\n"
                 f"  nvcc -arch=sm_80 -O3 -std=c++17 -Ibench bench/calibrate/{name}.cu -o {path}")
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
    print(f"  {label:<22} {value:>12.4g} {unit:<14} band [{low:.4g}, {high:.4g}]  {'ok' if ok else 'OUT OF BAND'}")
    if note:
        print(f"  {'':22} {note}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary-dir", default="build")
    ap.add_argument("--out", default=None, help="defaults to data/hardware/<gpu>.json")
    ap.add_argument("--check", action="store_true",
                    help="exit nonzero if any constant is out of band, and do not write the json")
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
        sys.exit(f"no datasheet entry for {gpu}. Add one to SPECS rather than reusing another card's numbers.")
    spec = SPECS[gpu]
    print(f"detected {gpu}, cross-checking against its datasheet figures")
    print(f"  peak {spec['peak_tflops']:.0f} TFLOP/s, HBM {spec['hbm_bytes_per_sec']/1e12:.3f} TB/s\n")

    required = ["mmaCyclesPerInst", "impliedPeakTflops", "smemBytesPerCycle", "hbmBytesPerSec", "warpsNeededToHide"]
    missing = [k for k in required if k not in all_results]
    if missing:
        sys.exit(f"benchmarks did not report: {', '.join(missing)}")

    print("sanity bands")
    ok = True

    # Tight, because sustained MMA issue rate tracks the datasheet closely and a
    # large deviation means the benchmark is wrong rather than the hardware
    # surprising.
    ok &= check("impliedPeakTflops", all_results["impliedPeakTflops"],
                spec["peak_tflops"] * 0.85, spec["peak_tflops"] * 1.05, "TFLOP/s",
                "tight band, the datasheet should be close here")

    # Wide and asymmetric. Far under the theoretical 128 is expected.
    ok &= check("smemBytesPerCycle", all_results["smemBytesPerCycle"],
                SMEM_THEORETICAL_BYTES_PER_CYCLE * 0.25, SMEM_THEORETICAL_BYTES_PER_CYCLE * 1.10,
                "B/cycle/SM", "well under 128 is expected")

    ok &= check("hbmBytesPerSec", all_results["hbmBytesPerSec"] / 1e12,
                spec["hbm_bytes_per_sec"] * 0.70 / 1e12, spec["hbm_bytes_per_sec"] * 1.02 / 1e12,
                "TB/s", "above spec means the buffer fit in L2 and this is not HBM")

    ok &= check("warpsNeededToHide", all_results["warpsNeededToHide"], 1, 32, "warps/SM")

    if any_benchmark_failed:
        ok = False
        print("\n  at least one benchmark reported unstable repeats")

    print()
    if not ok:
        print("at least one constant is outside its band, which means the microbenchmark")
        print("is wrong rather than the hardware being surprising. Fix the benchmark")
        print("rather than widening the band or importing a value from another GPU.")
        return 1

    if args.check:
        print("all constants inside their bands, re-run without --check to write")
        return 0

    out_path = pathlib.Path(args.out or f"data/hardware/{gpu.lower()}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": gpu,
        "note": "measured by bench/calibrate/run-calibration.py with clocks locked at 1410 MHz",
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
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
