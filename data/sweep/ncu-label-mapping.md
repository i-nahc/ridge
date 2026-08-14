# Nsight Compute label mapping, pre-registered

**Registered 2026-08-14, before any `ncu` run on this project.** Nothing in this
file was chosen after seeing profiler output.

The Phase 4 gate requires Ridge's `bottleneck` label to be cross-checked against
hardware counters on at least 10 configs. That check is only worth anything if
the translation from counters to labels is fixed in advance. Otherwise, when a
config disagrees, the cheapest repair is to reinterpret the counter rather than
to accept that the model mislabelled it, and the validation becomes a formality.
This is the same reasoning as the pre-registered sweep in
`phase4-registered.csv`, PLAN.md anti-pattern 8.

The thresholds below are judgement calls. They are defensible but not derived,
and some of them are round numbers. That is exactly why they are written down
before the run instead of after it.

---

## 1. Counters collected

```
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active
l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_active
dram__throughput.avg.pct_of_peak_sustained_elapsed
sm__warps_active.avg.pct_of_peak_sustained_active
sm__cycles_active.avg.pct_of_peak_sustained_elapsed
```

Read as: how busy the tensor pipe is, how busy the shared-memory path is, how
busy DRAM is, how many warps are resident, and how much of the wall clock the
average SM was doing anything at all. The last one is the one that detects idle
SMs, which is what wave quantization actually is.

## 2. Decision procedure

Applied top to bottom, first match wins. **The order is part of the
registration.** It goes machine-level first, then SM-internal, because an idle
SM cannot be diagnosed from a pipe utilisation measured only over active cycles.

| # | condition | label |
|---|---|---|
| 1 | `dram__throughput` >= 70% | `HBM` |
| 2 | `sm__cycles_active` < 80% | `WAVES` |
| 3 | `sm__warps_active` < 30% | `OCC` |
| 4 | `l1tex__data_pipe_lsu_wavefronts_mem_shared` >= 60% | `SMEM` |
| 5 | `sm__pipe_tensor_op_hmma_cycles_active` >= 70% | `TENSOR_CORE` |
| 6 | otherwise | `ENVELOPE` |

**Rule 2 before rule 4 is deliberate.** `pct_of_peak_sustained_active`
normalises over cycles the SM was active, so a half-idle GPU can still show a
busy shared-memory pipe. Checking SM idleness first stops a wave-quantized
config from being read as SMEM-bound.

**`ENVELOPE` is the fallthrough, and that is a weakness.** Its signature is
genuine (nothing is saturated, yet the SMs are busy, which is what ramp-up and
drain look like) but it is defined by exclusion, so it absorbs any config the
five rules above miss. **Agreement on `ENVELOPE` is therefore the weakest
evidence in this table and must be reported as such**, not counted as equal
confirmation alongside `SMEM` or `WAVES`.

## 3. Configs to profile

Twelve, four per predicted label, chosen deterministically: sort each label group
by `(tag, BM, BN, BK, WM, WN, stages)` and take the first four distinct shapes.
Chosen before the run from the committed development pair.

| predicted | shape | tile | warp | stages | measured | predicted |
|---|---|---|---|---|---|---|
| `SMEM` | canonical | 64x64x64 | 32x32 | 4 | 103.5 | 128.0 |
| `SMEM` | many-waves | 64x64x64 | 32x32 | 4 | 103.1 | 140.1 |
| `SMEM` | moderate | 64x64x64 | 32x32 | 4 | 98.4 | 103.8 |
| `SMEM` | non-power-of-two | 64x64x64 | 32x32 | 4 | 104.1 | 127.3 |
| `WAVES` | moderate | 128x128x32 | 64x64 | 3 | 114.2 | 101.2 |
| `WAVES` | small-m | 128x128x32 | 64x64 | 3 | 119.0 | 110.7 |
| `WAVES` | small-n | 128x128x32 | 64x64 | 3 | 118.7 | 110.7 |
| `WAVES` | sub-wave | 128x128x32 | 32x64 | 4 | 64.2 | 37.1 |
| `ENVELOPE` | short-k | 64x64x64 | 32x32 | 4 | 85.1 | 58.2 |
| `ENVELOPE` | shorter-k | 64x64x64 | 32x32 | 4 | 62.1 | 35.8 |
| `ENVELOPE` | sub-wave | 64x128x64 | 32x64 | 4 | 42.6 | 59.3 |
| `ENVELOPE` | very-short-k | 64x64x64 | 32x32 | 4 | 51.2 | 20.3 |

## 4. Pass criterion

**At least 9 of 12 configs must agree** between Ridge's label and the label the
table in section 2 produces from counters.

Registered before the run. If it lands at 8, that is a fail and it gets reported
as a fail. Moving this number after seeing the result is anti-pattern 8.

## 5. Three of six labels cannot be validated at all

The model never predicts `HBM`, `OCC` or `TENSOR_CORE` anywhere in the 336-row
sweep. The distribution is `SMEM` 184, `WAVES` 79, `ENVELOPE` 73.

So this exercise validates the three labels the model actually emits and says
nothing about the other three. Rules 1, 3 and 5 in section 2 exist to catch a
*disagreement*, where Ridge says `SMEM` and the counters say `HBM`. They are not
evidence that Ridge would label an HBM-bound kernel correctly, because the sweep
contains no config Ridge believes is HBM-bound.

This is a limitation of the registered sweep, not of the profiler, and
`docs/RESULTS.md` must state it rather than reporting "labels validated".

## 6. Running it

Each `gemmMmaKernel` instantiation has a distinct mangled name carrying its
template arguments, so a single variant can be isolated without a new binary.
`-c 1` profiles only the first matching launch, which matters because
`bench/measure.cu` launches each config thousands of times for timing and
profiling all of them would be unusable.

```bash
NCU_METRICS=sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__cycles_active.avg.pct_of_peak_sustained_elapsed

ncu --csv --metrics "$NCU_METRICS" --kernel-name-base mangled \
    -k "regex:gemmMmaKernel" -c 1 \
    ./build/measure data/sweep/ncu-shapes.csv /tmp/ncu-throwaway.csv
```

`-c 1` is global to the run, not per kernel, so a single invocation captures one
launch total. Isolating a specific variant means either narrowing the regex with
that instantiation's template arguments (the mangled name carries `BM`, `BN`,
`BK`, `WM`, `WN`, stages and group) or running once per config. Confirm the
mangled-name shape on the box with `--kernel-name-base mangled` and a dry listing
before committing to twelve invocations.

`ncu-shapes.csv` is a cut-down shape list holding only the shapes named in
section 3. It is an input filter for profiling convenience and **must never be
used to produce an accuracy number**, which comes from `phase4-registered.csv`
only.

Counters are per-launch and clocks must be locked for the run, PLAN.md Finding 9.
