#!/usr/bin/env bash
# Runs the twelve pre-registered Nsight Compute profiles for the bottleneck-label
# check. The configs, the counters, the counter-to-label decision table and the
# pass criterion are all fixed in advance in data/sweep/ncu-label-mapping.md. This script only collects the counters. It
# does not decide anything, and it must not grow a scoring step, because the
# scoring rule is the part that has to stay untouched after the data exists.
#
# Needs sudo: NVIDIA restricts performance counters to admin users by default
# (ERR_NVGPUCTRPERM). Run this as a normal user, it calls sudo itself.
#
# Each kernel variant is one template instantiation, so the demangled name
# carries <BM, BN, BK, WM, WN, stages, groupM> and a regex picks out exactly one.
# The shape is not in the name, so shape is selected by feeding measure a
# one-line sweep file instead.

set -u

OUT="${1:-data/measured/ncu-labels.csv}"
BUILD=build
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

command -v ncu >/dev/null || { echo "ncu not found. Try /opt/nvidia/nsight-compute/*/ncu"; exit 1; }
[ -x "$BUILD/measure" ] || { echo "$BUILD/measure missing. Run ./setup-box.sh first."; exit 1; }

METRICS=sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__cycles_active.avg.pct_of_peak_sustained_elapsed

# predicted_label | tag | M N K | BM BN BK WM WN stages groupM
# Order and membership come from ncu-label-mapping.md section 3. Do not edit
# this list to improve the result.
CONFIGS=(
  "SMEM|canonical|4096 4096 4096|64 64 64 32 32 4 8"
  "SMEM|many-waves|8192 8192 8192|64 64 64 32 32 4 8"
  "SMEM|moderate|2048 2048 2048|64 64 64 32 32 4 8"
  "SMEM|non-power-of-two|6144 6144 4096|64 64 64 32 32 4 8"
  "WAVES|moderate|2048 2048 2048|128 128 32 64 64 3 8"
  "WAVES|small-m|1024 4096 4096|128 128 32 64 64 3 8"
  "WAVES|small-n|4096 1024 4096|128 128 32 64 64 3 8"
  "WAVES|sub-wave|1024 1024 1024|128 128 32 32 64 4 8"
  "ENVELOPE|short-k|4096 4096 512|64 64 64 32 32 4 8"
  "ENVELOPE|shorter-k|4096 4096 256|64 64 64 32 32 4 8"
  "ENVELOPE|sub-wave|1024 1024 1024|64 128 64 32 64 4 8"
  "ENVELOPE|very-short-k|4096 4096 128|64 64 64 32 32 4 8"
)

# Itanium mangling puts each int template argument as "Li<n>E", so the seven
# knobs become ILi64ELi64ELi64ELi32ELi32ELi4ELi8EE. Matching the mangled name
# rather than the demangled one avoids the spaces in "<64, 64, 64>" and is
# unambiguous, and -k matches whatever --kernel-name-base selects.
mangled_fragment() {
    local out="gemmMmaKernelI"
    for n in $1; do out="${out}Li${n}E"; done
    printf '%sE' "$out"
}

echo "predicted_label,tag,M,N,K,variant,metric,value" > "$OUT"

i=0
for c in "${CONFIGS[@]}"; do
    i=$((i + 1))
    LABEL=${c%%|*}; rest=${c#*|}
    TAG=${rest%%|*};  rest=${rest#*|}
    DIMS=${rest%%|*}
    VARIANT=${rest#*|}
    set -- $DIMS

    FRAG=$(mangled_fragment "$VARIANT")
    printf '[%2d/12] %-9s %-18s %-22s ' "$i" "$LABEL" "$TAG" "$VARIANT"

    # measure sizes its device buffers from the largest shape in this file, but
    # its warmup and its canary are both hardcoded to 4096x4096x4096. Hand it a
    # single shape smaller than that and it reads off the end of its own
    # allocation, which shows up as an illegal memory access at measure.cu:146
    # and, worse, sometimes does not fault at all because the overrun lands in
    # the next allocation. So always append a 4096-cube sizing shape.
    #
    # The target shape goes FIRST so that -c 1 captures a launch from it rather
    # than from the sizing shape.
    {
        echo "$1,$2,$3,$TAG,ncu target"
        echo "4096,4096,4096,sizing,buffer sizing only, never profiled"
    } > "$WORK/shape.csv"

    # --replay-mode application is required, not a preference. The default
    # kernel replay snapshots and restores device memory so it can re-run the
    # same launch, which trips over cp.async traffic still in flight and kills
    # the process with an illegal memory access. Application replay re-runs the
    # whole process per pass instead. The kernel itself is fine.
    #
    # -c 1 is global to the run, so with the regex pinned to one instantiation
    # it captures the first launch of exactly the variant we want.
    # --clock-control none matters as much as the replay mode. ncu defaults to
    # locking clocks to base for reproducibility, which overrides the 1410 MHz
    # lock setup-box.sh applies and makes measure's cuBLAS canary drift about
    # 12% across the run. measure then correctly refuses the data. We pin clocks
    # ourselves, so the profiler must leave them alone.
    sudo "$(command -v ncu)" --csv --replay-mode application --clock-control none \
        --metrics "$METRICS" \
        --kernel-name-base mangled \
        -k "regex:$FRAG" -c 1 \
        "$BUILD/measure" "$WORK/shape.csv" "$WORK/throwaway.csv" \
        > "$WORK/raw.txt" 2>&1

    # Keep the raw output for every config, not only the ones that look wrong. A run
    # that captures nothing still looks like it succeeded from the exit code.
    cp "$WORK/raw.txt" "$OUT.$i.log"

    # ncu --csv rows end with Metric Name, Metric Unit, Metric Value. Matching
    # pct_of_peak is tighter than matching the kernel name, because the mangled
    # name also shows up in ncu's own status and error lines.
    BEFORE=$(wc -l < "$OUT")
    grep 'pct_of_peak' "$WORK/raw.txt" | grep -v '^==' \
      | awk -F'","' -v l="$LABEL" -v t="$TAG" -v m="$1" -v n="$2" -v k="$3" -v v="$VARIANT" \
        'NF > 3 {gsub(/"/,"",$(NF-2)); gsub(/"/,"",$NF);
          printf "%s,%s,%s,%s,%s,\"%s\",%s,%s\n", l,t,m,n,k,v,$(NF-2),$NF}' \
      >> "$OUT"
    AFTER=$(wc -l < "$OUT")

    GOT=$((AFTER - BEFORE))
    if [ "$GOT" -eq 0 ]; then
        echo "NO COUNTERS, see $OUT.$i.log"
    else
        echo "$GOT metrics"
    fi
done

echo
echo "wrote $OUT"
echo "score with the decision table in data/sweep/ncu-label-mapping.md, registered before this run"
