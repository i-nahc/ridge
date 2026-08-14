#!/usr/bin/env bash
# setup-box.sh: bring a freshly rented GPU box to a state where Ridge can run.
#
# Rented boxes are destroyed to stop the meter, so this gets run often and from
# a clean clone every time. Everything here is idempotent and safe to re-run.
#
# It REFUSES to continue on two conditions rather than warning about them,
# because both silently corrupt every number the project produces and neither is
# visible in the output afterwards:
#
#   1. A MIG-partitioned GPU. The model hardcodes numSMs=108. A MIG slice gives
#      a fraction of that, and every prediction would be wrong with no error.
#   2. Unlocked SM clocks. An idle A100 sits at 210 MHz against a 1410 MHz boost
#      clock, and clock locks do not survive a reboot or a new instance. See
#      PLAN.md Finding 9: this artifact was worth about 20% and it corrupted
#      every throughput measurement in the project before it was found.
#
# Usage:  bash setup-box.sh

set -euo pipefail

BUILD=build
ARCH=sm_80

say()  { printf '\n=== %s\n' "$1"; }
fail() { printf '\nFAIL: %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------------------
say "1. GPU identity"
# ---------------------------------------------------------------------------
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found. Is this a GPU box?"
nvidia-smi -L

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_COUNT=$(nvidia-smi -L | wc -l)
printf '  detected: %s (%s visible)\n' "$GPU_NAME" "$GPU_COUNT"

if nvidia-smi -L | grep -qi 'MIG'; then
    fail "MIG partitioning detected.
  The model hardcodes numSMs=108 for a full A100. A MIG slice has a fraction of
  that and every prediction would be silently wrong. Use a full GPU."
fi

case "$GPU_NAME" in
    *A100*) ;;
    *) printf '  WARNING: expected an A100. Calibration constants and the sm_80\n'
       printf '           target are for A100. Continuing, but nothing downstream\n'
       printf '           is valid for a different card.\n' ;;
esac

# ---------------------------------------------------------------------------
say "2. Lock SM clocks"
# ---------------------------------------------------------------------------
BEFORE=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1)
printf '  before: %s\n' "$BEFORE"

if sudo nvidia-smi -lgc 1410,1410 >/dev/null 2>&1; then
    AFTER=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1)
    printf '  after:  %s   locked\n' "$AFTER"
else
    fail "could not lock clocks (sudo nvidia-smi -lgc 1410,1410).
  Without a lock the SM clock floats between 210 and 1410 MHz and whatever is
  timed first is measured during the ramp. See PLAN.md Finding 9. If this box
  does not grant the permission, it is not usable for Phase 3 to 5."
fi

# ---------------------------------------------------------------------------
say "3. Build"
# ---------------------------------------------------------------------------
mkdir -p "$BUILD" data/measured

command -v nvcc >/dev/null || fail "nvcc not found. Use the Lambda Stack image or the container."
command -v g++  >/dev/null || fail "g++ not found."

printf '  ridge-predict (plain C++, no CUDA)... '
g++ -std=c++17 -O2 -Iinclude \
    src/model.cpp src/hardware.cpp src/mma.cpp tools/ridge-predict.cpp \
    -o "$BUILD/ridge-predict"
printf 'ok\n'

printf '  check-gemm (Phase 2 gate)... '
nvcc -arch=$ARCH -O3 -std=c++17 -Ibench \
     bench/kernels/gemm-mma.cu bench/check-gemm.cu -lcublas -o "$BUILD/check-gemm"
printf 'ok\n'

printf '  measure (Phase 4 sweep)... '
nvcc -arch=$ARCH -O3 -std=c++17 -Ibench \
     bench/kernels/gemm-mma.cu bench/measure.cu -lcublas -o "$BUILD/measure"
printf 'ok\n'

for c in cal-mma cal-smem-bw cal-hbm-bw cal-latency; do
    printf '  %s... ' "$c"
    nvcc -arch=$ARCH -O3 -std=c++17 -Ibench "bench/calibrate/$c.cu" -o "$BUILD/$c"
    printf 'ok\n'
done

# A spilling kernel is not a fair ground truth, and this is cheap to check.
printf '  register spill check... '
SPILLS=$(nvcc -arch=$ARCH -O3 -std=c++17 -Ibench -Xptxas -v \
         -c bench/kernels/gemm-mma.cu -o /dev/null 2>&1 \
         | grep -c 'spill stores, [1-9]' || true)
if [ "$SPILLS" != "0" ]; then
    fail "$SPILLS kernel variants spill registers. Phase 2 criterion 2 fails."
fi
printf 'none\n'

# ---------------------------------------------------------------------------
say "4. Calibrate this card"
# ---------------------------------------------------------------------------
# Calibration is run every session, on every box, deliberately.
#
# SPEC section 9 requires calibrating and validating on the same physical card.
# Carrying a json forward from a previous box breaks that: the constants would
# describe one piece of silicon and the sweep another, and nothing downstream
# would flag it. Recalibrating costs about two minutes against a sweep that costs
# five, so the efficiency argument for reusing an old file is worth very little
# and the correctness argument against it is absolute.
#
# It also removes a whole failure class. A stale json is invisible once written,
# and "which box did these constants come from" is not a question anyone will
# think to ask three weeks later.
#
# Running it here rather than leaving it as a step means it cannot be forgotten
# on the fourth box of the day, which is exactly when it would be.
if [ -f data/hardware/a100-sxm4-40gb.json ]; then
    printf '  a previous json exists and will be overwritten by this run\n'
fi

python3 bench/calibrate/run-calibration.py || fail "calibration failed.
  A constant outside its sanity band means the microbenchmark is wrong, not that
  the hardware is surprising. Do not widen the band. See PLAN.md anti-patterns
  7 and 8."

# ---------------------------------------------------------------------------
say "Ready"
# ---------------------------------------------------------------------------
cat <<'EOF'
  ./build/check-gemm          Phase 2 gate
  ./build/measure             Phase 4 sweep, 4-8 min, use tmux
  python3 bench/validate.py   Phase 4 gate

  GETTING RESULTS OFF THIS BOX

  Do not push from here, and do not put a git credential or an SSH key with
  write access on a rented machine. It is shared infrastructure you do not
  control, the disk is not yours, and the box is destroyed later by someone
  else. The clone above is public over HTTPS and needs no credential, so this
  box only ever pulls.

  Pull the results down from your own machine instead:

    scp ubuntu@<this-box-ip>:~/ridge/data/measured/a100.csv       data/measured/
    scp ubuntu@<this-box-ip>:'~/ridge/data/hardware/*.json'        data/hardware/

  Then commit from there. Instance storage is ephemeral, so anything under
  data/ that has not been copied off is a paid-for result thrown away.
EOF
