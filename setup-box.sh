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
say "4. Calibrated hardware model"
# ---------------------------------------------------------------------------
HW=$(ls data/hardware/*.json 2>/dev/null | grep -v -- '-placeholder' | head -1 || true)
CAL_HW="data/hardware/a100-sxm4-40gb.json"

if [ -f "$CAL_HW" ]; then
    printf '  found %s\n' "$CAL_HW"
    printf '  NOTE: these constants were measured on a *previous* box. Any\n'
    printf '        A100-SXM4-40GB should reproduce them within noise, but if you\n'
    printf '        want them measured on this card, re-run:\n'
    printf '          python3 bench/calibrate/run-calibration.py\n'
else
    printf '  NOT PRESENT. The model will fall back to placeholder constants,\n'
    printf '  which include the wrong cards HBM spec (PLAN.md Finding 10).\n'
    printf '  Run this before any validation:\n'
    printf '    python3 bench/calibrate/run-calibration.py\n'
fi

# ---------------------------------------------------------------------------
say "Ready"
# ---------------------------------------------------------------------------
cat <<'EOF'
  ./build/check-gemm                        Phase 2 gate
  python3 bench/calibrate/run-calibration.py  Phase 3, writes data/hardware/<gpu>.json
  ./build/measure                           Phase 4 sweep, 4-8 min, use tmux
  python3 bench/validate.py                 Phase 4 gate

  Before destroying this box, commit and push anything under data/:
    git add data/ && git commit -m "measurements" && git push
  Instance storage is ephemeral and an unpushed calibration run is a paid-for
  result thrown away.
EOF
