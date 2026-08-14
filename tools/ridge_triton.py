"""Ridge as a Triton autotuner performance model.

Triton's `triton.autotune` accepts `prune_configs_by={'perf_model': fn, 'top_k': n}`.
It calls `fn` once per candidate config, sorts ascending on the returned value,
and benchmarks only the best `n`. So `fn` must return an estimated **runtime**,
lower being better. That is the hook Ridge plugs into.

This module never reimplements the model. It shells out to `build/ridge-predict`,
the same binary `bench/validate.py` uses, so there is exactly one implementation
of the math and no chance of the Python drifting from the C++.

What is approximate here, stated up front because it bounds how much the numbers
mean:

1. **Warp tile is inferred, not given.** Ridge needs `warpM`/`warpN`. Triton
   exposes only `num_warps` and lets the compiler choose the decomposition. We
   pick the factorisation of `num_warps` that makes the per-warp tile closest to
   square, which is the usual choice and matches what our own kernel does, but it
   is an assumption about generated code we do not control.

2. **Registers come from compilation when available.** Ridge's occupancy term
   wants `regsPerThread`. Triton exposes `n_regs` on a compiled kernel, so we use
   it when the caller supplies one. Otherwise we pass 0, which tells Ridge to
   skip the register limit, and occupancy is then bounded only by shared memory
   and warps. That makes the model optimistic for register-heavy configs.

3. **Ridge was derived for our kernel's structure.** cp.async multi-stage,
   ldmatrix, mma.sync. Triton generates broadly this shape for Ampere matmul, but
   it is not the same code, and nothing here verifies that it is.
"""

import functools
import subprocess
from pathlib import Path

PREDICT_BIN = Path("build/ridge-predict")
HW_JSON = Path("data/hardware/a100-fresh.json")


def infer_warp_tile(block_m, block_n, num_warps):
    """Split num_warps into a 2D grid over the CTA tile, closest to square.

    Returns (warpM, warpN), or None if no split gives a legal MMA tile. Legal
    means at least 16 rows and 8 columns per warp, the m16n8k16 shape.
    """
    best = None
    for wm in range(1, num_warps + 1):
        if num_warps % wm:
            continue
        wn = num_warps // wm
        if block_m % wm or block_n % wn:
            continue
        tile_m, tile_n = block_m // wm, block_n // wn
        if tile_m < 16 or tile_n < 8:
            continue
        aspect = max(tile_m, tile_n) / min(tile_m, tile_n)
        if best is None or aspect < best[0]:
            best = (aspect, tile_m, tile_n)
    return (best[1], best[2]) if best else None


@functools.lru_cache(maxsize=4096)
def _predict(m, n, k, bm, bn, bk, wm, wn, stages, regs):
    """One ridge-predict invocation. Cached, because Triton asks repeatedly."""
    cmd = [
        str(PREDICT_BIN), "--m", str(m), "--n", str(n), "--k", str(k),
        "--bm", str(bm), "--bn", str(bn), "--bk", str(bk),
        "--warpm", str(wm), "--warpn", str(wn),
        "--stages", str(stages), "--regs", str(regs),
    ]
    if HW_JSON.exists():
        cmd += ["--hw", str(HW_JSON)]
    else:
        cmd += ["--gpu", "a100"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "predicted:" in line:
            # "  predicted:     120.2 TFLOP/s   bottleneck: SMEM"
            return float(line.split("predicted:")[1].split()[0])
    return 0.0


def estimated_seconds(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                      num_warps, num_stages, regs_per_thread=0):
    """Predicted runtime in seconds. Lower is better, which is what Triton wants.

    Returns +inf for configs Ridge cannot model or that do not fit an SM, so they
    sort last and are never benchmarked.
    """
    warp = infer_warp_tile(BLOCK_M, BLOCK_N, num_warps)
    if warp is None:
        return float("inf")
    tflops = _predict(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                      warp[0], warp[1], num_stages, regs_per_thread)
    if tflops <= 0.0:
        return float("inf")
    return (2.0 * M * N * K) / (tflops * 1e12)


def make_perf_model(m_key="M", n_key="N", k_key="K", regs_by_config=None):
    """Build a perf_model callable for triton.autotune.

    Triton invokes it as perf_model(**named_args, **config.kwargs,
    num_warps=..., num_stages=...), so it must tolerate arbitrary extra keys.

    regs_by_config, if given, maps a config key tuple to a measured n_regs. Build
    it by compiling the configs first, which is far cheaper than benchmarking
    them, and is what makes the occupancy term usable here.
    """
    def perf_model(**kwargs):
        try:
            M = int(kwargs[m_key]); N = int(kwargs[n_key]); K = int(kwargs[k_key])
            bm = int(kwargs["BLOCK_SIZE_M"]); bn = int(kwargs["BLOCK_SIZE_N"])
            bk = int(kwargs["BLOCK_SIZE_K"])
            nw = int(kwargs["num_warps"]); ns = int(kwargs["num_stages"])
        except (KeyError, TypeError, ValueError):
            return float("inf")
        regs = 0
        if regs_by_config:
            regs = regs_by_config.get((bm, bn, bk, nw, ns), 0)
        return estimated_seconds(M, N, K, bm, bn, bk, nw, ns, regs)
    return perf_model
