"""Ridge as a config pruner for Triton's GEMM autotuner.

The comparison this makes, and why it is the honest one:

`triton.autotune` benchmarks **every** config in its list on the first call for a
new shape, then caches the winner. So stock Triton is exhaustive search, and it
always finds the best config in the list by construction. That makes it a real
opponent with a real selection, not a strawman: the quality bar is "the config
Triton itself would have chosen".

So there is exactly one question. If Ridge ranks the configs and we benchmark
only the top k, do we still land on Triton's pick, and how much tuning time does
that save?

Reported:
  - tuning cost, in configs benchmarked and in wall-clock seconds
  - whether the pruned search found the same config stock Triton did
  - if not, how much slower the pruned pick is, in percent

Every config is timed once up front. Those timings are the ground truth, and they
are what stock autotune would have measured anyway, so no config is timed twice
and the comparison uses identical numbers for both arms.

Run:  python3 bench/triton-compare.py
Needs: triton, torch, an NVIDIA GPU, and build/ridge-predict already built.
"""

import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, "tools")
from ridge_triton import estimated_seconds, infer_warp_tile  # noqa: E402

TOP_K = 8


# The config list from Triton's own matmul tutorial. Not curated by us, which is
# the point: a curated list would let us choose a space Ridge happens to be good
# at. If you change this, say so in the writeup.
CONFIGS = [
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=3, num_warps=8),
    dict(BLOCK_SIZE_M=64,  BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_stages=4, num_warps=4),
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_stages=4, num_warps=4),
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64,  BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_stages=4, num_warps=4),
    dict(BLOCK_SIZE_M=64,  BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_stages=4, num_warps=4),
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=32,  BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_stages=4, num_warps=4),
    dict(BLOCK_SIZE_M=64,  BLOCK_SIZE_N=32,  BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_stages=5, num_warps=2),
    dict(BLOCK_SIZE_M=32,  BLOCK_SIZE_N=64,  BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_stages=5, num_warps=2),
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_stages=3, num_warps=8),
    dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_stages=3, num_warps=8),
    dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=64,  BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_stages=4, num_warps=4),
    dict(BLOCK_SIZE_M=64,  BLOCK_SIZE_N=256, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_stages=4, num_warps=4),
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_stages=4, num_warps=4),
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64,  BLOCK_SIZE_K=64,  GROUP_SIZE_M=8, num_stages=4, num_warps=4),
    dict(BLOCK_SIZE_M=64,  BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,  GROUP_SIZE_M=8, num_stages=4, num_warps=4),
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=32,  BLOCK_SIZE_K=64,  GROUP_SIZE_M=8, num_stages=4, num_warps=4),
]

# Same problem shapes as the Ridge validation sweep, so the two results can be
# read together.
SHAPES = [
    (4096, 4096, 4096, "canonical"),
    (8192, 8192, 8192, "many-waves"),
    (6144, 6144, 4096, "non-power-of-two"),
    (2048, 2048, 2048, "moderate"),
    (1024, 1024, 1024, "sub-wave"),
    (4096, 4096, 512,  "short-k"),
    (4096, 4096, 128,  "very-short-k"),
    (1024, 4096, 4096, "small-m"),
    (4096, 1024, 4096, "small-n"),
    (8192, 2048, 4096, "wide-m"),
]


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def time_config(a, b, c, M, N, K, cfg):
    """Milliseconds for one config, or None if it fails to run."""
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)
    try:
        def run():
            matmul_kernel[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"], BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
                BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"], GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
                num_stages=cfg["num_stages"], num_warps=cfg["num_warps"])
        run()
        torch.cuda.synchronize()
        return triton.testing.do_bench(run, warmup=25, rep=100)
    except Exception:
        return None


def main():
    torch.manual_seed(0)
    print("Ridge as a pruner for Triton's GEMM autotuner")
    print("stock triton.autotune benchmarks all %d configs and always finds the" % len(CONFIGS))
    print("best one, so it is the quality bar. Ridge benchmarks only the top %d.\n" % TOP_K)
    print("%-18s %7s %8s %9s %9s %9s  %s" % (
        "shape", "cfgs", "stock s", "ridge s", "speedup", "regret", "same pick"))

    tot_stock = tot_ridge = 0.0
    regrets = []
    same = 0
    for M, N, K, tag in SHAPES:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        c = torch.empty((M, N), device="cuda", dtype=torch.float16)

        # Ground truth: time every config once. This is exactly the work stock
        # autotune does, so its tuning cost is the sum of these.
        timings = {}
        t0 = time.perf_counter()
        for i, cfg in enumerate(CONFIGS):
            ms = time_config(a, b, c, M, N, K, cfg)
            if ms is not None:
                timings[i] = ms
        stock_wall = time.perf_counter() - t0
        if not timings:
            print("%-18s   all configs failed" % tag)
            continue

        stock_pick = min(timings, key=timings.get)

        # Ridge: score every config, benchmark only the top k.
        t0 = time.perf_counter()
        scored = []
        for i, cfg in enumerate(CONFIGS):
            s = estimated_seconds(M, N, K, cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"],
                                  cfg["BLOCK_SIZE_K"], cfg["num_warps"],
                                  cfg["num_stages"])
            scored.append((s, i))
        scored.sort()
        model_wall = time.perf_counter() - t0

        shortlist = [i for _, i in scored[:TOP_K] if i in timings]
        if not shortlist:
            print("%-18s   ridge shortlist all failed" % tag)
            continue
        ridge_pick = min(shortlist, key=timings.get)

        # Ridge's tuning cost: the model, plus benchmarking only the shortlist.
        # Estimated from measured per-config time so both arms use one timing set.
        per_cfg = stock_wall / max(len(timings), 1)
        ridge_wall = model_wall + per_cfg * len(shortlist)

        regret = (timings[ridge_pick] - timings[stock_pick]) / timings[stock_pick] * 100
        regrets.append(regret)
        hit = ridge_pick == stock_pick
        same += hit
        tot_stock += stock_wall
        tot_ridge += ridge_wall
        print("%-18s %7d %8.2f %9.2f %8.2fx %8.2f%%  %s" % (
            tag, len(timings), stock_wall, ridge_wall,
            stock_wall / ridge_wall, regret, "yes" if hit else "no"))

    n = len(regrets)
    if n:
        print()
        print("shapes                     %d" % n)
        print("same config as stock       %d/%d" % (same, n))
        print("mean regret vs stock pick  %.2f%%" % (sum(regrets) / n))
        print("max regret vs stock pick   %.2f%%" % max(regrets))
        print("total tuning time          stock %.1fs -> ridge %.1fs (%.2fx)" % (
            tot_stock, tot_ridge, tot_stock / tot_ridge))


if __name__ == "__main__":
    main()
