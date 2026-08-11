// gemm-mma.cuh is the ground truth GEMM kernel for Ridge.
//
// This is one parameterized tensor-core GEMM templated on exactly the knobs the
// Ridge model takes as input, so a measured config and a predicted config are
// the same object. It targets sm_80 and uses the real Ampere path: a multi stage
// cp.async pipeline from global to shared memory, ldmatrix to move operands into
// registers, and warp level mma.sync to accumulate.
//
// The kernel computes C = A * B where A is M by K row major, B is K by N row
// major, and C is M by N row major. Inputs are FP16 and accumulation is FP32,
// which is the v1 dtype in SPEC section 2.
//
// Quality matters here. PLAN.md gates Phase 2 on this kernel being correct
// against cuBLAS and reaching about 70% of it. A weak kernel makes every later
// accuracy number meaningless, so this is written as a real kernel rather than a
// teaching example.
//
// v1 constraints, deliberate and documented:
//   M, N, K must be divisible by BM, BN, BK. The measurement sweep uses aligned
//   shapes so predication is not needed yet.
//   The epilogue writes accumulators straight to global memory. Staging C
//   through shared memory would coalesce better and is a tuning item for once we
//   can measure.

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace ridgebench {

// The v1 MMA shape. This matches MmaShape in the model, see include/ridge/mma.h.
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;

// ---------------------------------------------------------------------------
// PTX wrappers
// ---------------------------------------------------------------------------

// Converts a generic shared memory pointer into the 32 bit address that the
// shared memory PTX instructions expect.
__device__ __forceinline__ uint32_t smemAddr(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Issues one 16 byte asynchronous copy from global to shared memory. The cg
// qualifier bypasses L1, which is what we want for streaming tile loads that
// are not reused out of L1.
__device__ __forceinline__ void cpAsyncCg16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(dst), "l"(src));
}

__device__ __forceinline__ void cpAsyncCommit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Waits until at most N committed copy groups are still in flight.
template <int N>
__device__ __forceinline__ void cpAsyncWaitGroup() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Loads a 16 by 16 FP16 tile into the register layout that the A operand of
// mma.sync.m16n8k16 expects. Every lane in the warp supplies one row address.
__device__ __forceinline__ void ldmatrixX4(uint32_t addr, uint32_t (&d)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(addr));
}

// Loads a 16 by 8 FP16 tile for the B operand. B lives in shared memory with N
// contiguous, so the trans qualifier does the transpose for free on the way into
// registers and we avoid transposing during the global load, which cp.async
// could not do anyway because it only copies contiguous bytes.
__device__ __forceinline__ void ldmatrixX2Trans(uint32_t addr, uint32_t (&d)[2]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(addr));
}

// One warp level FP16 multiply accumulate into FP32 accumulators.
__device__ __forceinline__ void mmaM16N8K16(float (&c)[4],
                                            const uint32_t (&a)[4],
                                            const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------------------
// Shared memory swizzle
// ---------------------------------------------------------------------------

// Shared memory tiles are stored row major in 16 byte chunks of 8 halves. A
// plain row major layout makes the 8 rows that ldmatrix reads land in the same
// banks, so we permute the chunk index within each row by XOR with the row
// index. Padding the rows would be the other classic fix, but padding breaks the
// 16 byte alignment that cp.async requires, so XOR swizzling is the option that
// keeps both working.
//
// This is conflict free when a row holds 8 chunks, which means BK of 64 for the
// A tile and BN of 64 for the B tile. Narrower rows wrap the XOR and leave a two
// way conflict. That is a known cost of the small BK configs and the sweep will
// show it, which is exactly the kind of surprise the model should explain.
template <int RowHalves>
__device__ __forceinline__ int swizzledOffset(int row, int chunk) {
    constexpr int kChunksPerRow = RowHalves / 8;
    return (row * kChunksPerRow + (chunk ^ (row % kChunksPerRow))) * 8;
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

template <int BM, int BN, int BK, int WM, int WN, int Stages>
__global__ __launch_bounds__((BM / WM) * (BN / WN) * 32) void gemmMmaKernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    constexpr int kWarpsM = BM / WM;
    constexpr int kWarpsN = BN / WN;
    constexpr int kNumWarps = kWarpsM * kWarpsN;
    constexpr int kThreads = kNumWarps * 32;

    static_assert(Stages >= 2, "the pipeline needs at least two shared memory buffers");
    static_assert(BM % WM == 0 && BN % WN == 0, "the CTA tile must divide into warp tiles");
    static_assert(WM % kMmaM == 0, "the warp tile must divide into mma tiles along M");
    static_assert(WN % kMmaN == 0, "the warp tile must divide into mma tiles along N");
    static_assert(BK % kMmaK == 0, "BK must be a whole number of mma K steps");
    static_assert(BK % 8 == 0 && BN % 8 == 0, "tile rows must be a whole number of 16 byte chunks");

    // Per warp mma tile counts. These are the same quantities the model uses to
    // compute mmasPerWarpPerStep, see docs/MODEL.md section 2.
    constexpr int kTilesM = WM / kMmaM;
    constexpr int kTilesN = WN / kMmaN;
    constexpr int kTilesK = BK / kMmaK;

    constexpr int kChunksPerRowA = BK / 8;
    constexpr int kChunksPerRowB = BN / 8;
    constexpr int kChunksA = BM * kChunksPerRowA;
    constexpr int kChunksB = BK * kChunksPerRowB;

    constexpr int kSmemHalvesA = BM * BK;
    constexpr int kSmemHalvesB = BK * BN;

    extern __shared__ half smem[];
    half* smemA = smem;
    half* smemB = smem + Stages * kSmemHalvesA;

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warpM = warpId / kWarpsN;
    const int warpN = warpId % kWarpsN;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    float acc[kTilesM][kTilesN][4];
#pragma unroll
    for (int i = 0; i < kTilesM; ++i)
#pragma unroll
        for (int j = 0; j < kTilesN; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    const int numKTiles = K / BK;

    // Stages one global tile load into shared memory buffer `stage`.
    auto loadTile = [&](int kTile, int stage) {
        half* dstA = smemA + stage * kSmemHalvesA;
        half* dstB = smemB + stage * kSmemHalvesB;

#pragma unroll
        for (int i = threadIdx.x; i < kChunksA; i += kThreads) {
            const int row = i / kChunksPerRowA;
            const int chunk = i % kChunksPerRowA;
            const half* src = A + (blockRow + row) * K + kTile * BK + chunk * 8;
            cpAsyncCg16(smemAddr(dstA + swizzledOffset<BK>(row, chunk)), src);
        }
#pragma unroll
        for (int i = threadIdx.x; i < kChunksB; i += kThreads) {
            const int row = i / kChunksPerRowB;
            const int chunk = i % kChunksPerRowB;
            const half* src = B + (kTile * BK + row) * N + blockCol + chunk * 8;
            cpAsyncCg16(smemAddr(dstB + swizzledOffset<BN>(row, chunk)), src);
        }
        cpAsyncCommit();
    };

    // Prologue. Fill every buffer but the last so the first main loop iteration
    // already has work in flight behind it.
#pragma unroll
    for (int s = 0; s < Stages - 1; ++s) {
        loadTile(s, s);
    }

    uint32_t fragA[kTilesM][4];
    uint32_t fragB[kTilesN][2];

    for (int kTile = 0; kTile < numKTiles; ++kTile) {
        const int stage = kTile % Stages;

        // Wait until only the copies for the still outstanding stages remain.
        cpAsyncWaitGroup<Stages - 2>();
        __syncthreads();

        // Issue the loads for the tile Stages-1 ahead before doing this tile's
        // math, so the copy flies underneath the mma work instead of trailing
        // it. The buffer being refilled is the one the previous iteration read,
        // and the barrier above already established that every warp is done
        // with it.
        //
        // The empty commit in the tail matters. cp.async.wait_group counts the
        // most recent groups, so if we simply stopped committing near the end of
        // K the count would fall below the threshold and the wait would return
        // without the tile having landed. Committing an empty group keeps the
        // accounting uniform so the wait always means "the tile I am about to
        // read has arrived".
        const int nextTile = kTile + Stages - 1;
        if (nextTile < numKTiles) {
            loadTile(nextTile, nextTile % Stages);
        } else {
            cpAsyncCommit();
        }

        const half* curA = smemA + stage * kSmemHalvesA;
        const half* curB = smemB + stage * kSmemHalvesB;

#pragma unroll
        for (int kStep = 0; kStep < kTilesK; ++kStep) {
            // Each lane hands ldmatrix the address of one 8 by 8 sub tile row.
            // For the x4 form the warp covers four sub tiles, which together are
            // the 16 by 16 A fragment.
#pragma unroll
            for (int i = 0; i < kTilesM; ++i) {
                const int mi = laneId / 8;
                const int rowInTile = laneId % 8;
                const int row = warpM * WM + i * kMmaM + (mi % 2) * 8 + rowInTile;
                const int kOff = kStep * kMmaK + (mi / 2) * 8;
                ldmatrixX4(smemAddr(curA + swizzledOffset<BK>(row, kOff / 8)), fragA[i]);
            }
#pragma unroll
            for (int j = 0; j < kTilesN; ++j) {
                const int mi = (laneId % 16) / 8;
                const int rowInTile = laneId % 8;
                const int kRow = kStep * kMmaK + mi * 8 + rowInTile;
                const int nOff = warpN * WN + j * kMmaN;
                ldmatrixX2Trans(smemAddr(curB + swizzledOffset<BN>(kRow, nOff / 8)), fragB[j]);
            }
#pragma unroll
            for (int i = 0; i < kTilesM; ++i)
#pragma unroll
                for (int j = 0; j < kTilesN; ++j)
                    mmaM16N8K16(acc[i][j], fragA[i], fragB[j]);
        }
    }

    // Epilogue. The m16n8k16 accumulator layout puts c0 and c1 on row lane/4 and
    // c2 and c3 eight rows below, with the column pair at (lane%4)*2.
#pragma unroll
    for (int i = 0; i < kTilesM; ++i) {
#pragma unroll
        for (int j = 0; j < kTilesN; ++j) {
            const int rowBase = blockRow + warpM * WM + i * kMmaM + laneId / 4;
            const int colBase = blockCol + warpN * WN + j * kMmaN + (laneId % 4) * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = rowBase + (e / 2) * 8;
                const int col = colBase + (e % 2);
                C[row * N + col] = acc[i][j][e];
            }
        }
    }
}

// Shared memory bytes the kernel needs for a given config. measure.cu uses this
// both to size the dynamic allocation and to opt into the large shared memory
// carveout, and Phase 3 compares it against the model's smemBytesPerCta.
template <int BM, int BN, int BK, int Stages>
constexpr int smemBytesForConfig() {
    return Stages * (BM * BK + BK * BN) * static_cast<int>(sizeof(half));
}

} // namespace ridgebench
