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
#include <cuda_runtime.h>
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

// Shared memory tiles are stored row major in 16 byte chunks of 8 halves, with
// every row padded by one extra chunk.
//
// The problem being solved: ldmatrix has 8 lanes read 8 consecutive rows at the
// same column offset. With an unpadded row stride those 8 addresses collide in
// the shared memory banks and the access serialises.
//
// This originally used an XOR swizzle on the chunk index, on the reasoning that
// padding would break the 16 byte alignment cp.async requires. That reasoning
// was wrong in a useful way. Padding by a whole 16 byte chunk keeps every row
// start 16 byte aligned, so cp.async is happy, and it spreads the banks cleanly.
// The XOR version was only conflict free when a row held exactly 8 chunks, which
// none of the interesting BK=32 configs do, so it left a two way conflict
// exactly where it hurt most.
//
// Why 8 halves of padding is the right amount, for a row of R halves. The stride
// becomes (R + 8) halves, which is (R + 8) * 2 bytes, and bank index advances by
// ((R + 8) * 2 / 4) mod 32 = ((R + 8) / 2) mod 32 per row. For R = 32 that is 20
// per row, giving banks 0, 20, 8, 28, 16, 4, 24, 12 across the 8 rows, which are
// distinct and cover all 32 banks with the 4 banks each 16 byte access touches.
// For R = 64, 128 and 256 the step is 4 per row, giving 0, 4, 8 ... 28, also
// distinct. So every row width in the sweep is conflict free.
//
// The cost is shared memory: a BK=32 A tile grows from 32 to 40 halves per row.
// That can cost one resident CTA, but every config in the sweep is already
// register limited at or below that count, so in practice it costs nothing.
template <int RowHalves>
__device__ __forceinline__ int padOffset(int row, int chunk) {
    return row * (RowHalves + 8) + chunk * 8;
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

template <int BM, int BN, int BK, int WM, int WN, int Stages, int GroupM>
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

    constexpr int kSmemHalvesA = BM * (BK + 8);
    constexpr int kSmemHalvesB = BK * (BN + 8);

    extern __shared__ half smem[];
    half* smemA = smem;
    half* smemB = smem + Stages * kSmemHalvesA;

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warpM = warpId / kWarpsN;
    const int warpN = warpId % kWarpsN;

    // CTA launch order, grouped along M so that neighbouring CTAs share operands
    // in L2.
    //
    // The natural row-major order walks all of N before advancing M, so the
    // CTAs live at any instant span the full B panel and only one A row band.
    // Their combined footprint is large and L2 hit rate suffers. Grouping
    // kGroupM row-tiles together means the CTAs in flight share both a small set
    // of A rows and a small set of B columns, which is a much smaller working
    // set for the same amount of work.
    //
    // This is the "block swizzle" that TileSight lists as a tile execution plan
    // parameter and that tritonBLAS selects analytically, see docs/PAPERS.md. It
    // changes no arithmetic, only which CTA computes which output tile, so it
    // cannot affect correctness.
    constexpr int kGroupM = GroupM;
    const int numPidM = M / BM;
    const int numPidN = N / BN;
    const int pid = int(blockIdx.x);
    const int numPidInGroup = kGroupM * numPidN;
    const int groupId = pid / numPidInGroup;
    const int firstPidM = groupId * kGroupM;
    const int groupRows = min(numPidM - firstPidM, kGroupM);
    const int pidM = firstPidM + ((pid % numPidInGroup) % groupRows);
    const int pidN = (pid % numPidInGroup) / groupRows;

    const int blockRow = pidM * BM;
    const int blockCol = pidN * BN;

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
            cpAsyncCg16(smemAddr(dstA + padOffset<BK>(row, chunk)), src);
        }
#pragma unroll
        for (int i = threadIdx.x; i < kChunksB; i += kThreads) {
            const int row = i / kChunksPerRowB;
            const int chunk = i % kChunksPerRowB;
            const half* src = B + (kTile * BK + row) * N + blockCol + chunk * 8;
            cpAsyncCg16(smemAddr(dstB + padOffset<BN>(row, chunk)), src);
        }
        cpAsyncCommit();
    };

    // Prologue. Fill every buffer but the last so the first main loop iteration
    // already has work in flight behind it.
    //
    // The bound check matters. When K is small enough that numKTiles is less
    // than Stages-1, an unguarded prologue issues copies for K-tiles that do not
    // exist, which reads past the end of A and B and leaves stray writes landing
    // in shared memory buffers long after the main loop has moved on. With
    // Stages=4 and BK=64, a K of 64 gives numKTiles=1 and issues two such
    // phantom tiles. An empty commit keeps the wait_group accounting uniform,
    // exactly as in the tail below.
#pragma unroll
    for (int s = 0; s < Stages - 1; ++s) {
        if (s < numKTiles) {
            loadTile(s, s);
        } else {
            cpAsyncCommit();
        }
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
                ldmatrixX4(smemAddr(curA + padOffset<BK>(row, kOff / 8)), fragA[i]);
            }
            // One N tile per instruction.
            //
            // Loading two tiles at a time with ldmatrix.x4.trans was tried and
            // measured slower, so it was reverted. It halves the B side
            // instruction count from eight to four, which is why it looked like
            // a clear win on paper. Two effects plausibly cancel it: the x4 form
            // has all 32 lanes address the shared memory banks where x2 uses
            // only 16, and the results arrive in one register quad that then has
            // to be split across two fragments, which the scheduler pays for.
            //
            // Recorded rather than deleted because "fewer instructions must be
            // faster" is exactly the kind of reasoning that keeps looking
            // correct until someone measures it.
#pragma unroll
            for (int j = 0; j < kTilesN; ++j) {
                const int mi = (laneId % 16) / 8;
                const int rowInTile = laneId % 8;
                const int kRow = kStep * kMmaK + mi * 8 + rowInTile;
                const int nOff = warpN * WN + j * kMmaN;
                ldmatrixX2Trans(smemAddr(curB + padOffset<BN>(kRow, nOff / 8)), fragB[j]);
            }
#pragma unroll
            for (int i = 0; i < kTilesM; ++i)
#pragma unroll
                for (int j = 0; j < kTilesN; ++j)
                    mmaM16N8K16(acc[i][j], fragA[i], fragB[j]);
        }
    }

    // Epilogue, staged through shared memory so the global stores are fully
    // coalesced.
    //
    // The reason this is worth the complexity: the m16n8k16 accumulator layout
    // puts lane l at row l/4 and column (l%4)*2, so a warp's 32 lanes cover an
    // 8 by 8 patch of C. Writing that straight to global means every store
    // instruction touches 8 separate rows, and no amount of widening the
    // individual store fixes that, because the rows are N floats apart. Direct
    // 4 byte stores and direct 8 byte stores are both bad for the same reason.
    //
    // Bouncing through shared memory decouples the accumulator layout from the
    // global layout. Warps scatter their accumulators into a shared C tile, then
    // all threads cooperatively stream it out in 16 byte chunks, so each warp
    // emits fully contiguous 128 byte transactions.
    //
    // Measured basis for doing this: swapping four 4 byte stores for two 8 byte
    // stores was worth about four points of cuBLAS, far more than the CTA
    // swizzle was, which identified the epilogue as the binding cost rather than
    // a rounding error.
    //
    // WAITING HERE IS NOT OPTIONAL. This buffer is the operand pipeline, and any
    // cp.async still in flight will land in it after we start writing C. A
    // barrier alone does not help, because __syncthreads() synchronises threads
    // and says nothing about outstanding asynchronous copies. wait_group<0>
    // drains every committed group first.
    //
    // This was a real bug, not a hypothetical. The first version of this
    // epilogue omitted the wait and silently corrupted C for configs where the
    // prologue had issued phantom tiles, and because it is a race it corrupted
    // only some of them on any given run.
    cpAsyncWaitGroup<0>();
    __syncthreads();

    // Row stride padded by 4 floats. Unpadded, a BN=128 row is exactly 512
    // bytes, so every row starts on bank 0 and the scatter phase below
    // serialises across the 8 rows a warp touches. Padding by 4 floats breaks
    // that while keeping the 16 byte alignment the streaming phase needs.
    constexpr int kEpiStride = BN + 4;
    float* smemC = reinterpret_cast<float*>(smem);

#pragma unroll
    for (int i = 0; i < kTilesM; ++i) {
#pragma unroll
        for (int j = 0; j < kTilesN; ++j) {
            const int r0 = warpM * WM + i * kMmaM + laneId / 4;
            const int c0 = warpN * WN + j * kMmaN + (laneId % 4) * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                smemC[(r0 + (e / 2) * 8) * kEpiStride + c0 + (e % 2)] = acc[i][j][e];
            }
        }
    }
    __syncthreads();

    // Stream the tile out. Consecutive threads take consecutive 16 byte chunks
    // of the same row, so a warp covers 512 contiguous bytes per instruction.
    constexpr int kVecPerRow = BN / 4;
    for (int idx = int(threadIdx.x); idx < BM * kVecPerRow; idx += kThreads) {
        const int r = idx / kVecPerRow;
        const int c = idx % kVecPerRow;
        const float4 v =
            *reinterpret_cast<const float4*>(&smemC[r * kEpiStride + c * 4]);
        *reinterpret_cast<float4*>(
            &C[size_t(blockRow + r) * N + blockCol + c * 4]) = v;
    }
}

// Shared memory bytes the kernel needs for a given config. measure.cu uses this
// both to size the dynamic allocation and to opt into the large shared memory
// carveout, and Phase 3 compares it against the model's smemBytesPerCta.
// Must match the padded layout in padOffset exactly. If this under-counts, the
// kernel writes past the end of its dynamic shared memory allocation.
//
// The allocation has to cover two different uses of the same buffer: the
// multi-stage operand pipeline during the main loop, and the C tile staged
// through shared memory in the epilogue after the loop is done. They never
// overlap in time, so the requirement is the larger of the two, not the sum.
template <int BM, int BN, int BK, int Stages>
constexpr int smemBytesForConfig() {
    const int pipeline =
        Stages * (BM * (BK + 8) + BK * (BN + 8)) * static_cast<int>(sizeof(half));
    const int epilogue = BM * (BN + 4) * static_cast<int>(sizeof(float));
    return pipeline > epilogue ? pipeline : epilogue;
}

// ---------------------------------------------------------------------------
// The sweep table
// ---------------------------------------------------------------------------

// One entry in the config sweep. The fields mirror ridge::GemmConfig so that a
// measured row and a predicted row line up field for field in Phase 4. The table
// itself lives in gemm-mma.cu, which is also where the launchers are
// instantiated, so every consumer of this header sees the same config space.
struct KernelVariant {
    int BM, BN, BK, WM, WN, stages;
    // CTA launch-order group size along M. This is a real tile execution plan
    // parameter, the "block swizzle" in TileSight's terms, so it belongs in the
    // sweep rather than hardcoded. groupM of 1 is plain row-major order, which
    // makes it the control.
    int groupM;
    int smemBytes;
    int threads;
    // Registers per thread as reported by ptxas -v for sm_80. A compile time
    // fact, so it is recorded rather than measured on hardware, and it is the
    // regsPerThread input the model needs for its occupancy term. Refresh with:
    //   nvcc -arch=sm_80 -O3 -std=c++17 -Xptxas -v -c bench/kernels/gemm-mma.cu
    int regsPerThread;
    cudaError_t (*launch)(const half*, const half*, float*, int, int, int, cudaStream_t);
};

int numVariants();
const KernelVariant& variant(int i);
void printVariantTable();

} // namespace ridgebench
