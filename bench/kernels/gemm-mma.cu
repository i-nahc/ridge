// gemm-mma.cu instantiates the ground truth kernel over the config space that
// Ridge sweeps.
//
// The kernel itself is a template in gemm-mma.cuh. This file turns it into a
// table of concrete launchers so measure.cu can walk the sweep without knowing
// anything about templates, and so ptxas reports register and shared memory
// usage per config when this file is compiled with -Xptxas -v. Those two numbers
// feed straight back into the model as regsPerThread and smemBytesPerCta, see
// SPEC section 5.

#include "gemm-mma.cuh"

#include <cstdio>

namespace ridgebench {

// KernelVariant is declared in gemm-mma.cuh so that measure.cu and check-gemm.cu
// see the same config space. This file owns the table and the instantiations.

template <int BM, int BN, int BK, int WM, int WN, int Stages>
static cudaError_t launchVariant(const half* A, const half* B, float* C,
                                 int M, int N, int K, cudaStream_t stream) {
    constexpr int kThreads = (BM / WM) * (BN / WN) * 32;
    constexpr int kSmem = smemBytesForConfig<BM, BN, BK, Stages>();

    auto kernel = gemmMmaKernel<BM, BN, BK, WM, WN, Stages>;

    // Anything above 48 KB of shared memory per block has to be opted into
    // explicitly on Ampere, and several of the interesting configs are above it.
    if (kSmem > 48 * 1024) {
        cudaError_t err = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
        if (err != cudaSuccess) return err;
    }

    dim3 grid(N / BN, M / BM);
    dim3 block(kThreads);
    kernel<<<grid, block, kSmem, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

// The sweep. These span the tile shapes the model should be able to tell apart,
// including configs that should be shared memory bound and configs that should
// run out of occupancy, so Phase 4 has something to explain.
#define RIDGE_VARIANT(BM, BN, BK, WM, WN, S, REGS)                             \
    KernelVariant{BM, BN, BK, WM, WN, S,                                       \
                  smemBytesForConfig<BM, BN, BK, S>(),                         \
                  (BM / WM) * (BN / WN) * 32,                                  \
                  REGS,                                                        \
                  &launchVariant<BM, BN, BK, WM, WN, S>}

static const KernelVariant kVariants[] = {
    // The default config from docs/MODEL.md section 9, so the worked example is
    // measurable directly.
    RIDGE_VARIANT(128, 128, 32, 64, 64, 3, 224),
    RIDGE_VARIANT(128, 128, 32, 64, 64, 4, 225),
    RIDGE_VARIANT(128, 128, 64, 64, 64, 3, 243),
    RIDGE_VARIANT(128, 128, 64, 64, 32, 3, 125),
    RIDGE_VARIANT(128, 256, 32, 64, 64, 3, 236),
    RIDGE_VARIANT(256, 128, 32, 64, 64, 3, 226),
    RIDGE_VARIANT(128, 64, 64, 64, 32, 4, 127),
    RIDGE_VARIANT(64, 128, 64, 32, 64, 4, 167),
    RIDGE_VARIANT(64, 64, 64, 32, 32, 4, 91),
    RIDGE_VARIANT(256, 128, 64, 64, 64, 3, 248),

    // Narrow warp tiles at BK=32, added after the first hardware run.
    //
    // That run reached only 59.5% of cuBLAS and the reason was visible in the
    // measurements: every config above tops out at 8 active warps per SM, which
    // is 12.5% occupancy. A 64x64 warp tile needs (64/16)*(64/8)*4 = 128
    // accumulator floats per thread before any addressing registers, and ptxas
    // reports 224, so only one or two CTAs fit an SM.
    //
    // Halving the warp tile in N halves the accumulators and doubles the warps
    // per CTA. The original table only ever paired a 64x32 warp tile with BK=64,
    // where the shared memory footprint then collapses occupancy back to one
    // CTA, so the combination that should actually work was never measured.
    // These fill that hole. Register counts come from ptxas, see the note on
    // KernelVariant.
    // Registers drop from 224 to 124 for the same CTA tile, which is the whole
    // point: 2 CTAs of 8 warps instead of 2 CTAs of 4, so 16 active warps and
    // 25% occupancy rather than 12.5%.
    RIDGE_VARIANT(128, 128, 32, 64, 32, 3, 124),
    RIDGE_VARIANT(128, 128, 32, 64, 32, 4, 124),
    RIDGE_VARIANT(128, 128, 32, 64, 32, 5, 124),
    RIDGE_VARIANT(128, 128, 32, 32, 64, 4, 127),
    RIDGE_VARIANT(128, 128, 32, 32, 32, 4, 95),
    RIDGE_VARIANT(256, 128, 32, 64, 32, 3, 125),
    RIDGE_VARIANT(128, 256, 32, 64, 32, 3, 125),

    // 256x256x32 with a 64x64 warp tile was tried and removed. It needs 16 warps
    // per CTA, so __launch_bounds__(512) caps ptxas at 128 registers while the
    // accumulators alone want about 250, and it spills 628 bytes. A spilling
    // kernel is not a fair ground truth, so it is not in the sweep.
};

#undef RIDGE_VARIANT

int numVariants() {
    return static_cast<int>(sizeof(kVariants) / sizeof(kVariants[0]));
}

const KernelVariant& variant(int i) {
    return kVariants[i];
}

// Prints the compile time facts about every variant. This runs without a GPU,
// which makes it useful before any hardware is rented. The shared memory column
// is the same quantity the model calls smemBytesPerCta.
void printVariantTable() {
    std::printf("%3s %5s %5s %5s %5s %5s %7s %8s %8s %6s\n",
                "idx", "BM", "BN", "BK", "WM", "WN", "stages", "smemB", "threads", "regs");
    for (int i = 0; i < numVariants(); ++i) {
        const KernelVariant& v = kVariants[i];
        std::printf("%3d %5d %5d %5d %5d %5d %7d %8d %8d %6d\n",
                    i, v.BM, v.BN, v.BK, v.WM, v.WN, v.stages,
                    v.smemBytes, v.threads, v.regsPerThread);
    }
}

} // namespace ridgebench
