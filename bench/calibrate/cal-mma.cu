// cal-mma.cu measures mmaCyclesPerInst: the SM-level cycle cost of one
// mma.sync.m16n8k16, folding in all tensor cores and warp schedulers.
//
// This is the constant behind peakTensorTFLOPS:
//   peakTensorTFLOPS = (mmaFlops / mmaCyclesPerInst) * clockHz * numSMs / 1e12
//
// THE METHODOLOGICAL TRAP: throughput is not latency. An mma has a result
// latency of order tens of cycles, but the SM can have many in flight. A loop of
// mma instructions that all accumulate into the *same* registers forms a
// dependency chain and measures latency, which would give a constant several
// times too large and a peak several times too low. The fix is instruction level
// parallelism: each warp keeps kIlp independent accumulator sets so the tensor
// cores always have work queued, and the loop measures issue rate.
//
// Everything is register resident. No shared memory, no global traffic, nothing
// that could make this a memory measurement wearing a compute measurement's
// clothes.
//
// The warp count is swept because "SM-level" only means anything once the SM is
// saturated. One warp cannot keep the tensor cores busy no matter how much ILP
// it has, so the reported constant is taken at the warp count that minimises
// cycles per MMA, and the whole curve is printed so the saturation point is
// visible rather than assumed.

#include "../kernels/gemm-mma.cuh"
#include "cal-common.cuh"

#include <cstdio>
#include <vector>

namespace {

// Independent accumulator sets per warp. Each costs 4 registers, so 8 sets is 32
// registers of accumulator, comfortably inside budget while giving the scheduler
// eight independent chains to interleave.
constexpr int kIlp = 8;

constexpr int kIters = 2000;
constexpr int kRepeats = 7;

template <int Warps>
__global__ __launch_bounds__(Warps * 32) void mmaThroughputKernel(
    int iters, float* sink, long long* cycleOut) {

    // Operands are held in registers for the whole measurement. The bit patterns
    // are two halves of approximately 1.0, perturbed per lane so nothing can be
    // constant-folded, and finite so no denormal or NaN path is exercised.
    const uint32_t perturb = threadIdx.x & 0xF;
    uint32_t a[4] = {0x3C003C00u + perturb, 0x3C003C00u + perturb + 1,
                     0x3C003C00u + perturb + 2, 0x3C003C00u + perturb + 3};
    uint32_t b[2] = {0x3C003C00u + perturb + 4, 0x3C003C00u + perturb + 5};

    float acc[kIlp][4];
#pragma unroll
    for (int j = 0; j < kIlp; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = float(threadIdx.x) * 1e-6f;

    __syncthreads();
    const long long t0 = clock64();

    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < kIlp; ++j) {
            ridgebench::mmaM16N8K16(acc[j], a, b);
        }
    }

    const long long t1 = clock64();

    // Consume the accumulators so nothing is dead code. The store is outside the
    // timed region and is predicated on a value that cannot occur.
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < kIlp; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) s += acc[j][e];
    if (s == 1.2345678e30f) sink[0] = s;

    if (threadIdx.x == 0) cycleOut[blockIdx.x] = t1 - t0;
}

// One measurement: cycles per MMA at the SM level, for a given warp count.
//
// One block per SM, so the block's elapsed cycles cover every MMA that SM
// issued. That is what makes the result an SM-level number rather than a
// per-warp one.
template <int Warps>
double measureCyclesPerMma(int numSMs, float* dSink, long long* dCycles) {
    mmaThroughputKernel<Warps><<<numSMs, Warps * 32>>>(kIters, dSink, dCycles);
    CAL_CUDA_CHECK(cudaGetLastError());
    CAL_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<long long> cycles(numSMs);
    CAL_CUDA_CHECK(cudaMemcpy(cycles.data(), dCycles, numSMs * sizeof(long long),
                              cudaMemcpyDeviceToHost));

    // Median across SMs. One straggler SM should not define the constant.
    std::vector<double> asDouble(cycles.begin(), cycles.end());
    const double medianCycles = ridgecal::median(asDouble);

    const double mmasPerSm = double(Warps) * kIters * kIlp;
    return medianCycles / mmasPerSm;
}

} // namespace

int main() {
    const cudaDeviceProp prop = ridgecal::requireLockedClocks();
    const int numSMs = prop.multiProcessorCount;

    float* dSink = nullptr;
    long long* dCycles = nullptr;
    CAL_CUDA_CHECK(cudaMalloc(&dSink, sizeof(float)));
    CAL_CUDA_CHECK(cudaMalloc(&dCycles, numSMs * sizeof(long long)));

    ridgecal::warmUpGpu();

    std::printf("\nmma.sync.m16n8k16 issue rate, %d ILP chains per warp, "
                "%d iterations\n", kIlp, kIters);
    std::printf("  %6s %16s %16s\n", "warps", "cycles/mma", "implied TFLOP/s");

    // mmaFlops for m16n8k16 is 2*16*8*16.
    constexpr double kMmaFlops = 2.0 * 16.0 * 8.0 * 16.0;
    const double clockHz = prop.clockRate * 1e3;

    double best = 1e30;
    int bestWarps = 0;
    std::vector<double> bestRepeats;

    // A lambda would need a template parameter, so the sweep is written out.
    // Warp counts are powers of two up to the 32 warp limit of an SM.
    #define CAL_SWEEP(W)                                                        \
        do {                                                                    \
            std::vector<double> reps;                                           \
            for (int r = 0; r < kRepeats; ++r)                                  \
                reps.push_back(measureCyclesPerMma<W>(numSMs, dSink, dCycles));  \
            const double c = ridgecal::median(reps);                            \
            const double tflops = (kMmaFlops / c) * clockHz * numSMs / 1e12;    \
            std::printf("  %6d %16.4f %16.1f\n", W, c, tflops);                 \
            if (c < best) { best = c; bestWarps = W; bestRepeats = reps; }      \
        } while (0)

    CAL_SWEEP(1);
    CAL_SWEEP(2);
    CAL_SWEEP(4);
    CAL_SWEEP(8);
    CAL_SWEEP(16);
    CAL_SWEEP(32);
    #undef CAL_SWEEP

    const double peakTflops = (kMmaFlops / best) * clockHz * numSMs / 1e12;

    std::printf("\nsaturated at %d warps per SM\n", bestWarps);
    const bool stable =
        ridgecal::reportConstant("mmaCyclesPerInst", best, "cycles", bestRepeats);
    std::printf("  %-22s %12.1f TFLOP/s\n", "implied peak", peakTflops);
    std::printf("  %-22s %12.1f TFLOP/s   (datasheet, for cross-check only)\n",
                "A100 FP16 dense", 312.0);

    std::printf("\nCAL_RESULT mmaCyclesPerInst %.6f\n", best);
    std::printf("CAL_RESULT impliedPeakTflops %.3f\n", peakTflops);
    std::printf("CAL_RESULT saturationWarps %d\n", bestWarps);

    CAL_CUDA_CHECK(cudaFree(dSink));
    CAL_CUDA_CHECK(cudaFree(dCycles));
    return stable ? 0 : 1;
}
