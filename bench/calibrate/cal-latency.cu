// cal-latency.cu measures warpsNeededToHide: how many active warps per SM are
// required before throughput stops improving.
//
// This is the constant behind the occupancy term in docs/MODEL.md section 4:
//   occFactor = min(1.0, activeWarps / warpsNeededToHide)
//
// WHY THIS IS NOT JUST cal-mma's SATURATION POINT. cal-mma sweeps warps on a
// pure register-resident MMA loop, which measures how many warps it takes to
// keep the tensor cores issuing. That is a lower number than the kernel needs,
// because the real inner loop also waits on ldmatrix returning from shared
// memory. Latency that has to be hidden is latency from the whole dependency
// chain, not from the arithmetic alone.
//
// So the workload here is the kernel's actual inner loop shape: ldmatrix pulling
// operands out of shared memory, immediately consumed by mma. The knee of that
// curve is the number the model wants.
//
// The result is deliberately reported as a curve and not only as a single
// number. "Where does it saturate" has a soft answer, the model treats it as
// hard, and hiding that behind one integer would be exactly the kind of tidy
// fiction the project is trying to avoid.

#include "../kernels/gemm-mma.cuh"
#include "cal-common.cuh"

#include <cstdio>
#include <vector>

namespace {

constexpr int kRowHalves = 64;
constexpr int kPaddedRow = kRowHalves + 8;
constexpr int kRows = 128;
constexpr int kSmemHalves = kRows * kPaddedRow;

// Deliberately low ILP. cal-mma uses 8 independent chains to measure pure issue
// rate with latency hidden by construction; here we want latency to matter, so
// each warp keeps only two chains in flight and the SM has to find its
// parallelism across warps instead. That is what makes this a measurement of
// warps needed rather than of ILP needed.
constexpr int kIlp = 2;

constexpr int kIters = 2000;
constexpr int kRepeats = 7;

// Fraction of the best observed throughput at which the curve is considered
// saturated. 95% is the same threshold the model uses when it decides a config
// is occupancy-bound (occFactor < 0.95 in Model.cpp).
constexpr double kSaturationFraction = 0.95;

template <int Warps>
__global__ __launch_bounds__(Warps * 32) void ldmatrixMmaKernel(
    int iters, float* sink, long long* cycleOut) {

    __shared__ half smem[kSmemHalves];
    for (int i = int(threadIdx.x); i < kSmemHalves; i += Warps * 32) {
        smem[i] = __float2half(float(i) * 1e-4f);
    }
    __syncthreads();

    const int lane = int(threadIdx.x) % 32;
    const int warp = int(threadIdx.x) / 32;
    const int mi = lane / 8;
    const int rowInTile = lane % 8;

    uint32_t addrA[kIlp], addrB[kIlp];
#pragma unroll
    for (int j = 0; j < kIlp; ++j) {
        const int band = ((warp * kIlp + j) * 16) % (kRows - 16);
        const int row = band + (mi % 2) * 8 + rowInTile;
        addrA[j] = ridgebench::smemAddr(
            smem + ridgebench::padOffset<kRowHalves>(row, mi / 2));
        addrB[j] = ridgebench::smemAddr(
            smem + ridgebench::padOffset<kRowHalves>((row + 32) % kRows, mi / 2));
    }

    float acc[kIlp][4];
#pragma unroll
    for (int j = 0; j < kIlp; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;

    __syncthreads();
    const long long t0 = clock64();

    // The dependency chain that matters: load operands from shared memory, then
    // immediately feed them to the tensor cores. The mma cannot issue until the
    // ldmatrix returns, which is precisely the latency the occupancy term exists
    // to describe.
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < kIlp; ++j) {
            uint32_t a[4], b[2];
            ridgebench::ldmatrixX4(addrA[j], a);
            uint32_t bpair[4];
            ridgebench::ldmatrixX4(addrB[j], bpair);
            b[0] = bpair[0];
            b[1] = bpair[1];
            ridgebench::mmaM16N8K16(acc[j], a, b);
        }
    }

    const long long t1 = clock64();

    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < kIlp; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) s += acc[j][e];
    if (s == 1.2345678e30f) sink[0] = s;

    if (threadIdx.x == 0) cycleOut[blockIdx.x] = t1 - t0;
}

// Returns MMAs issued per cycle per SM. Rises with warp count until latency is
// hidden, then flattens.
template <int Warps>
double measureMmaPerCycle(int numSMs, float* dSink, long long* dCycles) {
    ldmatrixMmaKernel<Warps><<<numSMs, Warps * 32>>>(kIters, dSink, dCycles);
    CAL_CUDA_CHECK(cudaGetLastError());
    CAL_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<long long> cycles(numSMs);
    CAL_CUDA_CHECK(cudaMemcpy(cycles.data(), dCycles, numSMs * sizeof(long long),
                              cudaMemcpyDeviceToHost));
    std::vector<double> asDouble(cycles.begin(), cycles.end());
    const double medianCycles = ridgecal::median(asDouble);

    const double mmas = double(Warps) * kIters * kIlp;
    return mmas / medianCycles;
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

    std::printf("\nldmatrix -> mma dependency chain, %d chains per warp\n", kIlp);
    std::printf("  %6s %18s %14s\n", "warps", "mma/cycle/SM", "vs best");

    std::vector<int> warpCounts;
    std::vector<double> throughputs;
    std::vector<double> bestRepeats;
    double best = 0.0;

    #define CAL_SWEEP(W)                                                        \
        do {                                                                    \
            std::vector<double> reps;                                           \
            for (int r = 0; r < kRepeats; ++r)                                  \
                reps.push_back(measureMmaPerCycle<W>(numSMs, dSink, dCycles));   \
            const double v = ridgecal::median(reps);                            \
            warpCounts.push_back(W);                                            \
            throughputs.push_back(v);                                           \
            if (v > best) { best = v; bestRepeats = reps; }                     \
        } while (0)

    CAL_SWEEP(1);
    CAL_SWEEP(2);
    CAL_SWEEP(4);
    CAL_SWEEP(8);
    CAL_SWEEP(16);
    CAL_SWEEP(32);
    #undef CAL_SWEEP

    for (size_t i = 0; i < warpCounts.size(); ++i) {
        std::printf("  %6d %18.4f %13.1f%%\n", warpCounts[i], throughputs[i],
                    100.0 * throughputs[i] / best);
    }

    // The knee: the smallest warp count reaching kSaturationFraction of the best
    // observed throughput.
    int needed = warpCounts.back();
    for (size_t i = 0; i < warpCounts.size(); ++i) {
        if (throughputs[i] >= kSaturationFraction * best) {
            needed = warpCounts[i];
            break;
        }
    }

    std::printf("\n  first warp count within %.0f%% of best: %d\n",
                kSaturationFraction * 100.0, needed);
    ridgecal::reportConstant("warpsNeededToHide", double(needed), "warps/SM",
                             bestRepeats);
    std::printf("\n  This is a knee on a smooth curve, not a hard threshold. The\n"
                "  model treats it as one, which is a simplification worth\n"
                "  remembering when a config lands near the boundary. Read the\n"
                "  curve above before trusting the single number.\n");

    std::printf("\nCAL_RESULT warpsNeededToHide %d\n", needed);
    for (size_t i = 0; i < warpCounts.size(); ++i) {
        std::printf("CAL_CURVE warps %d mmaPerCycle %.6f\n", warpCounts[i],
                    throughputs[i]);
    }

    CAL_CUDA_CHECK(cudaFree(dSink));
    CAL_CUDA_CHECK(cudaFree(dCycles));
    return 0;
}
