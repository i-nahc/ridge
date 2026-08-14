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

// The ILP the reported constant is taken from. This used to be the only ILP
// measured, and that was the problem: the knee it found (4 warps on A100) is a
// knee *at two chains per warp*, and the model then applied it to kernels whose
// dependency structure is nothing like that. Configs at exactly 4 active warps
// over-predict by about 39% as a result, which is the single largest error in
// the model. See PLAN.md Findings 17 and 19.
//
// The sweep below now measures the whole warps x ILP surface so the ILP
// dependence is visible rather than assumed. This constant only selects which
// row of that surface becomes warpsNeededToHide.
constexpr int kIlp = 2;

constexpr int kIters = 2000;
constexpr int kRepeats = 7;

// Fraction of the best observed throughput at which the curve is considered
// saturated. 95% is the same threshold the model uses when it decides a config
// is occupancy-bound (occFactor < 0.95 in Model.cpp).
constexpr double kSaturationFraction = 0.95;

template <int Warps, int Ilp>
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

    uint32_t addrA[Ilp], addrB[Ilp];
#pragma unroll
    for (int j = 0; j < Ilp; ++j) {
        const int band = ((warp * Ilp + j) * 16) % (kRows - 16);
        const int row = band + (mi % 2) * 8 + rowInTile;
        addrA[j] = ridgebench::smemAddr(
            smem + ridgebench::padOffset<kRowHalves>(row, mi / 2));
        addrB[j] = ridgebench::smemAddr(
            smem + ridgebench::padOffset<kRowHalves>((row + 32) % kRows, mi / 2));
    }

    float acc[Ilp][4];
#pragma unroll
    for (int j = 0; j < Ilp; ++j)
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
        for (int j = 0; j < Ilp; ++j) {
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
    for (int j = 0; j < Ilp; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) s += acc[j][e];
    if (s == 1.2345678e30f) sink[0] = s;

    if (threadIdx.x == 0) cycleOut[blockIdx.x] = t1 - t0;
}

// Returns MMAs issued per cycle per SM. Rises with warp count until latency is
// hidden, then flattens.
template <int Warps, int Ilp>
double measureMmaPerCycle(int numSMs, float* dSink, long long* dCycles) {
    ldmatrixMmaKernel<Warps, Ilp><<<numSMs, Warps * 32>>>(kIters, dSink, dCycles);
    CAL_CUDA_CHECK(cudaGetLastError());
    CAL_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<long long> cycles(numSMs);
    CAL_CUDA_CHECK(cudaMemcpy(cycles.data(), dCycles, numSMs * sizeof(long long),
                              cudaMemcpyDeviceToHost));
    std::vector<double> asDouble(cycles.begin(), cycles.end());
    const double medianCycles = ridgecal::median(asDouble);

    const double mmas = double(Warps) * kIters * Ilp;
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

    // Sweep the whole warps x ILP surface. The knee is not a property of the
    // hardware alone, it is a property of the hardware plus how much independent
    // work each warp carries, and collapsing that to one number is what produced
    // the model's largest error.
    std::printf("\nldmatrix -> mma dependency chain\n");
    std::printf("mma/cycle/SM, and %% of the best at that ILP\n\n");
    std::printf("  %5s", "warps");
    std::printf(" %22s %22s %22s %22s\n", "ILP=1", "ILP=2", "ILP=4", "ILP=8");

    constexpr int kWarpRow[] = {1, 2, 4, 8, 16, 32};
    constexpr int kNumWarpRows = 6;
    constexpr int kIlpCol[] = {1, 2, 4, 8};
    constexpr int kNumIlpCols = 4;

    // surface[ilpIndex][warpIndex]
    double surface[kNumIlpCols][kNumWarpRows] = {};
    std::vector<double> repsFor[kNumIlpCols][kNumWarpRows];

    #define CAL_CELL(IC, W_IDX, W, ILP)                                          \
        do {                                                                     \
            std::vector<double> reps;                                            \
            for (int r = 0; r < kRepeats; ++r)                                   \
                reps.push_back(measureMmaPerCycle<W, ILP>(numSMs, dSink, dCycles)); \
            surface[IC][W_IDX] = ridgecal::median(reps);                         \
            repsFor[IC][W_IDX] = reps;                                           \
        } while (0)
    #define CAL_ILP_COLUMN(IC, ILP)                                              \
        CAL_CELL(IC, 0, 1, ILP);  CAL_CELL(IC, 1, 2, ILP);                       \
        CAL_CELL(IC, 2, 4, ILP);  CAL_CELL(IC, 3, 8, ILP);                       \
        CAL_CELL(IC, 4, 16, ILP); CAL_CELL(IC, 5, 32, ILP)

    CAL_ILP_COLUMN(0, 1);
    CAL_ILP_COLUMN(1, 2);
    CAL_ILP_COLUMN(2, 4);
    CAL_ILP_COLUMN(3, 8);
    #undef CAL_ILP_COLUMN
    #undef CAL_CELL

    double bestPerIlp[kNumIlpCols] = {};
    for (int c = 0; c < kNumIlpCols; ++c)
        for (int w = 0; w < kNumWarpRows; ++w)
            if (surface[c][w] > bestPerIlp[c]) bestPerIlp[c] = surface[c][w];

    for (int w = 0; w < kNumWarpRows; ++w) {
        std::printf("  %5d", kWarpRow[w]);
        for (int c = 0; c < kNumIlpCols; ++c) {
            std::printf("   %10.4f (%5.1f%%)", surface[c][w],
                        100.0 * surface[c][w] / bestPerIlp[c]);
        }
        std::printf("\n");
    }

    // The knee per ILP column. If these differ, warpsNeededToHide is not a
    // hardware constant and the model must not treat it as one.
    std::printf("\n  knee (first warp count within %.0f%% of best) per ILP:\n",
                kSaturationFraction * 100.0);
    for (int c = 0; c < kNumIlpCols; ++c) {
        int knee = kWarpRow[kNumWarpRows - 1];
        for (int w = 0; w < kNumWarpRows; ++w) {
            if (surface[c][w] >= kSaturationFraction * bestPerIlp[c]) {
                knee = kWarpRow[w];
                break;
            }
        }
        std::printf("    ILP=%d  knee %2d warps/SM\n", kIlpCol[c], knee);
        std::printf("CAL_KNEE ilp %d warps %d\n", kIlpCol[c], knee);
    }
    for (int c = 0; c < kNumIlpCols; ++c)
        for (int w = 0; w < kNumWarpRows; ++w)
            std::printf("CAL_SURFACE ilp %d warps %d mmaPerCycle %.6f\n",
                        kIlpCol[c], kWarpRow[w], surface[c][w]);

    // Everything below reports the kIlp column, which is what the model consumes.
    const int kIlpIndex = 1; // kIlp == 2
    std::vector<int> warpCounts(kWarpRow, kWarpRow + kNumWarpRows);
    std::vector<double> throughputs(surface[kIlpIndex],
                                    surface[kIlpIndex] + kNumWarpRows);
    const double best = bestPerIlp[kIlpIndex];
    std::vector<double> bestRepeats;
    for (int w = 0; w < kNumWarpRows; ++w)
        if (surface[kIlpIndex][w] == best) bestRepeats = repsFor[kIlpIndex][w];

    std::printf("\n  reported constant is the ILP=%d column\n", kIlp);

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
