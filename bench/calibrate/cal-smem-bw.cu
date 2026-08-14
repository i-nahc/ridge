// cal-smem-bw.cu measures smemBytesPerCycle: sustained shared-memory read
// bandwidth per SM, on the path the GEMM kernel actually uses.
//
// WHY ldmatrix AND NOT A GENERIC LOAD. The model's shared-memory stage exists to
// describe operands being pulled out of shared memory to feed the tensor cores,
// and the kernel does that with ldmatrix. A vectorised float4 read would measure
// a different and generally higher number, because ldmatrix has its own issue
// cost and its own bank access pattern. Calibrating with float4 and predicting
// ldmatrix would be measuring one thing to model another.
//
// The layout here is the kernel's padded layout from gemm-mma.cuh, for the same
// reason. Bank behaviour is a property of the layout, not just the instruction,
// and the padded layout is what the kernel runs.
//
// EXPECT THIS TO COME IN WELL BELOW 128 B/cycle. The placeholder in the model is
// 32 banks x 4 bytes = 128, which is a bank capacity rather than an achievable
// load/store-unit rate. PLAN.md Finding 7 predicts the measured value lands
// materially lower, and that this is expected physics rather than a broken
// microbenchmark. Do not "fix" a low reading by importing a number from another
// GPU, which is the error anti-pattern 7 warns about.

#include "../kernels/gemm-mma.cuh"
#include "cal-common.cuh"

#include <cstdio>
#include <vector>

namespace {

constexpr int kRowHalves = 64;                    // matches a BK=64 A tile
constexpr int kPaddedRow = kRowHalves + 8;        // gemm-mma.cuh padOffset
constexpr int kRows = 128;
constexpr int kSmemHalves = kRows * kPaddedRow;
constexpr int kSmemBytes = kSmemHalves * int(sizeof(half));

// Independent ldmatrix issues per warp per iteration, so the measurement is
// bandwidth-bound rather than latency-bound. Same reasoning as cal-mma.cu.
constexpr int kIlp = 8;

// One ldmatrix.x4 moves four 8x8 tiles of 16-bit elements out of shared memory.
constexpr double kBytesPerLdmatrixX4 = 4.0 * 8.0 * 8.0 * 2.0;   // 512

constexpr int kIters = 2000;
constexpr int kRepeats = 7;

template <int Warps>
__global__ __launch_bounds__(Warps * 32) void smemBwKernel(
    int iters, uint32_t* sink, long long* cycleOut) {

    __shared__ half smem[kSmemHalves];
    for (int i = int(threadIdx.x); i < kSmemHalves; i += Warps * 32) {
        smem[i] = __float2half(float(i) * 1e-4f);
    }
    __syncthreads();

    const int lane = int(threadIdx.x) % 32;
    const int warp = int(threadIdx.x) / 32;
    const int mi = lane / 8;
    const int rowInTile = lane % 8;

    // Spread the ILP streams across distinct row bands so this measures
    // bandwidth out of the array rather than repeated access to one hot region.
    uint32_t addrs[kIlp];
#pragma unroll
    for (int j = 0; j < kIlp; ++j) {
        const int band = ((warp * kIlp + j) * 16) % (kRows - 16);
        const int row = band + (mi % 2) * 8 + rowInTile;
        const int chunk = mi / 2;
        addrs[j] = ridgebench::smemAddr(
            smem + ridgebench::padOffset<kRowHalves>(row, chunk));
    }

    uint32_t frag[kIlp][4];

    __syncthreads();
    const long long t0 = clock64();

    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < kIlp; ++j) {
            ridgebench::ldmatrixX4(addrs[j], frag[j]);
        }
    }

    const long long t1 = clock64();

    uint32_t s = 0;
#pragma unroll
    for (int j = 0; j < kIlp; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) s ^= frag[j][e];
    if (s == 0xDEADBEEFu) sink[0] = s;

    if (threadIdx.x == 0) cycleOut[blockIdx.x] = t1 - t0;
}

template <int Warps>
double measureBytesPerCycle(int numSMs, uint32_t* dSink, long long* dCycles) {
    smemBwKernel<Warps><<<numSMs, Warps * 32>>>(kIters, dSink, dCycles);
    CAL_CUDA_CHECK(cudaGetLastError());
    CAL_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<long long> cycles(numSMs);
    CAL_CUDA_CHECK(cudaMemcpy(cycles.data(), dCycles, numSMs * sizeof(long long),
                              cudaMemcpyDeviceToHost));
    std::vector<double> asDouble(cycles.begin(), cycles.end());
    const double medianCycles = ridgecal::median(asDouble);

    const double bytes = double(Warps) * kIters * kIlp * kBytesPerLdmatrixX4;
    return bytes / medianCycles;
}

} // namespace

int main() {
    const cudaDeviceProp prop = ridgecal::requireLockedClocks();
    const int numSMs = prop.multiProcessorCount;

    uint32_t* dSink = nullptr;
    long long* dCycles = nullptr;
    CAL_CUDA_CHECK(cudaMalloc(&dSink, sizeof(uint32_t)));
    CAL_CUDA_CHECK(cudaMalloc(&dCycles, numSMs * sizeof(long long)));

    ridgecal::warmUpGpu();

    std::printf("\nldmatrix.x4 read bandwidth from shared memory, padded layout, "
                "%d B/instruction\n", int(kBytesPerLdmatrixX4));
    std::printf("  shared memory per block: %d bytes\n", kSmemBytes);
    std::printf("  %6s %20s\n", "warps", "bytes/cycle/SM");

    double best = 0.0;
    int bestWarps = 0;
    std::vector<double> bestRepeats;

    #define CAL_SWEEP(W)                                                        \
        do {                                                                    \
            std::vector<double> reps;                                           \
            for (int r = 0; r < kRepeats; ++r)                                  \
                reps.push_back(measureBytesPerCycle<W>(numSMs, dSink, dCycles)); \
            const double v = ridgecal::median(reps);                            \
            std::printf("  %6d %20.2f\n", W, v);                                \
            if (v > best) { best = v; bestWarps = W; bestRepeats = reps; }      \
        } while (0)

    CAL_SWEEP(1);
    CAL_SWEEP(2);
    CAL_SWEEP(4);
    CAL_SWEEP(8);
    CAL_SWEEP(16);
    #undef CAL_SWEEP

    std::printf("\nsaturated at %d warps per SM\n", bestWarps);
    const bool stable = ridgecal::reportConstant("smemBytesPerCycle", best,
                                                 "bytes/cycle/SM", bestRepeats);
    std::printf("  %-22s %12.1f %-16s (32 banks x 4 B, a capacity not a rate)\n",
                "theoretical", 128.0, "bytes/cycle/SM");
    std::printf("  %-22s %12.1f %%\n", "achieved vs theoretical",
                100.0 * best / 128.0);
    std::printf("\n  A value well below 128 is the expected outcome, see PLAN.md "
                "Finding 7.\n");

    std::printf("\nCAL_RESULT smemBytesPerCycle %.6f\n", best);
    std::printf("CAL_RESULT smemSaturationWarps %d\n", bestWarps);

    CAL_CUDA_CHECK(cudaFree(dSink));
    CAL_CUDA_CHECK(cudaFree(dCycles));
    return stable ? 0 : 1;
}
