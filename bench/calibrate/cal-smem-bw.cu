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
// Expect this well below 128 B/cycle. The 32 banks x 4 bytes figure is a bank
// capacity rather than an achievable load/store-unit rate, so a materially lower
// reading is expected physics rather than a broken benchmark. Do not "fix" a low
// reading by importing a number from another GPU.

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

// Bank capacity: 32 banks x 4 bytes. A hard physical ceiling, not a target.
constexpr double kTheoretical = 128.0;

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
    uint32_t s = 0;

    __syncthreads();
    const long long t0 = clock64();

    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < kIlp; ++j) {
            ridgebench::ldmatrixX4(addrs[j], frag[j]);
            // Consume the result INSIDE the loop. This is the whole correctness
            // of the measurement and the first version got it wrong.
            //
            // Without this, frag[] is only read after the timed region, so its
            // registers are dead within the loop and ptxas is free to delete all
            // but the last iteration's loads. asm volatile survives the NVVM
            // frontend but does not stop ptxas doing dead code elimination on an
            // instruction whose destination registers are never read.
            //
            // The symptom was unmistakable once the number was checked against
            // physics: 2022 B/cycle/SM, sixteen times the 128 B/cycle the banks
            // can deliver, and 3.95 ldmatrix per cycle, which is exactly the four
            // instructions per cycle an Ampere SM can issue across its four sub
            // partitions. We had measured the issue rate of instructions that
            // mostly were not there.
            s ^= frag[j][0];
        }
    }

    const long long t1 = clock64();

    if (s == 0xDEADBEEFu) sink[0] = s;

    if (threadIdx.x == 0) cycleOut[blockIdx.x] = t1 - t0;
}

// Control measurement: plain vectorised LDS rather than ldmatrix.
//
// This exists to catch the failure that produced a 2022 B/cycle reading. Generic
// float4 shared memory loads have a well understood ceiling: 32 banks x 4 bytes
// is 128 B/cycle, and a warp's 32 lane float4 load is 512 bytes, so it cannot
// complete in fewer than 4 cycles. If this control ever reports above 128, the
// harness is broken rather than the hardware being interesting, and no reading
// from the ldmatrix path should be believed either.
//
// A benchmark that can only be validated by a human remembering to compare it
// against a datasheet is a benchmark that will eventually be believed when it is
// wrong.
template <int Warps>
__global__ __launch_bounds__(Warps * 32) void ldsControlKernel(
    int iters, uint32_t* sink, long long* cycleOut) {

    // Power of two so the address can be rotated per iteration with a single
    // AND rather than a modulo.
    constexpr int kVecs = 1024;
    __shared__ float4 smem[kVecs];
    for (int i = int(threadIdx.x); i < kVecs; i += Warps * 32) {
        smem[i] = make_float4(float(i), float(i) + 1, float(i) + 2, float(i) + 3);
    }
    __syncthreads();

    // Consecutive lanes read consecutive float4, which is the conflict free
    // pattern and therefore the best case this path can achieve.
    int idx[kIlp];
#pragma unroll
    for (int j = 0; j < kIlp; ++j) idx[j] = (int(threadIdx.x) + j * 32) & (kVecs - 1);

    float acc = 0.0f;
    __syncthreads();
    const long long t0 = clock64();

    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < kIlp; ++j) {
            // The address must depend on the loop counter. Without that, these
            // are loop-invariant loads from an array the loop never writes, so
            // the compiler hoists all eight out of the loop and what remains is
            // 2000 iterations of floating point adds.
            //
            // That is exactly what the first version did, and it reported 1007
            // B/cycle/SM, about 7.9 times the bank capacity. Note that the
            // ldmatrix path above was never vulnerable to this, because asm
            // volatile cannot be hoisted out of a loop. The control was the
            // broken one, which is a reminder that a cross-check needs checking
            // too.
            const float4 v = smem[(idx[j] + i) & (kVecs - 1)];
            acc += v.x + v.y + v.z + v.w;
        }
    }

    const long long t1 = clock64();

    if (acc == 1.2345678e30f) sink[0] = uint32_t(acc);
    if (threadIdx.x == 0) cycleOut[blockIdx.x] = t1 - t0;
}

template <int Warps>
double measureLdsBytesPerCycle(int numSMs, uint32_t* dSink, long long* dCycles) {
    ldsControlKernel<Warps><<<numSMs, Warps * 32>>>(kIters, dSink, dCycles);
    CAL_CUDA_CHECK(cudaGetLastError());
    CAL_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<long long> cycles(numSMs);
    CAL_CUDA_CHECK(cudaMemcpy(cycles.data(), dCycles, numSMs * sizeof(long long),
                              cudaMemcpyDeviceToHost));
    std::vector<double> asDouble(cycles.begin(), cycles.end());
    const double medianCycles = ridgecal::median(asDouble);

    // 32 lanes x 16 bytes per float4 load.
    const double bytes = double(Warps) * kIters * kIlp * 32.0 * 16.0;
    return bytes / medianCycles;
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

    // Control: plain LDS, whose ceiling is not in doubt.
    std::printf("\nLDS float4 control (same structure, known 128 B/cycle ceiling)\n");
    double ldsBest = 0.0;
    #define CAL_LDS(W)                                                          \
        do {                                                                    \
            const double v = measureLdsBytesPerCycle<W>(numSMs, dSink, dCycles); \
            std::printf("  %6d %20.2f\n", W, v);                                \
            if (v > ldsBest) ldsBest = v;                                       \
        } while (0)
    CAL_LDS(4);
    CAL_LDS(8);
    CAL_LDS(16);
    #undef CAL_LDS

    std::printf("\nsaturated at %d warps per SM\n", bestWarps);
    const bool stable = ridgecal::reportConstant("smemBytesPerCycle", best,
                                                 "bytes/cycle/SM", bestRepeats);
    std::printf("  %-22s %12.1f %-16s\n", "LDS control", ldsBest, "bytes/cycle/SM");
    std::printf("  %-22s %12.1f %-16s (32 banks x 4 B, a capacity not a rate)\n",
                "theoretical", kTheoretical, "bytes/cycle/SM");
    std::printf("  %-22s %12.1f %%\n", "achieved vs theoretical",
                100.0 * best / kTheoretical);

    // Physics check. Exceeding the bank capacity is not a surprising result, it
    // is proof the measurement is not measuring shared memory.
    bool physical = true;
    if (best > kTheoretical) {
        std::printf("\n  FAIL: %.1f B/cycle/SM exceeds the %.0f B/cycle the banks can\n"
                    "  deliver. This is not a fast GPU, it is a broken benchmark. The\n"
                    "  usual cause is the loaded values not being consumed inside the\n"
                    "  timed loop, which lets ptxas delete the loads and leaves the\n"
                    "  instruction issue rate being measured instead.\n",
                    best, kTheoretical);
        physical = false;
    }
    if (ldsBest > kTheoretical) {
        std::printf("\n  the LDS control also exceeds theoretical (%.1f B/cycle), so the\n"
                    "  harness itself is wrong and neither number here means anything\n", ldsBest);
        physical = false;
    }
    if (physical) {
        std::printf("\n  a value well below 128 is the expected outcome\n");
    }

    std::printf("\nCAL_RESULT smemBytesPerCycle %.6f\n", best);
    std::printf("CAL_RESULT smemSaturationWarps %d\n", bestWarps);
    std::printf("CAL_RESULT ldsControlBytesPerCycle %.6f\n", ldsBest);

    CAL_CUDA_CHECK(cudaFree(dSink));
    CAL_CUDA_CHECK(cudaFree(dCycles));
    return (stable && physical) ? 0 : 1;
}
