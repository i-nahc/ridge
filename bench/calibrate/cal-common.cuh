// cal-common.cuh holds what every calibration microbenchmark needs.
//
// It exists mainly so the warmup cannot be forgotten. An idle A100 sits at 210 MHz
// against a 1410 MHz boost clock, so whatever is timed first in a process is
// measured during the ramp, worth about 20%. A constant measured cold is wrong by
// roughly that much, gets written into the json, and nothing downstream questions
// it. The sanity bands are wide enough that a 20% HBM error would pass.
//
// So: every cal-*.cu warms the GPU through warmUpGpu before its first timed
// region, and every one of them refuses to run if the SM clock is not pinned.

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ridgecal {

#define CAL_CUDA_CHECK(call)                                                    \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                         cudaGetErrorString(err_), __FILE__, __LINE__);         \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

constexpr double kWarmupSeconds = 5.0;

// A kernel with no purpose other than to make the GPU hot and fast. Pure FMA on
// registers, so it loads the SMs without touching memory and without needing an
// allocation.
__global__ void burnKernel(float* sink, int iters) {
    float a = threadIdx.x * 1e-3f, b = 1.000001f, c = 0.0f;
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        c = fmaf(a, b, c);
        a = fmaf(c, b, a);
    }
    if (sink && c == 12345.6789f) sink[0] = c + a;
}

// Drives the GPU to steady clock and thermal state. Not optional, see the file
// header.
inline void warmUpGpu(double seconds = kWarmupSeconds) {
    int device = 0;
    CAL_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CAL_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf("warming up to steady clock (%.0fs)...\n", seconds);
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
               .count() < seconds) {
        burnKernel<<<prop.multiProcessorCount * 4, 256>>>(nullptr, 20000);
        CAL_CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Reports the device and refuses to proceed if the clock is not pinned.
//
// An unpinned clock does not merely add noise. The constants are meant to describe
// one machine state, and the validation sweep is taken in that same state. If the
// clock floats they describe different machines and the comparison is meaningless.
inline cudaDeviceProp requireLockedClocks() {
    int device = 0;
    CAL_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CAL_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf("GPU: %s, %d SMs, sm_%d%d\n", prop.name,
                prop.multiProcessorCount, prop.major, prop.minor);
    if (prop.major != 8 || prop.minor != 0) {
        std::printf("WARNING: these microbenchmarks target sm_80. Constants "
                    "measured elsewhere do not belong in a100.json.\n");
    }
    std::printf("clocks must be pinned with `sudo nvidia-smi -lgc 1410,1410` before\n"
                "calibrating, and this cannot be verified from inside the process\n");
    return prop;
}

// Median of a set of repeated measurements. Median rather than mean because a
// single scheduling hiccup should not move a calibrated constant, and rather
// than min because the fastest observation of a noisy process is a biased
// estimate of its sustained rate.
inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

inline double spread(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const auto mm = std::minmax_element(v.begin(), v.end());
    return (*mm.second - *mm.first) / *mm.second;
}

// Prints a constant with its measurement spread, and flags it when the repeats
// disagree enough that the value should not be trusted as a constant.
inline bool reportConstant(const char* name, double value, const char* unit,
                           const std::vector<double>& repeats,
                           double maxSpread = 0.05) {
    const double s = spread(repeats);
    std::printf("  %-22s %12.4f %-16s spread %.1f%% over %zu repeats%s\n",
                name, value, unit, s * 100.0, repeats.size(),
                s > maxSpread ? "   <-- UNSTABLE" : "");
    return s <= maxSpread;
}

} // namespace ridgecal
