// cal-hbm-bw.cu measures hbmBytesPerSec: sustained HBM read bandwidth for the
// whole GPU.
//
// THE TRAP HERE IS L2. An A100 has 40 MB of L2. A streaming benchmark over a
// buffer that fits, or even nearly fits, in L2 measures L2 bandwidth and reports
// it as HBM. The number comes out several times too high, looks plausible
// because it is merely "fast", and then the model's HBM roofline is wrong in the
// optimistic direction for every memory-bound config. The buffer below is 4 GB,
// two orders of magnitude past L2, so every access misses.
//
// Timed with CUDA events rather than clock64 because this is a device-wide rate,
// not a per-SM one, and the whole GPU has to be saturated for the number to mean
// anything.
//
// Read-only rather than copy. The model's HBM term counts operand traffic into
// shared memory, which is a read stream. A copy benchmark would report a
// read+write figure that does not correspond to anything in docs/MODEL.md
// section 5.

#include "cal-common.cuh"

#include <cstdio>
#include <vector>

namespace {

// Two orders of magnitude beyond the 40 MB L2, so the L2 hit rate is negligible
// and this is genuinely a DRAM measurement.
constexpr size_t kBufferBytes = 4ull * 1024 * 1024 * 1024;
constexpr size_t kFloat4Count = kBufferBytes / sizeof(float4);

constexpr int kRepeats = 7;

// Grid-stride read. Each thread accumulates so nothing is dead, and the
// accumulator is only stored under an impossible condition so the store never
// costs bandwidth.
__global__ void hbmReadKernel(const float4* __restrict__ src, size_t n,
                              float* sink) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const float4 v = src[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    const float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 1.2345678e30f) sink[0] = s;
}

} // namespace

int main() {
    const cudaDeviceProp prop = ridgecal::requireLockedClocks();

    size_t freeBytes = 0, totalBytes = 0;
    CAL_CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    std::printf("device memory: %.1f GB total, %.1f GB free\n",
                totalBytes / 1e9, freeBytes / 1e9);
    if (freeBytes < kBufferBytes + (256ull << 20)) {
        std::fprintf(stderr,
                     "not enough free memory for a %.1f GB buffer.\n"
                     "A smaller buffer risks measuring L2 rather than HBM, so this "
                     "tool will not silently shrink it.\n",
                     kBufferBytes / 1e9);
        return 2;
    }

    float4* dSrc = nullptr;
    float* dSink = nullptr;
    CAL_CUDA_CHECK(cudaMalloc(&dSrc, kBufferBytes));
    CAL_CUDA_CHECK(cudaMalloc(&dSink, sizeof(float)));
    CAL_CUDA_CHECK(cudaMemset(dSrc, 1, kBufferBytes));

    ridgecal::warmUpGpu();

    // Enough blocks to fill every SM several times over, so the memory system
    // rather than the launch configuration is the limit.
    const int blocks = prop.multiProcessorCount * 32;
    const int threads = 256;

    cudaEvent_t start, stop;
    CAL_CUDA_CHECK(cudaEventCreate(&start));
    CAL_CUDA_CHECK(cudaEventCreate(&stop));

    std::printf("\nHBM streaming read, %.2f GB buffer (L2 is %.0f MB), "
                "%d blocks x %d threads\n",
                kBufferBytes / 1e9, prop.l2CacheSize / 1e6, blocks, threads);

    std::vector<double> reps;
    for (int r = 0; r < kRepeats; ++r) {
        // One untimed pass so the first repeat is not paying for anything the
        // later ones are not.
        hbmReadKernel<<<blocks, threads>>>(dSrc, kFloat4Count, dSink);
        CAL_CUDA_CHECK(cudaDeviceSynchronize());

        CAL_CUDA_CHECK(cudaEventRecord(start));
        hbmReadKernel<<<blocks, threads>>>(dSrc, kFloat4Count, dSink);
        CAL_CUDA_CHECK(cudaEventRecord(stop));
        CAL_CUDA_CHECK(cudaEventSynchronize(stop));
        CAL_CUDA_CHECK(cudaGetLastError());

        float ms = 0.0f;
        CAL_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        reps.push_back(double(kBufferBytes) / (double(ms) * 1e-3));
    }

    const double bytesPerSec = ridgecal::median(reps);

    std::printf("  %6s %20s\n", "repeat", "GB/s");
    for (size_t i = 0; i < reps.size(); ++i)
        std::printf("  %6zu %20.1f\n", i, reps[i] / 1e9);

    std::printf("\n");
    const bool stable = ridgecal::reportConstant("hbmBytesPerSec",
                                                 bytesPerSec / 1e9, "GB/s", reps);
    std::printf("  %-22s %12.1f GB/s   (A100 40GB spec, cross-check only)\n",
                "datasheet", 1555.0);
    std::printf("  %-22s %12.1f %%\n", "achieved vs datasheet",
                100.0 * bytesPerSec / 1.555e12);
    std::printf("\n  Roughly 80 to 90%% of spec is normal for a streaming read. "
                "Far above spec means\n  the buffer is being served from L2 and "
                "the measurement is wrong.\n");

    std::printf("\nCAL_RESULT hbmBytesPerSec %.6e\n", bytesPerSec);

    CAL_CUDA_CHECK(cudaEventDestroy(start));
    CAL_CUDA_CHECK(cudaEventDestroy(stop));
    CAL_CUDA_CHECK(cudaFree(dSrc));
    CAL_CUDA_CHECK(cudaFree(dSink));
    return stable ? 0 : 1;
}
