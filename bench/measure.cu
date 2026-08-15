// Times every registered shape against every kernel variant that divides it and
// writes the result as CSV. This is the ground truth the model is validated
// against, so most of what follows is about not lying.
//
// Clocks must be locked (nvidia-smi -lgc 1410,1410) before running. An idle A100
// sits at 210 MHz against a 1410 MHz boost clock, so whatever is timed first in a
// process is measured during the ramp, which is worth about 20%.
//
// A fixed reference config is re-timed throughout the run. If it drifts, the
// machine changed during the sweep and early measurements are not comparable
// with late ones. Its value at the time of each measurement goes into every row.

#include "kernels/gemm-mma.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kWarmupSeconds = 5.0;

// Each measurement aims for this much timed work, so a small shape gets many
// iterations and a large one gets few. A fixed count would either take forever on
// 8192 cubed or be statistically thin on 512 squared.
constexpr double kTargetSecondsPerMeasurement = 0.4;
constexpr int kMinTimedIters = 20;
constexpr int kMaxTimedIters = 400;
constexpr int kWarmupIters = 10;

constexpr int kCanaryEvery = 25;
constexpr double kCanaryMaxDrift = 0.05;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                            \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(st_), __FILE__, __LINE__); \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

struct Shape {
    int M, N, K;
    std::string tag;
};

// Any parse failure is fatal rather than skipped. A silently dropped shape is a
// narrowed sweep.
std::vector<Shape> loadRegisteredSweep(const char* path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(2);
    }
    std::vector<Shape> shapes;
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string m, n, k, tag;
        if (!std::getline(ss, m, ',') || !std::getline(ss, n, ',') || !std::getline(ss, k, ',') || !std::getline(ss, tag, ',')) {
            std::fprintf(stderr, "%s:%d: malformed row: %s\n", path, lineNo, line.c_str());
            std::exit(2);
        }
        shapes.push_back({std::atoi(m.c_str()), std::atoi(n.c_str()), std::atoi(k.c_str()), tag});
    }
    if (shapes.empty()) {
        std::fprintf(stderr, "%s contains no shapes\n", path);
        std::exit(2);
    }
    return shapes;
}

void fillDeterministic(std::vector<half>& v, unsigned seed) {
    unsigned s = seed;
    for (size_t i = 0; i < v.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = __float2half((float((s >> 16) & 0x7fff) / 32767.0f) - 0.5f);
    }
}

void cublasGemm(cublasHandle_t handle, const half* dA, const half* dB, float* dC, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, N,
                              dA, CUDA_R_16F, K, &beta, dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void warmUpGpu(cublasHandle_t handle, const half* dA, const half* dB, float* dC, int M, int N, int K, double seconds) {
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds) {
        for (int i = 0; i < 10; ++i) cublasGemm(handle, dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Probes for an iteration count that gives roughly the target amount of timed
// work, then measures.
template <typename Fn>
double timeTflops(Fn launch, double flops) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < kWarmupIters; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 5; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float probeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&probeMs, start, stop));

    int iters = int(kTargetSecondsPerMeasurement * 1000.0 / std::max(double(probeMs) / 5.0, 1e-6));
    iters = std::min(std::max(iters, kMinTimedIters), kMaxTimedIters);

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return flops / ((double(ms) / iters) * 1e-3) / 1e12;
}

} // namespace

int main(int argc, char** argv) {
    const char* sweepPath = argc > 1 ? argv[1] : "data/sweep/phase4-registered.csv";
    const char* outPath = argc > 2 ? argv[2] : "data/measured/a100.csv";

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int smClockKhz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smClockKhz, cudaDevAttrClockRate, device));

    std::printf("GPU: %s, %d SMs, sm_%d%d, nominal clock %.0f MHz\n",
                prop.name, prop.multiProcessorCount, prop.major, prop.minor, smClockKhz / 1000.0);
    if (prop.major != 8 || prop.minor != 0) {
        std::printf("WARNING: this sweep targets sm_80.\n");
    }

    const std::vector<Shape> shapes = loadRegisteredSweep(sweepPath);
    std::printf("sweep: %s, %zu shapes, %d variants\n", sweepPath, shapes.size(), ridgebench::numVariants());

    // Prove the output is writable before spending minutes of GPU time on work
    // that cannot be saved. git does not track empty directories, so
    // data/measured/ is absent from a fresh clone.
    {
        std::error_code ec;
        const std::filesystem::path p(outPath);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
        std::ofstream probe(outPath, std::ios::app);
        if (!probe) {
            std::fprintf(stderr, "cannot write %s\n", outPath);
            return 2;
        }
    }

    // Allocate once for the largest shape and reuse, so allocator time stays out
    // of the measured region.
    size_t maxA = 0, maxB = 0, maxC = 0;
    for (const Shape& s : shapes) {
        maxA = std::max(maxA, size_t(s.M) * s.K);
        maxB = std::max(maxB, size_t(s.K) * s.N);
        maxC = std::max(maxC, size_t(s.M) * s.N);
    }
    std::vector<half> hA(maxA), hB(maxB);
    fillDeterministic(hA, 0x1234u);
    fillDeterministic(hB, 0x9abcu);

    half *dA = nullptr, *dB = nullptr;
    float* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, maxA * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB, maxB * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, maxC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), maxA * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), maxB * sizeof(half), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::printf("warmup %.0fs\n", kWarmupSeconds);
    warmUpGpu(handle, dA, dB, dC, 4096, 4096, 4096, kWarmupSeconds);

    const double canaryFlops = 2.0 * 4096.0 * 4096.0 * 4096.0;
    auto measureCanary = [&] { return timeTflops([&] { cublasGemm(handle, dA, dB, dC, 4096, 4096, 4096); }, canaryFlops); };
    const double canaryFirst = measureCanary();
    double canaryNow = canaryFirst;
    double canaryMin = canaryFirst, canaryMax = canaryFirst;
    std::printf("canary (cuBLAS 4096^3) %.2f TFLOP/s\n\n", canaryFirst);

    std::ofstream out(outPath);
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", outPath);
        return 2;
    }
    out << "# gpu," << prop.name << "\n"
        << "# sms," << prop.multiProcessorCount << "\n"
        << "# nominal_clock_mhz," << smClockKhz / 1000.0 << "\n"
        << "# sweep," << sweepPath << "\n"
        << "# warmup_seconds," << kWarmupSeconds << "\n"
        << "# canary_start_tflops," << canaryFirst << "\n"
        << "M,N,K,tag,BM,BN,BK,WM,WN,stages,groupM,regsPerThread,smemBytes,threads,"
           "tflops,cublas_tflops,fraction_of_cublas,canary_tflops\n";

    int measurements = 0, skipped = 0;
    for (const Shape& s : shapes) {
        const double flops = 2.0 * s.M * s.N * s.K;
        const double cublasTflops = timeTflops([&] { cublasGemm(handle, dA, dB, dC, s.M, s.N, s.K); }, flops);
        std::printf("%5dx%5dx%5d  %-22s  cuBLAS %7.2f TFLOP/s\n", s.M, s.N, s.K, s.tag.c_str(), cublasTflops);

        for (int v = 0; v < ridgebench::numVariants(); ++v) {
            const ridgebench::KernelVariant& kv = ridgebench::variant(v);
            if (s.M % kv.BM || s.N % kv.BN || s.K % kv.BK) {
                std::printf("    skip variant %d (%dx%dx%d), does not divide this shape\n", v, kv.BM, kv.BN, kv.BK);
                ++skipped;
                continue;
            }

            if (kv.launch(dA, dB, dC, s.M, s.N, s.K, nullptr) != cudaSuccess) {
                std::printf("    skip variant %d, launch failed\n", v);
                ++skipped;
                continue;
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            const double tflops = timeTflops([&] { kv.launch(dA, dB, dC, s.M, s.N, s.K, nullptr); }, flops);

            if (++measurements % kCanaryEvery == 0) {
                canaryNow = measureCanary();
                canaryMin = std::min(canaryMin, canaryNow);
                canaryMax = std::max(canaryMax, canaryNow);
            }

            out << s.M << ',' << s.N << ',' << s.K << ',' << s.tag << ','
                << kv.BM << ',' << kv.BN << ',' << kv.BK << ','
                << kv.WM << ',' << kv.WN << ',' << kv.stages << ',' << kv.groupM << ','
                << kv.regsPerThread << ',' << kv.smemBytes << ',' << kv.threads << ','
                << tflops << ',' << cublasTflops << ',' << (tflops / cublasTflops) << ',' << canaryNow << '\n';
        }
    }

    const double canaryLast = measureCanary();
    canaryMin = std::min(canaryMin, canaryLast);
    canaryMax = std::max(canaryMax, canaryLast);
    const double canaryDrift = (canaryMax - canaryMin) / canaryMax;

    out << "# canary_end_tflops," << canaryLast << "\n"
        << "# canary_min_tflops," << canaryMin << "\n"
        << "# canary_max_tflops," << canaryMax << "\n"
        << "# canary_drift_fraction," << canaryDrift << "\n";
    out.close();

    std::printf("\nwrote %d measurements to %s, %d skipped\n", measurements, outPath, skipped);
    std::printf("canary: start %.2f, end %.2f, min %.2f, max %.2f, drift %.1f%%\n",
                canaryFirst, canaryLast, canaryMin, canaryMax, canaryDrift * 100.0);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    if (skipped != 0) {
        std::printf("\n%d configs skipped, so the measured set no longer matches the sweep\n", skipped);
        return 1;
    }
    if (canaryDrift > kCanaryMaxDrift) {
        std::printf("\ncanary drifted %.1f%%, above the %.0f%% limit, so early and late measurements are not comparable\n",
                    canaryDrift * 100.0, kCanaryMaxDrift * 100.0);
        return 1;
    }
    return 0;
}
