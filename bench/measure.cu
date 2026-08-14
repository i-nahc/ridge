// measure.cu produces the ground-truth measurements Phase 4 validates against.
//
// It reads the pre-registered sweep in data/sweep/phase4-registered.csv, crosses
// every shape with every variant in the kVariants table that divides it, times
// each pair, and writes data/measured/a100.csv. That file is the entire input to
// bench/validate.py, so anything wrong here becomes a model error that nobody can
// attribute.
//
// This is a measurement tool, so most of what follows is about not lying.
//
// WARM FIRST. An idle A100 sits at 210 MHz against a 1410 MHz boost clock, and
// whatever is timed first in a process gets measured during the ramp. That
// artifact was worth about 20% and it corrupted every throughput number this
// project produced before it was found. See PLAN.md Finding 9. Clocks must also
// be locked with nvidia-smi -lgc 1410,1410 before running this.
//
// CANARY. A fixed reference config is re-timed throughout the run. If it drifts,
// the machine changed underneath the sweep and measurements taken early are not
// comparable with measurements taken late. The canary value at the time of each
// measurement is written into every row, so validate.py can see the drift rather
// than inheriting it invisibly.
//
// The sweep is pre-registered and this tool does not choose what to measure. It
// measures what the CSV and the variant table say, and it records every skip so
// a config cannot silently drop out. See PLAN.md anti-pattern 8.

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

// Each timed measurement aims for this much wall clock work, so a small shape
// gets many iterations and a large one gets few. A fixed iteration count would
// either take forever on 8192 cubed or be statistically thin on 512 squared.
constexpr double kTargetSecondsPerMeasurement = 0.4;
constexpr int kMinTimedIters = 20;
constexpr int kMaxTimedIters = 400;
constexpr int kWarmupIters = 10;

// Re-time the canary every this many measurements.
constexpr int kCanaryEvery = 25;

// Drift of the canary across the whole run above which the sweep is not
// internally comparable and should be re-run on a quiet machine.
constexpr double kCanaryMaxDrift = 0.05;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                         cudaGetErrorString(err_), __FILE__, __LINE__);         \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                            \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",                  \
                         int(st_), __FILE__, __LINE__);                         \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

struct Shape {
    int M, N, K;
    std::string tag;
};

// Reads the registered sweep. Lines starting with # are comments. Any parse
// failure is fatal rather than skipped, because a silently dropped shape is a
// narrowed sweep, which is the thing anti-pattern 8 forbids.
std::vector<Shape> loadRegisteredSweep(const char* path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr,
                     "cannot open %s\n"
                     "The sweep is pre-registered and this tool will not invent one.\n",
                     path);
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
        if (!std::getline(ss, m, ',') || !std::getline(ss, n, ',') ||
            !std::getline(ss, k, ',') || !std::getline(ss, tag, ',')) {
            std::fprintf(stderr, "%s:%d: malformed row: %s\n", path, lineNo, line.c_str());
            std::exit(2);
        }
        shapes.push_back({std::atoi(m.c_str()), std::atoi(n.c_str()),
                          std::atoi(k.c_str()), tag});
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

void cublasGemm(cublasHandle_t handle, const half* dA, const half* dB, float* dC,
                int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                              dB, CUDA_R_16F, N, dA, CUDA_R_16F, K, &beta,
                              dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// Drives the GPU to steady clock and thermal state. See PLAN.md Finding 9.
void warmUpGpu(cublasHandle_t handle, const half* dA, const half* dB, float* dC,
               int M, int N, int K, double seconds) {
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
               .count() < seconds) {
        for (int i = 0; i < 10; ++i) cublasGemm(handle, dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// One timed measurement. Runs a short probe to pick an iteration count that
// gives roughly kTargetSecondsPerMeasurement of timed work, then measures.
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
    const double perIterMs = double(probeMs) / 5.0;

    int iters = int(kTargetSecondsPerMeasurement * 1000.0 / std::max(perIterMs, 1e-6));
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
                prop.name, prop.multiProcessorCount, prop.major, prop.minor,
                smClockKhz / 1000.0);
    if (prop.major != 8 || prop.minor != 0) {
        std::printf("WARNING: this sweep targets sm_80. Other architectures are "
                    "not the Phase 4 dataset.\n");
    }

    const std::vector<Shape> shapes = loadRegisteredSweep(sweepPath);
    std::printf("registered sweep: %s, %zu shapes, %d variants\n",
                sweepPath, shapes.size(), ridgebench::numVariants());

    // Prove the output is writable before doing any work.
    //
    // git does not track empty directories, so data/measured/ is absent from a
    // fresh clone and ofstream will not create parents. Discovering that after
    // the warmup and the sweep means throwing away minutes of GPU time that
    // somebody is paying for by the hour. Fail in the first second instead.
    {
        std::error_code ec;
        const std::filesystem::path p(outPath);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
        std::ofstream probe(outPath, std::ios::app);
        if (!probe) {
            std::fprintf(stderr,
                         "cannot write %s\n"
                         "Create the directory first:  mkdir -p %s\n",
                         outPath, p.parent_path().string().c_str());
            return 2;
        }
    }

    // Allocate once for the largest shape and reuse. Repeated allocation of
    // hundreds of megabytes between shapes would fragment and would also put
    // allocator time inside the measured region.
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

    std::printf("warming up to steady clock (%.0fs)...\n", kWarmupSeconds);
    warmUpGpu(handle, dA, dB, dC, 4096, 4096, 4096, kWarmupSeconds);

    // The canary. A fixed config re-timed through the run so drift is visible
    // rather than silently spread across the dataset.
    const double canaryFlops = 2.0 * 4096.0 * 4096.0 * 4096.0;
    auto measureCanary = [&]() {
        return timeTflops([&] { cublasGemm(handle, dA, dB, dC, 4096, 4096, 4096); },
                          canaryFlops);
    };
    const double canaryFirst = measureCanary();
    double canaryNow = canaryFirst;
    double canaryMin = canaryFirst, canaryMax = canaryFirst;
    std::printf("canary (cuBLAS 4096^3) at start: %.2f TFLOP/s\n\n", canaryFirst);

    std::ofstream out(outPath);
    if (!out) {
        std::fprintf(stderr,
                     "cannot write %s\n"
                     "Check the directory exists and is writable:  mkdir -p %s\n",
                     outPath,
                     std::filesystem::path(outPath).parent_path().string().c_str());
        return 2;
    }
    out << "# Ridge Phase 4 ground-truth measurements\n"
        << "# gpu," << prop.name << "\n"
        << "# sms," << prop.multiProcessorCount << "\n"
        << "# nominal_clock_mhz," << smClockKhz / 1000.0 << "\n"
        << "# sweep," << sweepPath << "\n"
        << "# warmup_seconds," << kWarmupSeconds << "\n"
        << "# canary_start_tflops," << canaryFirst << "\n"
        << "# NOTE clocks must be locked (nvidia-smi -lgc 1410,1410), see PLAN.md Finding 9\n"
        << "M,N,K,tag,BM,BN,BK,WM,WN,stages,groupM,regsPerThread,smemBytes,threads,"
           "tflops,cublas_tflops,fraction_of_cublas,canary_tflops\n";

    int measurements = 0, skipped = 0;
    for (const Shape& s : shapes) {
        const double flops = 2.0 * s.M * s.N * s.K;
        const double cublasTflops =
            timeTflops([&] { cublasGemm(handle, dA, dB, dC, s.M, s.N, s.K); }, flops);
        std::printf("%5dx%5dx%5d [%-22s] cuBLAS %7.2f TFLOP/s\n",
                    s.M, s.N, s.K, s.tag.c_str(), cublasTflops);

        for (int v = 0; v < ridgebench::numVariants(); ++v) {
            const ridgebench::KernelVariant& kv = ridgebench::variant(v);
            if (s.M % kv.BM || s.N % kv.BN || s.K % kv.BK) {
                // Recorded, not silently dropped. The registered shapes are all
                // multiples of 256 and 64 so this should never fire; if it does,
                // the sweep has narrowed and that must be visible.
                std::printf("    SKIP variant %d (%dx%dx%d) does not divide this shape\n",
                            v, kv.BM, kv.BN, kv.BK);
                ++skipped;
                continue;
            }

            if (kv.launch(dA, dB, dC, s.M, s.N, s.K, nullptr) != cudaSuccess) {
                std::printf("    SKIP variant %d launch failed\n", v);
                ++skipped;
                continue;
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            const double tflops = timeTflops(
                [&] { kv.launch(dA, dB, dC, s.M, s.N, s.K, nullptr); }, flops);

            if (++measurements % kCanaryEvery == 0) {
                canaryNow = measureCanary();
                canaryMin = std::min(canaryMin, canaryNow);
                canaryMax = std::max(canaryMax, canaryNow);
            }

            out << s.M << ',' << s.N << ',' << s.K << ',' << s.tag << ','
                << kv.BM << ',' << kv.BN << ',' << kv.BK << ','
                << kv.WM << ',' << kv.WN << ',' << kv.stages << ',' << kv.groupM << ','
                << kv.regsPerThread << ',' << kv.smemBytes << ',' << kv.threads << ','
                << tflops << ',' << cublasTflops << ','
                << (tflops / cublasTflops) << ',' << canaryNow << '\n';
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

    std::printf("\nwrote %d measurements to %s (%d skipped)\n",
                measurements, outPath, skipped);
    std::printf("canary: start %.2f, end %.2f, min %.2f, max %.2f, drift %.1f%%\n",
                canaryFirst, canaryLast, canaryMin, canaryMax, canaryDrift * 100.0);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    if (skipped != 0) {
        std::printf("\nFAIL: %d configs were skipped. The measured set no longer "
                    "matches the registered sweep,\nso validate.py will reject it. "
                    "See PLAN.md anti-pattern 8.\n", skipped);
        return 1;
    }
    if (canaryDrift > kCanaryMaxDrift) {
        std::printf("\nFAIL: canary drifted %.1f%% across the run, above the %.0f%% "
                    "limit.\nMeasurements taken early are not comparable with those "
                    "taken late, so this\ndataset is not usable for model validation. "
                    "Lock clocks and re-run on a quiet machine.\n",
                    canaryDrift * 100.0, kCanaryMaxDrift * 100.0);
        return 1;
    }
    std::printf("\nsweep complete and internally consistent.\n");
    return 0;
}
