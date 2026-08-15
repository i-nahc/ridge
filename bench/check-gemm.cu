// Correctness and performance check for the GEMM kernel.
//
// Correctness is judged against a float64 CPU reference on shapes small enough
// to afford it, and against cuBLAS on the rest. Two criteria must both hold, a
// relative Frobenius norm and a worst-element error, because each misses a class
// of error the other catches.
//
// Part 0 checks the checker: it feeds the comparison deliberately corrupted
// results and asserts every one is rejected. A correctness check that cannot
// fail says nothing when it passes.
//
// Exit code 0 means everything passed.

#include "kernels/gemm-mma.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// FP16 carries about 3 decimal digits (2^-11 is 4.9e-4). Products accumulate in
// FP32, so input rounding dominates and the error grows roughly with sqrt(K),
// putting a K=4096 dot product near 1e-2 in the worst case. A real kernel bug
// fails this by orders of magnitude rather than marginally.
constexpr double kRelTol = 1e-2;

// Largest tolerated single-element error, scaled by the matrix maximum. Looser
// than the norm, but tight enough that the handful of badly wrong outputs an
// addressing or pipeline-tail bug produces cannot hide.
constexpr double kMaxElemTol = 5e-2;

constexpr double kMinCublasFraction = 0.70;

// Seconds of sustained load before timing. An idle A100 sits at 210 MHz against
// a 1410 MHz boost clock, and the correctness section above is CPU-heavy, so
// without this the first timed workload is measured during the ramp. That alone
// moved a cuBLAS reading from 207 to 256 TFLOP/s.
constexpr double kWarmupSeconds = 5.0;

constexpr int kWarmupIters = 20;
constexpr int kTimedIters = 100;

// A100 dense FP16 tensor peak from the datasheet, used only as an absolute
// anchor for the sanity checks below.
constexpr double kA100PeakTflops = 312.0;

// cuBLAS should land near this fraction of datasheet peak on a healthy,
// clock-locked card. Materially below means cuBLAS is being measured badly,
// which inflates our ratio without the kernel improving.
constexpr double kCublasExpectedPeakFraction = 0.70;

// Drift between the cuBLAS readings taken before and after the sweep, past which
// the machine changed during the run.
constexpr double kCublasMaxDrift = 0.10;

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

struct Shape { int M, N, K; };

// Deliberately not all large and square. The pipeline tail is the most likely
// place for the kernel to be silently wrong, and a sweep of only large aligned K
// would never exercise it. Every K is a multiple of the largest BK in the
// variant table.
const Shape kCorrectnessShapes[] = {
    {256,  256,   64},
    {256,  512,  128},
    {512,  256,  192},
    {1024, 1024,  64},
    {1024, 512, 1024},
    {2048, 2048, 2048},
};

// Ground truth in float64 on the CPU. Slow on purpose: a different precision and
// a different algorithm is what makes it worth comparing against.
void referenceGemm(const std::vector<half>& A, const std::vector<half>& B, std::vector<double>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += double(__half2float(A[size_t(i) * K + k])) * double(__half2float(B[size_t(k) * N + j]));
            }
            C[size_t(i) * N + j] = acc;
        }
    }
}

struct ErrStats {
    double frob;    // relative Frobenius norm over the whole matrix
    double maxRel;  // largest single-element error, scaled by the matrix maximum
};

// The norm catches systematic error but is insensitive to sparse error: a
// fraction f of elements wrong by r moves it only about sqrt(f)*r. maxRel covers
// that case, and is scaled by the largest reference magnitude rather than
// per-element so a near-zero element that suffered cancellation cannot produce a
// false positive. A kernel has to pass both.
template <typename RefT>
ErrStats errorStats(const std::vector<float>& got, const std::vector<RefT>& ref) {
    double num = 0.0, den = 0.0, maxDiff = 0.0, maxRef = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = double(ref[i]);
        const double d = double(got[i]) - r;
        num += d * d;
        den += r * r;
        maxDiff = std::max(maxDiff, std::fabs(d));
        maxRef = std::max(maxRef, std::fabs(r));
    }
    ErrStats s;
    s.frob = den > 0.0 ? std::sqrt(num) / std::sqrt(den) : (num == 0.0 ? 0.0 : 1.0);
    s.maxRel = maxRef > 0.0 ? maxDiff / maxRef : (maxDiff == 0.0 ? 0.0 : 1.0);
    return s;
}

// Values kept small and centred so the FP32 accumulator does not lose precision
// to magnitude, which would confuse a real failure with an expected numerical one.
void fillDeterministic(std::vector<half>& v, unsigned seed) {
    unsigned s = seed;
    for (size_t i = 0; i < v.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = __float2half((float((s >> 16) & 0x7fff) / 32767.0f) - 0.5f);
    }
}

// C = A*B with everything row-major. cuBLAS is column-major, so computing
// C^T = B^T * A^T by swapping the operands yields row-major C without transposes.
void cublasReference(cublasHandle_t handle, const half* dA, const half* dB, float* dC, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, N,
                              dA, CUDA_R_16F, K, &beta, dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void warmUpGpu(cublasHandle_t handle, const half* A, const half* B, float* C, int M, int N, int K, double seconds) {
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds) {
        for (int i = 0; i < 10; ++i) cublasReference(handle, A, B, C, M, N, K);
        cudaDeviceSynchronize();
    }
}

double timeMs(void (*body)(void*), void* ctx) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < kWarmupIters; ++i) body(ctx);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kTimedIters; ++i) body(ctx);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return double(ms) / kTimedIters;
}

} // namespace

int main() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU: %s, %d SMs, sm_%d%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    if (prop.major != 8 || prop.minor != 0) {
        std::printf("WARNING: this kernel targets sm_80.\n");
    }
    std::printf("tolerance: %.1e relative Frobenius, %.1e max element\n", kRelTol, kMaxElemTol);
    std::printf("floor: %.0f%% of cuBLAS\n\n", kMinCublasFraction * 100.0);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int failures = 0;
    const int nVariants = ridgebench::numVariants();

    // Prove the checker can fail before trusting anything it passes. The three
    // corruptions are the shapes real kernel bugs take, not arbitrary noise.
    std::printf("=== checker self-test ===\n");
    {
        const size_t n = 4096;
        std::vector<double> ref(n);
        unsigned s = 0xc0ffeeu;
        for (size_t i = 0; i < n; ++i) {
            s = s * 1664525u + 1013904223u;
            ref[i] = (double((s >> 16) & 0x7fff) / 32767.0) - 0.5;
        }

        auto expectRejected = [&](const char* name, const std::vector<float>& bad) {
            const ErrStats e = errorStats(bad, ref);
            const bool rejected = (e.frob > kRelTol) || (e.maxRel > kMaxElemTol);
            std::printf("  %-38s frob %.3e  maxRel %.3e  %s\n", name, e.frob, e.maxRel, rejected ? "rejected" : "ACCEPTED");
            if (!rejected) failures++;
        };

        // A clean copy has to pass, or the three rejections below mean nothing.
        {
            std::vector<float> clean(n);
            for (size_t i = 0; i < n; ++i) clean[i] = float(ref[i]);
            const ErrStats e = errorStats(clean, ref);
            const bool accepted = (e.frob <= kRelTol) && (e.maxRel <= kMaxElemTol);
            std::printf("  %-38s frob %.3e  maxRel %.3e  %s\n", "exact copy", e.frob, e.maxRel, accepted ? "accepted" : "REJECTED");
            if (!accepted) failures++;
        }

        // A wrong accumulation count or a missed K-step.
        {
            std::vector<float> bad(n);
            for (size_t i = 0; i < n; ++i) bad[i] = float(ref[i] * 1.05);
            expectRejected("systematic 5% scale", bad);
        }

        // A pipeline tail bug, and the case the norm check is weakest against.
        {
            std::vector<float> bad(n);
            for (size_t i = 0; i < n; ++i) bad[i] = float(ref[i]);
            for (size_t i = size_t(n * 0.97); i < n; ++i) bad[i] = float(ref[i] * 0.5);
            expectRejected("last 3% of outputs halved", bad);
        }

        // A swizzle or ldmatrix addressing mistake at low volume, which the norm
        // check alone would miss.
        {
            std::vector<float> bad(n);
            for (size_t i = 0; i < n; ++i) bad[i] = float(ref[i]);
            bad[n / 3] = float(ref[n / 3] + 1.0);
            expectRejected("one element off by a full unit", bad);
        }

        if (failures != 0) {
            std::printf("\n  the checker does not reliably reject wrong results\n");
            return 1;
        }
        std::printf("  passed\n\n");
    }

    std::printf("=== correctness ===\n");
    for (const Shape& s : kCorrectnessShapes) {
        const size_t szA = size_t(s.M) * s.K, szB = size_t(s.K) * s.N, szC = size_t(s.M) * s.N;

        std::vector<half> hA(szA), hB(szB);
        fillDeterministic(hA, 0x1234u + unsigned(s.M));
        fillDeterministic(hB, 0x9abcu + unsigned(s.N));

        half *dA = nullptr, *dB = nullptr;
        float* dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, szA * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dB, szB * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dC, szC * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * sizeof(half), cudaMemcpyHostToDevice));

        // float64 where it is affordable, cuBLAS where it is not. The cutoff is
        // how long we are willing to wait, not how much each is trusted.
        const bool useCpuReference = szC * size_t(s.K) <= (1ull << 32);
        std::vector<double> refCpu;
        std::vector<float> refGpu(szC);
        if (useCpuReference) {
            refCpu.resize(szC);
            referenceGemm(hA, hB, refCpu, s.M, s.N, s.K);
        }
        cublasReference(handle, dA, dB, dC, s.M, s.N, s.K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(refGpu.data(), dC, szC * sizeof(float), cudaMemcpyDeviceToHost));

        // If both references exist, confirm they agree first. A disagreement here
        // is a harness fault that would otherwise be blamed on the kernel.
        if (useCpuReference) {
            const ErrStats refErr = errorStats(refGpu, refCpu);
            if (refErr.frob > kRelTol || refErr.maxRel > kMaxElemTol) {
                std::printf("  cuBLAS disagrees with the float64 reference on %dx%dx%d: frob %.3e maxRel %.3e\n",
                            s.M, s.N, s.K, refErr.frob, refErr.maxRel);
                failures++;
            }
        }

        for (int v = 0; v < nVariants; ++v) {
            const ridgebench::KernelVariant& kv = ridgebench::variant(v);
            if (s.M % kv.BM || s.N % kv.BN || s.K % kv.BK) continue;

            CUDA_CHECK(cudaMemset(dC, 0, szC * sizeof(float)));
            const cudaError_t launchErr = kv.launch(dA, dB, dC, s.M, s.N, s.K, nullptr);
            if (launchErr != cudaSuccess) {
                std::printf("  FAIL %dx%dx%d tile %dx%dx%d: %s\n", s.M, s.N, s.K, kv.BM, kv.BN, kv.BK, cudaGetErrorString(launchErr));
                failures++;
                continue;
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> got(szC);
            CUDA_CHECK(cudaMemcpy(got.data(), dC, szC * sizeof(float), cudaMemcpyDeviceToHost));

            const ErrStats err = useCpuReference ? errorStats(got, refCpu) : errorStats(got, refGpu);
            const bool ok = (err.frob <= kRelTol) && (err.maxRel <= kMaxElemTol);
            if (!ok) failures++;
            std::printf("  %-4s %5dx%5dx%5d  tile %3dx%3dx%2d s%d w%2dx%2d  frob %.3e  maxRel %.3e  (%s)\n",
                        ok ? "ok" : "FAIL", s.M, s.N, s.K, kv.BM, kv.BN, kv.BK, kv.stages, kv.WM, kv.WN,
                        err.frob, err.maxRel, useCpuReference ? "f64" : "cuBLAS");
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    std::printf("\n=== throughput vs cuBLAS, 4096x4096x4096 ===\n");
    {
        const int M = 4096, N = 4096, K = 4096;
        const size_t szA = size_t(M) * K, szB = size_t(K) * N, szC = size_t(M) * N;
        const double flops = 2.0 * M * N * K;

        std::vector<half> hA(szA), hB(szB);
        fillDeterministic(hA, 0x55aau);
        fillDeterministic(hB, 0xaa55u);

        half *dA = nullptr, *dB = nullptr;
        float* dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, szA * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dB, szB * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dC, szC * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * sizeof(half), cudaMemcpyHostToDevice));

        std::printf("  warmup %.0fs\n", kWarmupSeconds);
        warmUpGpu(handle, dA, dB, dC, M, N, K, kWarmupSeconds);

        struct CublasCtx { cublasHandle_t h; const half* a; const half* b; float* c; int m, n, k; };
        CublasCtx cctx{handle, dA, dB, dC, M, N, K};
        auto timeCublas = [&] {
            return timeMs([](void* p) { CublasCtx* c = static_cast<CublasCtx*>(p); cublasReference(c->h, c->a, c->b, c->c, c->m, c->n, c->k); }, &cctx);
        };

        const double cublasTflopsBefore = flops / (timeCublas() * 1e-3) / 1e12;
        std::printf("  cuBLAS before   %8.2f TFLOP/s   %5.1f%% of %.0f peak\n",
                    cublasTflopsBefore, 100.0 * cublasTflopsBefore / kA100PeakTflops, kA100PeakTflops);

        std::vector<std::pair<int, double>> results;
        double bestTflops = 0.0;
        int bestIdx = -1;
        for (int v = 0; v < nVariants; ++v) {
            const ridgebench::KernelVariant& kv = ridgebench::variant(v);
            if (M % kv.BM || N % kv.BN || K % kv.BK) continue;

            struct KCtx { const ridgebench::KernelVariant* kv; const half* a; const half* b; float* c; int m, n, k; };
            KCtx kctx{&kv, dA, dB, dC, M, N, K};
            if (kv.launch(dA, dB, dC, M, N, K, nullptr) != cudaSuccess) {
                std::printf("  tile %3dx%3dx%2d s%d  launch failed, skipped\n", kv.BM, kv.BN, kv.BK, kv.stages);
                continue;
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            const double ms = timeMs([](void* p) { KCtx* c = static_cast<KCtx*>(p); c->kv->launch(c->a, c->b, c->c, c->m, c->n, c->k, nullptr); }, &kctx);
            const double tflops = flops / (ms * 1e-3) / 1e12;
            if (tflops > bestTflops) { bestTflops = tflops; bestIdx = v; }
            results.push_back({v, tflops});
        }

        // Re-measure cuBLAS at the end. If it degrades partway through, every
        // fraction printed afterwards is inflated by a change that has nothing to
        // do with the kernel.
        const double cublasTflopsAfter = flops / (timeCublas() * 1e-3) / 1e12;
        std::printf("  cuBLAS after    %8.2f TFLOP/s   %5.1f%% of %.0f peak\n\n",
                    cublasTflopsAfter, 100.0 * cublasTflopsAfter / kA100PeakTflops, kA100PeakTflops);

        // Take the best cuBLAS reading as the denominator, so a slow reading
        // cannot inflate the fraction.
        const double cublasTflops = std::max(cublasTflopsBefore, cublasTflopsAfter);
        const double drift = std::fabs(cublasTflopsAfter - cublasTflopsBefore) / cublasTflopsBefore;

        if (drift > kCublasMaxDrift) {
            std::printf("  FAIL: cuBLAS drifted %.1f%% across the run. Check that clocks are locked\n"
                        "  (nvidia-smi -lgc 1410,1410) and that nothing else is using the GPU.\n\n", drift * 100.0);
            failures++;
        } else if (drift > 0.0) {
            std::printf("  cuBLAS drift %.1f%%\n\n", drift * 100.0);
        }
        if (cublasTflops < kA100PeakTflops * kCublasExpectedPeakFraction) {
            std::printf("  FAIL: cuBLAS reached only %.1f%% of datasheet peak against an expected %.0f%%.\n"
                        "  A slow cuBLAS inflates the fraction below without the kernel improving.\n\n",
                        100.0 * cublasTflops / kA100PeakTflops, kCublasExpectedPeakFraction * 100.0);
            failures++;
        }

        for (const auto& r : results) {
            const ridgebench::KernelVariant& kv = ridgebench::variant(r.first);
            std::printf("  tile %3dx%3dx%2d s%d w%2dx%2d g%-2d  %8.2f TFLOP/s  %5.1f%% of cuBLAS\n",
                        kv.BM, kv.BN, kv.BK, kv.stages, kv.WM, kv.WN, kv.groupM, r.second, 100.0 * r.second / cublasTflops);
        }

        const double fraction = cublasTflops > 0.0 ? bestTflops / cublasTflops : 0.0;
        std::printf("\n  best %.2f TFLOP/s, %.1f%% of cuBLAS", bestTflops, fraction * 100.0);
        if (bestIdx >= 0) {
            const ridgebench::KernelVariant& kv = ridgebench::variant(bestIdx);
            std::printf("  (tile %dx%dx%d s%d w%dx%d g%d)", kv.BM, kv.BN, kv.BK, kv.stages, kv.WM, kv.WN, kv.groupM);
        }
        std::printf("\n");

        if (fraction < kMinCublasFraction) {
            std::printf("  below the %.0f%% floor\n", kMinCublasFraction * 100.0);
            failures++;
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    if (failures != 0) {
        std::printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
