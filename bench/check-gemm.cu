// check-gemm.cu is the Phase 2 acceptance gate.
//
// It decides, by exit code, whether the ground truth kernel is good enough to
// validate a model against. Two questions, both of which must pass:
//
//   1. Is it correct? Checked against a float64 CPU reference on small shapes,
//      where the reference is unambiguous, and against cuBLAS on large shapes,
//      where a CPU reference would take too long.
//   2. Is it fast enough? At least 70% of cuBLAS at its best config, so that
//      "the model predicts this kernel accurately" is a statement about a real
//      kernel and not about a slow one.
//
// Exit code 0 means both passed. Anything else means the gate failed. Nothing
// here should be judged by reading the output, see the acceptance gate rules in
// PLAN.md.
//
// A NOTE ON "BIT-EXACT". PLAN.md and SPEC describe this check as bit-exact
// agreement with cuBLAS. That criterion is not satisfiable and asking for it
// would guarantee the gate either blocks forever or gets quietly weakened later,
// which is its own anti-pattern. cuBLAS picks its own tile shapes, split-K and
// accumulation order, so its FP32 accumulation of FP16 products rounds
// differently from ours on the same input. Two correct implementations disagree
// in the last bits by construction.
//
// What this file does instead is stricter in the way that matters. Correctness
// is judged against a float64 CPU reference, which is a real ground truth rather
// than a second implementation with its own error, and cuBLAS is used as a
// cross-check where the CPU reference is impractical. Two criteria must both
// hold, a Frobenius norm error and a worst-element error, because either one
// alone has a blind spot the other covers. See kRelTol and kMaxElemTol.
//
// PART 0 IS THE POINT. Before checking the kernel at all, the harness checks
// itself: it feeds the comparison deliberately corrupted results and asserts
// every one is rejected. A correctness check that cannot fail is decoration, and
// "it passed" only means something if the check demonstrably catches wrong
// answers. You do not have to understand the kernel to read part 0 and see
// whether the gate has teeth. That is deliberate.

#include "kernels/gemm-mma.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// Relative Frobenius-norm tolerance for the correctness check.
//
// Chosen rather than guessed. FP16 has about 3 decimal digits of mantissa
// (2^-11 is about 4.9e-4). Products accumulate in FP32, so the dominant error is
// the input rounding, and over a K-length dot product the errors partially
// cancel and grow roughly with sqrt(K). For K = 4096 that puts the expected
// relative error near 4.9e-4 * sqrt(4096) / sqrt(3), which is on the order of
// 1e-2 in the worst case and well under it in practice.
//
// A norm-relative check is used rather than max element-wise, because a single
// output element can suffer near-total cancellation and show a huge relative
// error while the result as a whole is fine. The norm check is the standard
// criterion for GEMM validation and it does not have that false-positive mode.
//
// If a kernel bug exists, it fails this by orders of magnitude, not marginally.
// A result that sits just under the threshold deserves suspicion, not a pass, so
// the actual measured value is always printed.
constexpr double kRelTol = 1e-2;

// Largest tolerated single-element error, scaled by the matrix maximum. Looser
// than the norm tolerance because one element is allowed to be worse than the
// aggregate, but tight enough that a handful of badly wrong outputs, which is
// what an addressing or pipeline-tail bug produces, cannot hide.
constexpr double kMaxElemTol = 5e-2;

// Fraction of cuBLAS the best config must reach. Below this the kernel is not a
// fair stand-in for a tuned kernel and any later accuracy claim is meaningless.
constexpr double kMinCublasFraction = 0.70;

constexpr int kWarmupIters = 20;
constexpr int kTimedIters = 100;

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
    const char* note;
};

// The correctness shapes. Deliberately not all large and square.
//
// The small non-square and short-K entries are here because of PLAN.md Finding
// 5: the pipeline has no prologue or epilogue term and its tail is the most
// likely place for the kernel to be silently wrong. A sweep of only large
// aligned K would never exercise the tail and would let a real bug through.
// Every K below is a multiple of the largest BK in the variant table, which is
// the current v1 alignment constraint.
const Shape kCorrectnessShapes[] = {
    {256,  256,   64, "small square, K is one pipeline fill"},
    {256,  512,  128, "non-square, very short K, exercises the tail"},
    {512,  256,  192, "non-square the other way, K not a power of two"},
    {1024, 1024,  64, "wide and shallow, tail dominates"},
    {1024, 512,  1024, "moderate, both dimensions unequal"},
    {2048, 2048, 2048, "large square, steady state dominates"},
};

// Reference GEMM in float64 on the CPU. Slow on purpose. This is the ground
// truth for the small shapes, and being a different precision and a different
// algorithm is exactly what makes it worth comparing against.
void referenceGemm(const std::vector<half>& A, const std::vector<half>& B,
                   std::vector<double>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += double(__half2float(A[size_t(i) * K + k])) *
                       double(__half2float(B[size_t(k) * N + j]));
            }
            C[size_t(i) * N + j] = acc;
        }
    }
}

struct ErrStats {
    double frob;    // relative Frobenius norm error over the whole matrix
    double maxRel;  // largest single-element error, scaled by the matrix maximum
};

// Two criteria, because one is not enough and the reason is worth knowing.
//
// The Frobenius norm catches systematic errors, where many elements are a bit
// wrong. It is insensitive to sparse errors: if a fraction f of elements are
// wrong by relative amount r, the norm error is only about sqrt(f) * r, so a few
// badly wrong elements can hide under the threshold.
//
// maxRel catches exactly that case. It is scaled by the largest reference
// magnitude rather than by each element, because dividing by a near-zero element
// that suffered cancellation produces a huge relative error for a result that is
// completely fine. That false positive is why a naive max-relative check is a
// bad idea and this one is not.
//
// A kernel has to pass both.
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

// Deterministic pseudo-random fill. Values are kept small and centred so that
// the FP32 accumulator does not lose precision to magnitude, which would confuse
// a correctness failure with an expected numerical one.
void fillDeterministic(std::vector<half>& v, unsigned seed) {
    unsigned s = seed;
    for (size_t i = 0; i < v.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        const float f = (float((s >> 16) & 0x7fff) / 32767.0f) - 0.5f;
        v[i] = __float2half(f);
    }
}

// C = A * B with A row-major MxK, B row-major KxN, C row-major MxN, FP16 in and
// FP32 accumulate. cuBLAS is column-major, so we compute C^T = B^T * A^T by
// swapping the operands, which yields row-major C without any transposes.
void cublasReference(cublasHandle_t handle, const half* dA, const half* dB,
                     float* dC, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              dB, CUDA_R_16F, N,
                              dA, CUDA_R_16F, K,
                              &beta,
                              dC, CUDA_R_32F, N,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
    std::printf("GPU: %s, %d SMs, compute capability %d.%d\n",
                prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    if (prop.major != 8 || prop.minor != 0) {
        std::printf("WARNING: this kernel targets sm_80. Results on other "
                    "architectures are not the Phase 2 gate.\n");
    }
    std::printf("Correctness tolerance (relative Frobenius): %.1e\n", kRelTol);
    std::printf("Performance floor: %.0f%% of cuBLAS\n\n",
                kMinCublasFraction * 100.0);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int failures = 0;
    const int nVariants = ridgebench::numVariants();

    // ------------------------------------------------------------------
    // Part 0: prove the checker can fail
    //
    // A passing correctness check is worth nothing unless the check is capable
    // of failing. This runs the comparison against deliberately corrupted
    // results and asserts every one is rejected. If any injected bug slips
    // through, the tolerances are too loose and the whole gate is decoration, so
    // this runs first and a failure here stops everything.
    //
    // The three patterns are the shapes real kernel bugs actually take, not
    // arbitrary noise. Anyone can read this section and see the gate has teeth
    // without needing to understand the kernel itself.
    // ------------------------------------------------------------------
    std::printf("=== part 0: checker self-test ===\n");
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
            std::printf("  %-38s frob %.3e  maxRel %.3e  -> %s\n",
                        name, e.frob, e.maxRel, rejected ? "rejected (good)" : "ACCEPTED (BAD)");
            if (!rejected) failures++;
        };

        // A clean copy must pass, or the checker rejects everything and the
        // other three results below would be meaningless.
        {
            std::vector<float> clean(n);
            for (size_t i = 0; i < n; ++i) clean[i] = float(ref[i]);
            const ErrStats e = errorStats(clean, ref);
            const bool accepted = (e.frob <= kRelTol) && (e.maxRel <= kMaxElemTol);
            std::printf("  %-38s frob %.3e  maxRel %.3e  -> %s\n",
                        "exact copy (control)", e.frob, e.maxRel,
                        accepted ? "accepted (good)" : "REJECTED (BAD)");
            if (!accepted) failures++;
        }

        // Bug 1: everything slightly scaled. What a wrong accumulation count or
        // a missed K-step looks like.
        {
            std::vector<float> bad(n);
            for (size_t i = 0; i < n; ++i) bad[i] = float(ref[i] * 1.05);
            expectRejected("systematic 5% scale error", bad);
        }

        // Bug 2: the last 3% of outputs wrong. This is the shape a pipeline tail
        // bug takes, which PLAN.md Finding 5 identifies as this kernel's most
        // likely silent failure. It is also the case the norm check is weakest
        // against, so it is the one most worth proving.
        {
            std::vector<float> bad(n);
            for (size_t i = 0; i < n; ++i) bad[i] = float(ref[i]);
            for (size_t i = size_t(n * 0.97); i < n; ++i) bad[i] = float(ref[i] * 0.5);
            expectRejected("last 3% of outputs halved (tail bug)", bad);
        }

        // Bug 3: a single element badly wrong. What a swizzle or ldmatrix
        // addressing mistake produces at low volume. The norm check alone would
        // miss this, which is why kMaxElemTol exists.
        {
            std::vector<float> bad(n);
            for (size_t i = 0; i < n; ++i) bad[i] = float(ref[i]);
            bad[n / 3] = float(ref[n / 3] + 1.0);
            expectRejected("one element off by a full unit", bad);
        }

        if (failures != 0) {
            std::printf("\n  The checker cannot reliably detect wrong results. "
                        "Fix the tolerances before trusting any pass below.\n");
            std::printf("\n=== Phase 2 gate: FAIL (self-test) ===\n");
            return 1;
        }
        std::printf("  self-test passed, the checker rejects known-bad results\n\n");
    }

    // ------------------------------------------------------------------
    // Part 1: correctness
    // ------------------------------------------------------------------
    std::printf("=== correctness ===\n");
    for (const Shape& s : kCorrectnessShapes) {
        const size_t szA = size_t(s.M) * s.K, szB = size_t(s.K) * s.N;
        const size_t szC = size_t(s.M) * s.N;

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

        // Ground truth. Use the float64 CPU reference where it is affordable,
        // and cuBLAS where it is not. The cutoff is about how long we are
        // willing to wait, not about how much we trust each one.
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

        // If both references are available, confirm they agree before trusting
        // either. A disagreement here means the harness is wrong, not the
        // kernel, and it would otherwise be misattributed.
        if (useCpuReference) {
            const ErrStats refErr = errorStats(refGpu, refCpu);
            if (refErr.frob > kRelTol || refErr.maxRel > kMaxElemTol) {
                std::printf("  HARNESS ERROR: cuBLAS disagrees with the float64 "
                            "reference on %dx%dx%d (frob %.3e, maxRel %.3e). Fix "
                            "the harness before judging the kernel.\n",
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
                std::printf("  FAIL %dx%dx%d tile %dx%dx%d: launch error %s\n",
                            s.M, s.N, s.K, kv.BM, kv.BN, kv.BK,
                            cudaGetErrorString(launchErr));
                failures++;
                continue;
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> got(szC);
            CUDA_CHECK(cudaMemcpy(got.data(), dC, szC * sizeof(float), cudaMemcpyDeviceToHost));

            const ErrStats err = useCpuReference ? errorStats(got, refCpu)
                                                 : errorStats(got, refGpu);
            const bool ok = (err.frob <= kRelTol) && (err.maxRel <= kMaxElemTol);
            if (!ok) failures++;
            std::printf("  %-4s %5dx%5dx%5d tile %3dx%3dx%2d  frob %.3e  maxRel %.3e  (%s ref)  %s\n",
                        ok ? "ok" : "FAIL", s.M, s.N, s.K, kv.BM, kv.BN, kv.BK,
                        err.frob, err.maxRel, useCpuReference ? "f64" : "cuBLAS", s.note);
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    // ------------------------------------------------------------------
    // Part 2: performance against cuBLAS
    // ------------------------------------------------------------------
    std::printf("\n=== performance vs cuBLAS (4096x4096x4096) ===\n");
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

        struct CublasCtx { cublasHandle_t h; const half* a; const half* b; float* c; int m, n, k; };
        CublasCtx cctx{handle, dA, dB, dC, M, N, K};
        const double cublasMs = timeMs([](void* p) {
            CublasCtx* c = static_cast<CublasCtx*>(p);
            cublasReference(c->h, c->a, c->b, c->c, c->m, c->n, c->k);
        }, &cctx);
        const double cublasTflops = flops / (cublasMs * 1e-3) / 1e12;
        std::printf("  cuBLAS                            %8.2f TFLOP/s\n", cublasTflops);

        double bestTflops = 0.0;
        int bestIdx = -1;
        for (int v = 0; v < nVariants; ++v) {
            const ridgebench::KernelVariant& kv = ridgebench::variant(v);
            if (M % kv.BM || N % kv.BN || K % kv.BK) continue;

            struct KCtx { const ridgebench::KernelVariant* kv; const half* a; const half* b; float* c; int m, n, k; };
            KCtx kctx{&kv, dA, dB, dC, M, N, K};
            if (kv.launch(dA, dB, dC, M, N, K, nullptr) != cudaSuccess) {
                std::printf("  tile %3dx%3dx%2d stages %d          launch failed, skipped\n",
                            kv.BM, kv.BN, kv.BK, kv.stages);
                continue;
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            const double ms = timeMs([](void* p) {
                KCtx* c = static_cast<KCtx*>(p);
                c->kv->launch(c->a, c->b, c->c, c->m, c->n, c->k, nullptr);
            }, &kctx);
            const double tflops = flops / (ms * 1e-3) / 1e12;
            if (tflops > bestTflops) { bestTflops = tflops; bestIdx = v; }
            std::printf("  tile %3dx%3dx%2d stages %d warp %2dx%2d  %8.2f TFLOP/s  %5.1f%% of cuBLAS\n",
                        kv.BM, kv.BN, kv.BK, kv.stages, kv.WM, kv.WN,
                        tflops, 100.0 * tflops / cublasTflops);
        }

        const double fraction = cublasTflops > 0.0 ? bestTflops / cublasTflops : 0.0;
        std::printf("\n  best: %.2f TFLOP/s, %.1f%% of cuBLAS", bestTflops, fraction * 100.0);
        if (bestIdx >= 0) {
            const ridgebench::KernelVariant& kv = ridgebench::variant(bestIdx);
            std::printf(" (tile %dx%dx%d stages %d)", kv.BM, kv.BN, kv.BK, kv.stages);
        }
        std::printf("\n");

        if (fraction < kMinCublasFraction) {
            std::printf("  FAIL: below the %.0f%% floor. The kernel is not a valid "
                        "ground truth yet, so do not start model validation.\n",
                        kMinCublasFraction * 100.0);
            failures++;
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    std::printf("\n=== Phase 2 gate: %s ===\n", failures == 0 ? "PASS" : "FAIL");
    if (failures != 0) {
        std::printf("%d check(s) failed. Fix the kernel before Phase 3.\n", failures);
        return 1;
    }
    return 0;
}
