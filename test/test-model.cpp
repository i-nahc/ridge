#include "ridge/model.h"
#include "ridge/hardware.h"

#include <cmath>
#include <cstdio>

using namespace ridge;

static int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL: %s (line %d)\n", #cond, __LINE__);       \
            failures++;                                                 \
        }                                                               \
    } while (0)

// Asserts a computed value against the number written down in docs/MODEL.md.
// Invariant checks alone cannot catch a math regression that happens to preserve
// the invariants, which is why the worked example is pinned value by value.
#define CHECK_NEAR(actual, expected, tol)                                       \
    do {                                                                        \
        const double a_ = (actual), e_ = (expected);                            \
        if (!(std::fabs(a_ - e_) <= (tol))) {                                   \
            std::printf("FAIL: %s = %.4f, expected %.4f +/- %.4f (line %d)\n",  \
                        #actual, a_, e_, double(tol), __LINE__);                \
            failures++;                                                         \
        }                                                                       \
    } while (0)

int main() {
    const HardwareModel hw = HardwareModel::a100();

    // ---------------------------------------------------------------------
    // Phase 1 gate: the docs/MODEL.md section 9 worked example.
    //
    // This is the acceptance gate from PLAN.md, enforced here rather than by
    // eyeballing ridge-predict output. The config is built explicitly instead of
    // relying on GemmConfig's defaults, because the anchor verifies the math
    // against a fixed input vector. Changing a default is not a divergence from
    // the spec and must not be able to fail this test.
    //
    // regsPerThread is 128 here as a stand-in for an unknown kernel. The real
    // Phase 2 kernel measures 224 at this tile config, see PLAN.md Finding 4.
    //
    // IF YOU ARE HERE BECAUSE A MODEL CHANGE BROKE THESE ASSERTIONS, STOP.
    // Do not paste in the new ridge-predict output. Re-derive the docs/MODEL.md
    // section 9 worked example by hand from the new math, update section 9, and
    // only then copy the hand-derived numbers here. The test follows the
    // derivation, never the code. Pasting the code's output turns this from an
    // independent check of the math into a snapshot of whatever the code now
    // does, which cannot detect that the code is wrong. See PLAN.md
    // anti-pattern 9.
    // ---------------------------------------------------------------------
    {
        GemmConfig cfg;
        cfg.M = cfg.N = cfg.K = 4096;
        cfg.dtype = DType::FP16;
        cfg.BM = 128; cfg.BN = 128; cfg.BK = 32;
        cfg.warpM = 64; cfg.warpN = 64;
        cfg.stages = 3;
        cfg.regsPerThread = 128;

        const Prediction p = predict(cfg, hw);

        CHECK_NEAR(p.numWarps,             4,      0);
        CHECK_NEAR(p.tComputeCycles,       512.0,  1e-6);
        CHECK_NEAR(p.tSmemCycles,          256.0,  1e-6);   // Finding 8: read traffic, 4 warps x (64+64) x 32 x 2 = 32768 B / 128
        // Finding 17: ldmatrix is synchronous and the next mma consumes what it
        // wrote, so these are in series. 512 + 256 = 768, not max(512,256).
        CHECK_NEAR(p.tStepCycles,          768.0,  1e-6);
        CHECK_NEAR(p.smEfficiency,         512.0 / 768.0, 1e-9);
        CHECK_NEAR(p.ctasPerSM,            3,      0);
        CHECK_NEAR(p.occFactor,            0.75,   1e-9);

        // Wave quantization, hand-derived:
        //   grid           4096/128 x 4096/128 = 32 x 32 = 1024 CTAs
        //   slots          3 CTAs/SM x 108 SMs = 324
        //   waves          1024 / 324 = 3.1605 -> ceil 4
        //   waveEfficiency 1024 / (4 x 324) = 1024/1296 = 0.790123
        CHECK_NEAR(p.totalCtas,            1024.0, 1e-9);
        CHECK_NEAR(p.ctaSlots,             324.0,  1e-9);
        CHECK_NEAR(p.wavesExact,           1024.0 / 324.0, 1e-9);
        CHECK_NEAR(p.waveEfficiency,       1024.0 / 1296.0, 1e-9);

        // Envelope, hand-derived:
        //   kSteps    4096/32 = 128
        //   gmemBpc   2.0e12 / (1.41e9 x 108) = 13.133701 B/cycle
        //   tileBytes (128x32 + 32x128) x 2 = 16384
        //   tPrologue (3-1) x 16384 / 13.133701 / 3 =  831.651840  (shared across 3 CTAs)
        //   tEpilogue 128 x 128 x 4 / 13.133701 / 3 = 1663.303680
        //   tBody     128 x 768 = 98304
        //   envEff    98304 / (831.651840 + 98304 + 1663.303680) = 0.975248
        CHECK_NEAR(p.kSteps,               128.0,        1e-9);
        CHECK_NEAR(p.tPrologueCycles,       831.651840,  1e-4);
        CHECK_NEAR(p.tEpilogueCycles,      1663.303680,  1e-4);
        CHECK_NEAR(p.tBodyCycles,          98304.0,      1e-6);
        CHECK_NEAR(p.envelopeEfficiency,   0.975248,     1e-6);

        CHECK_NEAR(p.peakTensorTFLOPS,     311.869440, 0.001);
        // 311.869440 x 0.666667 x 0.75 x 0.790123 x 0.975248 = 120.158068
        CHECK_NEAR(p.computeTFLOPS,        120.158068, 1e-4);

        // L2 reuse across the wave: sqrt(324) = 18 exactly.
        CHECK_NEAR(p.baseIntensity,        64.0,   1e-9);
        CHECK_NEAR(p.l2ReuseFactor,        18.0,   1e-9);
        CHECK_NEAR(p.arithmeticIntensity,  1152.0, 1e-9);
        CHECK_NEAR(p.hbmTFLOPS,            2304.0, 1e-6);

        // HBM no longer binds anywhere near this config, which is the point of
        // the L2 term. The compute side binds and the worst derating factor is
        // smEfficiency at 0.667, the ldmatrix traffic feeding the tensor cores.
        CHECK_NEAR(p.predictedTFLOPS,      120.158068, 1e-4);
        CHECK(p.bottleneck == Bottleneck::Smem);
    }

    // Wave quantization: a grid smaller than one wave should be WAVES-bound and
    // derated hard. Hand-derived for 512x512x4096 on the placeholder model:
    //   grid   512/128 x 512/128 = 4 x 4 = 16 CTAs
    //   slots  324, so waves = 16/324 = 0.0494 -> ceil 1
    //   waveEfficiency = 16 / (1 x 324) = 0.049383
    // This is the config the Phase 4 baseline mispredicted by +1109%.
    {
        GemmConfig cfg;
        cfg.M = cfg.N = 512;
        cfg.K = 4096;
        cfg.BM = 128; cfg.BN = 128; cfg.BK = 32;
        cfg.warpM = 64; cfg.warpN = 64;
        cfg.stages = 3;
        cfg.regsPerThread = 128;

        const Prediction p = predict(cfg, hw);
        CHECK_NEAR(p.totalCtas,      16.0,          1e-9);
        CHECK_NEAR(p.ctaSlots,       324.0,         1e-9);
        CHECK_NEAR(p.waveEfficiency, 16.0 / 324.0,  1e-9);
        // 311.869440 x 0.666667 x 0.75 x 0.049383 x 0.975248 = 7.509879.
        // waveEfficiency at 0.0494 is by far the worst factor, so WAVES is the
        // label even though smEfficiency and the envelope also derate.
        CHECK_NEAR(p.computeTFLOPS,   7.509879,     1e-4);
        CHECK_NEAR(p.predictedTFLOPS, 7.509879,     1e-4);
        CHECK(p.bottleneck == Bottleneck::Waves);
    }

    // Peak sanity: A100 FP16 dense tensor peak is ~312 TFLOPS.
    {
        GemmConfig cfg;
        const Prediction p = predict(cfg, hw);
        CHECK(p.peakTensorTFLOPS > 250.0 && p.peakTensorTFLOPS < 400.0);
    }

    // Invariants on a large square GEMM: prediction is the min of the two
    // ceilings, and the efficiency terms are in [0, 1].
    {
        GemmConfig cfg;
        cfg.M = cfg.N = cfg.K = 8192;
        const Prediction p = predict(cfg, hw);
        CHECK(p.predictedTFLOPS > 0.0);
        CHECK(p.predictedTFLOPS <= p.computeTFLOPS + 1e-6);
        CHECK(p.predictedTFLOPS <= p.hbmTFLOPS + 1e-6);
        CHECK(p.smEfficiency > 0.0 && p.smEfficiency <= 1.0 + 1e-9);
        CHECK(p.occFactor > 0.0 && p.occFactor <= 1.0 + 1e-9);
        CHECK(p.envelopeEfficiency > 0.0 && p.envelopeEfficiency <= 1.0 + 1e-9);
        CHECK(p.ctasPerSM >= 1);
        // smEfficiency can no longer reach 1.0: a kernel that feeds its tensor
        // cores from shared memory always pays the ldmatrix time (Finding 17).
        CHECK(p.smEfficiency < 1.0);
    }

    // Tile-size sanity: a larger CTA tile reuses each loaded operand more, so it
    // has higher intensity before any L2 effect. This checks baseIntensity
    // rather than arithmeticIntensity on purpose, because the L2 factor depends
    // on ctaSlots and a bigger tile fits fewer CTAs per SM, so the effective
    // intensity confounds two things. The tile-shape claim is about the base.
    {
        GemmConfig big;   big.BM = 256; big.BN = 256; big.warpM = 64; big.warpN = 64;
        GemmConfig small; small.BM = 64; small.BN = 64; small.warpM = 64; small.warpN = 64;
        CHECK(predict(big, hw).baseIntensity >
              predict(small, hw).baseIntensity);
    }

    // Envelope sanity: shrinking K at a fixed tile leaves less steady state to
    // amortise pipeline fill and accumulator drain over, so the envelope must
    // derate harder and eventually become the binding factor. This is the term
    // that fixed the short-K blowup in PLAN.md Finding 17.
    {
        GemmConfig deep;
        deep.M = deep.N = 4096; deep.K = 4096;
        deep.BM = 128; deep.BN = 128; deep.BK = 32;
        deep.warpM = 64; deep.warpN = 64; deep.stages = 3; deep.regsPerThread = 128;
        GemmConfig shallow = deep;
        shallow.K = 128;
        const Prediction pd = predict(deep, hw);
        const Prediction ps = predict(shallow, hw);
        CHECK(ps.envelopeEfficiency < pd.envelopeEfficiency);
        CHECK(ps.kSteps < pd.kSteps);
        CHECK(ps.bottleneck == Bottleneck::Envelope);
    }

    // Does-not-fit sanity: an absurd tile should not fit an SM.
    {
        GemmConfig cfg;
        cfg.BM = 512; cfg.BN = 512; cfg.BK = 128; cfg.stages = 8;
        cfg.warpM = 64; cfg.warpN = 64;
        CHECK(predict(cfg, hw).bottleneck == Bottleneck::DoesNotFit);
    }

    if (failures == 0) {
        std::printf("all tests passed\n");
        return 0;
    }
    std::printf("%d failure(s)\n", failures);
    return 1;
}
