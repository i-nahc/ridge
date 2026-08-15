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

    // The worked example, pinned value by value.
    // Invariant checks alone cannot catch a math regression that happens to
    // preserve the invariants.
    //
    // The config is built explicitly rather than relying on GemmConfig defaults,
    // because this verifies the math against a fixed input vector and changing a
    // default is not a divergence from the spec.
    //
    // IF A MODEL CHANGE BROKE THESE ASSERTIONS, do not paste in the new
    // ridge-predict output. Re-derive the example by hand from the new math and
    // copy the hand-derived numbers here, keeping the working below in step. The
    // test follows the derivation, never the code. Pasting the code's output
    // turns an independent check into a snapshot of whatever the code now does.
    //
    // regsPerThread is 128 here as a stand-in for an unknown kernel. The real
    // kernel measures 224 at this tile config.
    {
        GemmConfig cfg;
        cfg.M = cfg.N = cfg.K = 4096;
        cfg.dtype = DType::FP16;
        cfg.BM = 128; cfg.BN = 128; cfg.BK = 32;
        cfg.warpM = 64; cfg.warpN = 64;
        cfg.stages = 3;
        cfg.regsPerThread = 128;

        const Prediction p = predict(cfg, hw);

        CHECK_NEAR(p.numWarps,       4,     0);
        CHECK_NEAR(p.tComputeCycles, 512.0, 1e-6);
        CHECK_NEAR(p.tSmemCycles,    256.0, 1e-6);   // 4 warps x (64+64) x 32 x 2 = 32768 B / 128
        CHECK_NEAR(p.tStepCycles,    768.0, 1e-6);   // in series, not max(512, 256)
        CHECK_NEAR(p.smEfficiency,   512.0 / 768.0, 1e-9);
        CHECK_NEAR(p.ctasPerSM,      3,     0);
        CHECK_NEAR(p.occFactor,      0.75,  1e-9);

        //   grid           4096/128 x 4096/128 = 32 x 32 = 1024 CTAs
        //   slots          3 CTAs/SM x 108 SMs = 324
        //   waves          1024 / 324 = 3.1605 -> ceil 4
        //   waveEfficiency 1024 / (4 x 324) = 0.790123
        CHECK_NEAR(p.totalCtas,      1024.0,          1e-9);
        CHECK_NEAR(p.ctaSlots,       324.0,           1e-9);
        CHECK_NEAR(p.wavesExact,     1024.0 / 324.0,  1e-9);
        CHECK_NEAR(p.waveEfficiency, 1024.0 / 1296.0, 1e-9);

        //   kSteps    4096/32 = 128
        //   gmemBpc   2.0e12 / (1.41e9 x 108) = 13.133701 B/cycle
        //   tileBytes (128x32 + 32x128) x 2 = 16384
        //   tPrologue (3-1) x 16384 / 13.133701 / 3 =  831.651840
        //   tEpilogue 128 x 128 x 4 / 13.133701 / 3 = 1663.303680
        //   tBody     128 x 768 = 98304
        //   envEff    98304 / (831.651840 + 98304 + 1663.303680) = 0.975248
        CHECK_NEAR(p.kSteps,             128.0,       1e-9);
        CHECK_NEAR(p.tPrologueCycles,     831.651840, 1e-4);
        CHECK_NEAR(p.tEpilogueCycles,    1663.303680, 1e-4);
        CHECK_NEAR(p.tBodyCycles,       98304.0,      1e-6);
        CHECK_NEAR(p.envelopeEfficiency,    0.975248, 1e-6);

        CHECK_NEAR(p.peakTensorTFLOPS, 311.869440, 0.001);
        // 311.869440 x 0.666667 x 0.75 x 0.790123 x 0.975248 = 120.158068
        CHECK_NEAR(p.computeTFLOPS,    120.158068, 1e-4);

        CHECK_NEAR(p.baseIntensity,       64.0,   1e-9);
        CHECK_NEAR(p.l2ReuseFactor,       18.0,   1e-9);   // sqrt(324)
        CHECK_NEAR(p.arithmeticIntensity, 1152.0, 1e-9);
        CHECK_NEAR(p.hbmTFLOPS,           2304.0, 1e-6);

        // HBM no longer binds anywhere near this config, so the compute side
        // binds and smEfficiency at 0.667 is the worst factor.
        CHECK_NEAR(p.predictedTFLOPS, 120.158068, 1e-4);
        CHECK(p.bottleneck == Bottleneck::Smem);
    }

    // A grid smaller than one wave is WAVES-bound and derated hard.
    //   grid   512/128 x 512/128 = 16 CTAs
    //   slots  324, waves = 0.0494 -> ceil 1
    //   waveEfficiency = 16 / 324 = 0.049383
    {
        GemmConfig cfg;
        cfg.M = cfg.N = 512;
        cfg.K = 4096;
        cfg.BM = 128; cfg.BN = 128; cfg.BK = 32;
        cfg.warpM = 64; cfg.warpN = 64;
        cfg.stages = 3;
        cfg.regsPerThread = 128;

        const Prediction p = predict(cfg, hw);
        CHECK_NEAR(p.totalCtas,      16.0,         1e-9);
        CHECK_NEAR(p.ctaSlots,       324.0,        1e-9);
        CHECK_NEAR(p.waveEfficiency, 16.0 / 324.0, 1e-9);
        // 311.869440 x 0.666667 x 0.75 x 0.049383 x 0.975248 = 7.509879
        CHECK_NEAR(p.computeTFLOPS,   7.509879, 1e-4);
        CHECK_NEAR(p.predictedTFLOPS, 7.509879, 1e-4);
        CHECK(p.bottleneck == Bottleneck::Waves);
    }

    // A100 FP16 dense tensor peak is about 312 TFLOPS.
    {
        GemmConfig cfg;
        CHECK(predict(cfg, hw).peakTensorTFLOPS > 250.0 && predict(cfg, hw).peakTensorTFLOPS < 400.0);
    }

    // Invariants on a large square GEMM.
    {
        GemmConfig cfg;
        cfg.M = cfg.N = cfg.K = 8192;
        const Prediction p = predict(cfg, hw);
        CHECK(p.predictedTFLOPS > 0.0);
        CHECK(p.predictedTFLOPS <= p.computeTFLOPS + 1e-6);
        CHECK(p.predictedTFLOPS <= p.hbmTFLOPS + 1e-6);
        CHECK(p.occFactor > 0.0 && p.occFactor <= 1.0 + 1e-9);
        CHECK(p.envelopeEfficiency > 0.0 && p.envelopeEfficiency <= 1.0 + 1e-9);
        CHECK(p.ctasPerSM >= 1);
        // A kernel feeding its tensor cores from shared memory always pays the
        // ldmatrix time, so this can never reach 1.0.
        CHECK(p.smEfficiency > 0.0 && p.smEfficiency < 1.0);
    }

    // A larger CTA tile reuses each loaded operand more. This checks
    // baseIntensity rather than the effective intensity, because the L2 factor
    // depends on ctaSlots and a bigger tile fits fewer CTAs per SM, which would
    // confound the two effects.
    {
        GemmConfig big;   big.BM = 256; big.BN = 256; big.warpM = 64; big.warpN = 64;
        GemmConfig small; small.BM = 64; small.BN = 64; small.warpM = 64; small.warpN = 64;
        CHECK(predict(big, hw).baseIntensity > predict(small, hw).baseIntensity);
    }

    // Shrinking K leaves less steady state to amortise fill and drain over, so
    // the envelope derates harder and eventually binds.
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

    // An absurd tile should not fit an SM.
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
