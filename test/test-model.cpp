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
        CHECK_NEAR(p.smEfficiency,         1.00,   1e-9);
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

        CHECK_NEAR(p.peakTensorTFLOPS,     311.9,  0.05);
        // 311.87 x 1.00 x 0.75 x 0.790123 = 184.81, was 233.90 before waves
        CHECK_NEAR(p.computeTFLOPS,        184.81, 0.05);
        CHECK_NEAR(p.arithmeticIntensity,  64.0,   1e-9);
        CHECK_NEAR(p.hbmTFLOPS,            128.0,  0.05);
        // HBM still binds, so the headline prediction is unchanged at 128.0.
        CHECK_NEAR(p.predictedTFLOPS,      128.0,  0.05);
        CHECK(p.bottleneck == Bottleneck::Hbm);
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
        // 311.87 x 1.00 x 0.75 x 0.049383 = 11.55, far under the 128.0 HBM
        // ceiling, so the compute side now binds and the label says why.
        CHECK_NEAR(p.computeTFLOPS,  11.55,         0.05);
        CHECK_NEAR(p.predictedTFLOPS, 11.55,        0.05);
        CHECK(p.bottleneck == Bottleneck::Waves);
    }

    // Peak sanity: A100 FP16 dense tensor peak is ~312 TFLOPS.
    {
        GemmConfig cfg;
        const Prediction p = predict(cfg, hw);
        CHECK(p.peakTensorTFLOPS > 250.0 && p.peakTensorTFLOPS < 400.0);
    }

    // Invariants on a large square GEMM: prediction is the min of the two
    // ceilings, and the efficiency terms are in [0, 1]. NOTE: v1 does not model
    // L2 reuse, so this config may report HBM-bound (PLAN.md Known Issue 1). We
    // check invariants, not a specific bottleneck.
    {
        GemmConfig cfg;
        cfg.M = cfg.N = cfg.K = 8192;
        const Prediction p = predict(cfg, hw);
        CHECK(p.predictedTFLOPS > 0.0);
        CHECK(p.predictedTFLOPS <= p.computeTFLOPS + 1e-6);
        CHECK(p.predictedTFLOPS <= p.hbmTFLOPS + 1e-6);
        CHECK(p.smEfficiency > 0.0 && p.smEfficiency <= 1.0 + 1e-9);
        CHECK(p.occFactor > 0.0 && p.occFactor <= 1.0 + 1e-9);
        CHECK(p.ctasPerSM >= 1);
    }

    // Tile-size sanity: a larger CTA tile reuses each loaded operand more, so it
    // has higher arithmetic intensity and a higher HBM ceiling. NOTE: in v1 the
    // per-CTA roofline makes intensity depend only on tile size, not K, because
    // L2 reuse is not modeled yet (PLAN.md Known Issue 1). So this compares tile
    // sizes, not K values.
    {
        GemmConfig big;   big.BM = 256; big.BN = 256; big.warpM = 64; big.warpN = 64;
        GemmConfig small; small.BM = 64; small.BN = 64; small.warpM = 64; small.warpN = 64;
        CHECK(predict(big, hw).arithmeticIntensity >
              predict(small, hw).arithmeticIntensity);
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
