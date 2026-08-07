#include "ridge/Model.h"
#include "ridge/Hardware.h"

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

int main() {
    const HardwareModel hw = HardwareModel::a100();

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
