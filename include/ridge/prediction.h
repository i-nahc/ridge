#pragma once

namespace ridge {

enum class Bottleneck { TensorCore, Smem, Hbm, Occupancy, DoesNotFit };

const char* toString(Bottleneck b);

// Result of a prediction. predictedTFLOPS and bottleneck are the headline. The
// rest are diagnostics that explain the number (SPEC section 4).
struct Prediction {
    double predictedTFLOPS = 0.0;
    Bottleneck bottleneck = Bottleneck::TensorCore;

    double computeTFLOPS = 0.0;       // compute ceiling after efficiency and occupancy
    double hbmTFLOPS = 0.0;           // HBM roofline ceiling
    double peakTensorTFLOPS = 0.0;    // SM tensor peak, no derating
    double smEfficiency = 0.0;        // tCompute / tStep, 1.0 = never waits on smem
    double occFactor = 0.0;           // latency-hiding factor, 1.0 = fully hidden
    double arithmeticIntensity = 0.0; // FLOP per global byte (per-CTA, no L2 reuse in v1)

    int numWarps = 0;
    int ctasPerSM = 0;
    double tComputeCycles = 0.0;
    double tSmemCycles = 0.0;
};

} // namespace ridge
