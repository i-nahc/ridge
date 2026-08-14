#pragma once

namespace ridge {

// Waves is distinct from Occupancy on purpose. Occupancy means too few warps
// resident on an SM to hide latency. Waves means too few CTAs in the grid to
// fill the machine at all, so SMs sit idle no matter how well occupied the busy
// ones are. They are different failures at different levels and they want
// different fixes, so collapsing them would make the diagnosis less useful.
enum class Bottleneck { TensorCore, Smem, Hbm, Occupancy, Waves, DoesNotFit };

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

    // Wave quantization (SPEC 4.5 item 3, PLAN.md Findings 6 and 12).
    // waveEfficiency is the fraction of the machine busy averaged over the
    // kernel: 1.0 when the CTA grid divides evenly into concurrent slots, and as
    // low as totalCtas/ctaSlots when the grid is smaller than one wave.
    double waveEfficiency = 0.0;
    double totalCtas = 0.0;           // grid size, (M/BM) * (N/BN)
    double ctaSlots = 0.0;            // ctasPerSM * numSMs, concurrent capacity
    double wavesExact = 0.0;          // totalCtas / ctaSlots, before rounding up

    int numWarps = 0;
    int ctasPerSM = 0;
    double tComputeCycles = 0.0;
    double tSmemCycles = 0.0;
};

} // namespace ridge
