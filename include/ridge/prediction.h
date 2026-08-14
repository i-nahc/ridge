#pragma once

namespace ridge {

// Waves is distinct from Occupancy on purpose. Occupancy means too few warps
// resident on an SM to hide latency. Waves means too few CTAs in the grid to
// fill the machine at all, so SMs sit idle no matter how well occupied the busy
// ones are. They are different failures at different levels and they want
// different fixes, so collapsing them would make the diagnosis less useful.
// Envelope is separate from Waves for the same reason. Waves means the grid is
// too small to fill the machine. Envelope means the kernel spends its time
// filling the software pipeline and draining accumulators rather than running
// the steady state, which is what a short-K GEMM does. The fix is a smaller BK
// or fewer stages, not more parallelism, so merging the two would point a tuner
// at the wrong knob.
enum class Bottleneck {
    TensorCore, Smem, Hbm, Occupancy, Waves, Envelope, DoesNotFit
};

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
    double arithmeticIntensity = 0.0; // FLOP per global byte, after L2 reuse
    double baseIntensity = 0.0;       // FLOP per global byte before L2 reuse
    double l2ReuseFactor = 0.0;       // sqrt(ctaSlots), wave-geometry sharing

    // Wave quantization (SPEC 4.5 item 3, PLAN.md Findings 6 and 12).
    // waveEfficiency is the fraction of the machine busy averaged over the
    // kernel: 1.0 when the CTA grid divides evenly into concurrent slots, and as
    // low as totalCtas/ctaSlots when the grid is smaller than one wave.
    double waveEfficiency = 0.0;
    double totalCtas = 0.0;           // grid size, (M/BM) * (N/BN)
    double ctaSlots = 0.0;            // ctasPerSM * numSMs, concurrent capacity
    double wavesExact = 0.0;          // totalCtas / ctaSlots, before rounding up

    // Envelope (docs/MODEL.md section 5). A kernel does not start in steady
    // state: the pipeline must fill before the first MMA issues and the
    // accumulators must drain after the last one. envelopeEfficiency is the
    // fraction of the kernel spent in steady state. Negligible for large K and
    // dominant for small K.
    double envelopeEfficiency = 0.0;
    double tPrologueCycles = 0.0;     // (stages-1) tile loads with nothing to overlap
    double tEpilogueCycles = 0.0;     // fp32 accumulators out to global
    double tBodyCycles = 0.0;         // kSteps * tStep
    double kSteps = 0.0;              // K / BK

    int numWarps = 0;
    int ctasPerSM = 0;
    double tComputeCycles = 0.0;
    double tSmemCycles = 0.0;
    double tStepCycles = 0.0;         // tCompute + tSmem, they are in series
};

} // namespace ridge
