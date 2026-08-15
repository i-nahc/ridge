#pragma once

namespace ridge {

// Waves, Occupancy and Envelope are separate labels because they want different
// fixes. Occupancy means too few warps resident to hide latency, Waves means too
// few CTAs in the grid to fill the machine, and Envelope means the kernel spends
// its time filling the pipeline and draining accumulators rather than in steady
// state.
enum class Bottleneck { TensorCore, Smem, Hbm, Occupancy, Waves, Envelope, DoesNotFit };

const char* toString(Bottleneck b);

// predictedTFLOPS and bottleneck are the result. The rest explain it.
struct Prediction {
    double predictedTFLOPS = 0.0;
    Bottleneck bottleneck = Bottleneck::TensorCore;

    double computeTFLOPS = 0.0;       // compute ceiling after all deratings
    double hbmTFLOPS = 0.0;           // HBM roofline ceiling
    double peakTensorTFLOPS = 0.0;    // SM tensor peak, no derating
    double smEfficiency = 0.0;        // tCompute / tStep
    double occFactor = 0.0;           // latency hiding, 1.0 = fully hidden
    double arithmeticIntensity = 0.0; // FLOP per global byte, after L2 reuse
    double baseIntensity = 0.0;       // FLOP per global byte, before L2 reuse
    double l2ReuseFactor = 0.0;       // sqrt(ctaSlots)

    // Fraction of the machine busy averaged over the kernel. 1.0 when the grid
    // divides evenly into concurrent slots, as low as totalCtas/ctaSlots when the
    // grid is smaller than one wave.
    double waveEfficiency = 0.0;
    double totalCtas = 0.0;
    double ctaSlots = 0.0;            // ctasPerSM * numSMs
    double wavesExact = 0.0;          // totalCtas / ctaSlots, before rounding up

    // Fraction of the kernel spent in steady state. Negligible for large K,
    // dominant for small K.
    double envelopeEfficiency = 0.0;
    double tPrologueCycles = 0.0;
    double tEpilogueCycles = 0.0;
    double tBodyCycles = 0.0;
    double kSteps = 0.0;

    int numWarps = 0;
    int ctasPerSM = 0;
    double tComputeCycles = 0.0;
    double tSmemCycles = 0.0;
    double tStepCycles = 0.0;         // tCompute + tSmem, they are in series
};

} // namespace ridge
