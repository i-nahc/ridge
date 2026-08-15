#include "ridge/model.h"
#include "ridge/mma.h"

#include <algorithm>
#include <cmath>

namespace ridge {

const char* toString(Bottleneck b) {
    switch (b) {
        case Bottleneck::TensorCore: return "TENSOR_CORE";
        case Bottleneck::Smem:       return "SMEM";
        case Bottleneck::Hbm:        return "HBM";
        case Bottleneck::Occupancy:  return "OCC";
        case Bottleneck::Waves:      return "WAVES";
        case Bottleneck::Envelope:   return "ENVELOPE";
        case Bottleneck::DoesNotFit: return "DOES_NOT_FIT";
    }
    return "UNKNOWN";
}

Prediction predict(const GemmConfig& cfgIn, const HardwareModel& hw) {
    GemmConfig cfg = cfgIn;
    const MmaShape mma = mmaShapeFor(cfg.dtype);
    if (cfg.mmaM == 0) { cfg.mmaM = mma.m; cfg.mmaN = mma.n; cfg.mmaK = mma.k; }
    const int db = dtypeBytes(cfg.dtype);

    Prediction p;

    const int numWarps = (cfg.BM / cfg.warpM) * (cfg.BN / cfg.warpN);
    const int threadsPerCta = numWarps * 32;
    p.numWarps = numWarps;

    // Per K-step stage times for one CTA.
    const double mmasPerWarpPerStep = double(cfg.warpM / cfg.mmaM) * (cfg.warpN / cfg.mmaN) * (cfg.BK / cfg.mmaK);
    const double tCompute = mmasPerWarpPerStep * numWarps * hw.mmaCyclesPerInst;

    // ldmatrix read traffic, not the cp.async write traffic. Per warp per K-step
    // the A operand costs (warpM/16) x (BK/16) ldmatrix.x4 at 512 B and the B
    // operand (warpN/8) x (BK/16) ldmatrix.x2 at 256 B.
    const double smemBytesPerStep = double(numWarps) * (cfg.warpM + cfg.warpN) * cfg.BK * db;
    const double tSmem = smemBytesPerStep / hw.smemBytesPerCycle;

    // These add rather than overlap. cp.async is asynchronous by construction and
    // does overlap compute, which is why it is absent here and appears only in the
    // envelope below. ldmatrix is synchronous and the next mma consumes the
    // registers it just wrote, so it sits in series with tensor time.
    const double tStep = tCompute + tSmem;
    const double smEfficiency = tCompute / tStep;
    p.tComputeCycles = tCompute;
    p.tSmemCycles = tSmem;
    p.tStepCycles = tStep;
    p.smEfficiency = smEfficiency;

    // Occupancy. The tightest of the four limits binds.
    const double smemBytesPerCta = double(cfg.stages) * (cfg.BM * cfg.BK + cfg.BK * cfg.BN) * db;
    const double ctasBySmem = std::floor(hw.smemBytesPerSM / smemBytesPerCta);
    const double ctasByWarps = std::floor(double(hw.maxWarpsPerSM) / numWarps);
    double ctasByRegs = 1e9;
    if (cfg.regsPerThread > 0) {
        ctasByRegs = std::floor(double(hw.regsPerSM) / (double(cfg.regsPerThread) * threadsPerCta));
    }
    const double ctasPerSM = std::min(std::min(ctasBySmem, ctasByWarps), std::min(ctasByRegs, double(hw.maxCtasPerSM)));

    if (ctasPerSM < 1.0) {
        p.bottleneck = Bottleneck::DoesNotFit;
        p.ctasPerSM = 0;
        return p;
    }
    p.ctasPerSM = int(ctasPerSM);

    const double activeWarps = ctasPerSM * numWarps;
    const double occFactor = std::min(1.0, activeWarps / hw.warpsNeededToHide);
    p.occFactor = occFactor;

    // Wave quantization. The machine runs ctaSlots CTAs at once, so a grid that is
    // not a whole multiple of that ends on a partial wave with most SMs idle.
    // Exact for uniform work, approximate for ragged work since hardware refills
    // slots from a queue rather than in lock step.
    const double ctaSlots = ctasPerSM * double(hw.numSMs);
    const double totalCtas = double((cfg.M / cfg.BM) * (cfg.N / cfg.BN));
    const double wavesExact = totalCtas / ctaSlots;
    const double fullWaves = std::ceil(wavesExact);
    const double waveEfficiency = fullWaves > 0.0 ? totalCtas / (fullWaves * ctaSlots) : 1.0;

    p.ctaSlots = ctaSlots;
    p.totalCtas = totalCtas;
    p.wavesExact = wavesExact;
    p.waveEfficiency = waveEfficiency;

    // Prologue and epilogue. stages-1 tile loads have to land before the first mma
    // can issue, and the accumulators drain after the last one. Both are divided by
    // ctasPerSM: nothing synchronises CTAs with each other, so with several
    // resident only the SM's first fill and last drain are exposed.
    const double gmemBytesPerCycle = hw.hbmBytesPerSec / (hw.clockHz * double(hw.numSMs));
    const double tileBytes = double(cfg.BM * cfg.BK + cfg.BK * cfg.BN) * db;
    const double kSteps = double(cfg.K) / cfg.BK;
    const double tPrologue = double(cfg.stages - 1) * tileBytes / gmemBytesPerCycle / ctasPerSM;
    const double tEpilogue = double(cfg.BM) * cfg.BN * 4.0 / gmemBytesPerCycle / ctasPerSM;
    const double tBody = kSteps * tStep;
    const double envelopeEfficiency = tBody / (tPrologue + tBody + tEpilogue);

    p.kSteps = kSteps;
    p.tPrologueCycles = tPrologue;
    p.tEpilogueCycles = tEpilogue;
    p.tBodyCycles = tBody;
    p.envelopeEfficiency = envelopeEfficiency;

    // Compute ceiling. Resident CTAs share the tensor cores, so occupancy hides
    // latency through occFactor but never multiplies the peak.
    const double peakTensorTFLOPS = (double(mma.flops) / hw.mmaCyclesPerInst) * hw.clockHz * hw.numSMs / 1e12;
    const double computeTFLOPS = peakTensorTFLOPS * smEfficiency * occFactor * waveEfficiency * envelopeEfficiency;
    p.peakTensorTFLOPS = peakTensorTFLOPS;
    p.computeTFLOPS = computeTFLOPS;

    // HBM roofline. The resident CTAs form a wave covering a roughly square region
    // of the output, sqrt(ctaSlots) tiles on a side, so one DRAM fetch of an A row
    // panel or B column panel serves about that many consumers through L2. No
    // capacity term, so this is optimistic for very large tiles.
    const double globalBytesPerCta = double(cfg.BM * cfg.K + cfg.K * cfg.BN) * db;
    const double flopsPerCta = 2.0 * double(cfg.BM) * cfg.BN * cfg.K;
    const double baseIntensity = flopsPerCta / globalBytesPerCta;
    const double l2ReuseFactor = std::sqrt(ctaSlots);
    const double arithmeticIntensity = baseIntensity * l2ReuseFactor;
    p.baseIntensity = baseIntensity;
    p.l2ReuseFactor = l2ReuseFactor;
    p.arithmeticIntensity = arithmeticIntensity;
    p.hbmTFLOPS = hw.hbmBytesPerSec * arithmeticIntensity / 1e12;

    // The label names whichever derating factor is furthest from 1.0. Testing in a
    // fixed order would report the first one under threshold even when another
    // costs twice as much throughput.
    p.predictedTFLOPS = std::min(computeTFLOPS, p.hbmTFLOPS);
    if (p.hbmTFLOPS < computeTFLOPS) {
        p.bottleneck = Bottleneck::Hbm;
    } else {
        const double worst = std::min(std::min(smEfficiency, occFactor), std::min(waveEfficiency, envelopeEfficiency));
        if (worst >= 0.95)                    p.bottleneck = Bottleneck::TensorCore;
        else if (worst == waveEfficiency)     p.bottleneck = Bottleneck::Waves;
        else if (worst == envelopeEfficiency) p.bottleneck = Bottleneck::Envelope;
        else if (worst == smEfficiency)       p.bottleneck = Bottleneck::Smem;
        else                                  p.bottleneck = Bottleneck::Occupancy;
    }

    return p;
}

} // namespace ridge
