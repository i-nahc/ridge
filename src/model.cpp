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

    // --- CTA and warp geometry ---
    const int numWarps = (cfg.BM / cfg.warpM) * (cfg.BN / cfg.warpN);
    p.numWarps = numWarps;
    const int threadsPerCta = numWarps * 32;

    // --- Per K-step stage times, one CTA (SPEC 4.2) ---
    const double mmasPerWarpPerStep =
        double(cfg.warpM / cfg.mmaM) * (cfg.warpN / cfg.mmaN) * (cfg.BK / cfg.mmaK);
    const double mmasPerCtaPerStep = mmasPerWarpPerStep * numWarps;
    const double tCompute = mmasPerCtaPerStep * hw.mmaCyclesPerInst;

    const double smemBytesPerStep = double(cfg.BM * cfg.BK + cfg.BK * cfg.BN) * db;
    const double tSmem = smemBytesPerStep / hw.smemBytesPerCycle;

    const double tStep = std::max(tCompute, tSmem);
    const double smEfficiency = tCompute / tStep;
    p.tComputeCycles = tCompute;
    p.tSmemCycles = tSmem;
    p.smEfficiency = smEfficiency;

    // --- Occupancy (SPEC 4.3) ---
    const double smemBytesPerCta =
        double(cfg.stages) * (cfg.BM * cfg.BK + cfg.BK * cfg.BN) * db;
    const double ctasBySmem = std::floor(hw.smemBytesPerSM / smemBytesPerCta);
    const double ctasByWarps = std::floor(double(hw.maxWarpsPerSM) / numWarps);
    double ctasByRegs = 1e9;
    if (cfg.regsPerThread > 0) {
        ctasByRegs = std::floor(double(hw.regsPerSM) /
                                (double(cfg.regsPerThread) * threadsPerCta));
    }
    const double ctasPerSM = std::min(std::min(ctasBySmem, ctasByWarps),
                                      std::min(ctasByRegs, double(hw.maxCtasPerSM)));

    if (ctasPerSM < 1.0) {
        // Shared memory or registers too large, the kernel does not fit an SM.
        p.bottleneck = Bottleneck::DoesNotFit;
        p.ctasPerSM = 0;
        return p;
    }
    p.ctasPerSM = int(ctasPerSM);

    const double activeWarps = ctasPerSM * numWarps;
    const double occFactor = std::min(1.0, activeWarps / hw.warpsNeededToHide);
    p.occFactor = occFactor;

    // --- Wave quantization (SPEC 4.5 item 3) ---
    //
    // The GPU runs ctasPerSM * numSMs CTAs concurrently. A grid that is not a
    // whole multiple of that finishes with a partial wave in which most SMs sit
    // idle, and a grid smaller than one wave never fills the machine at all.
    //
    // Total time is ceil(waves) * timePerWave, so the fraction of the machine
    // doing useful work, averaged over the kernel, is
    //     totalCtas / (ceil(waves) * ctaSlots)
    //
    // This is exact when every CTA takes the same time, which holds for a
    // uniform dense GEMM. It is an approximation for ragged work, because real
    // hardware refills slots from a queue rather than in lock step.
    //
    // Measured justification, not just borrowed: the Phase 4 baseline found the
    // model's ranking ability tracks wave count with rank correlation 0.754 and
    // *inverts* below about 5 waves, reaching Spearman -0.882 at 0.15 waves.
    // Without this term the model prefers exactly the large tiles that leave the
    // GPU idle on small problems. See PLAN.md Finding 12.
    const double ctaSlots = ctasPerSM * double(hw.numSMs);
    const double totalCtas =
        double((cfg.M / cfg.BM) * (cfg.N / cfg.BN));
    const double wavesExact = totalCtas / ctaSlots;
    const double fullWaves = std::ceil(wavesExact);
    const double waveEfficiency =
        fullWaves > 0.0 ? totalCtas / (fullWaves * ctaSlots) : 1.0;

    p.ctaSlots = ctaSlots;
    p.totalCtas = totalCtas;
    p.wavesExact = wavesExact;
    p.waveEfficiency = waveEfficiency;

    // --- Two ceilings (SPEC 4.4) ---
    // Compute ceiling. The SM tensor peak is fixed. Occupancy does not multiply
    // it, it only hides latency (occFactor). Do NOT scale by ctasPerSM.
    const double peakTensorTFLOPS =
        (double(mma.flops) / hw.mmaCyclesPerInst) * hw.clockHz * hw.numSMs / 1e12;
    const double computeTFLOPS =
        peakTensorTFLOPS * smEfficiency * occFactor * waveEfficiency;
    p.peakTensorTFLOPS = peakTensorTFLOPS;
    p.computeTFLOPS = computeTFLOPS;

    // HBM roofline. NOTE (v1 limitation): per-CTA global bytes assume no L2 reuse
    // of A row-panels and B col-panels, so arithmetic intensity is underestimated
    // and this ceiling is pessimistic. Adding L2 reuse is the first Phase 4 fix
    // (see PLAN.md Known Issue 1).
    const double globalBytesPerCta = double(cfg.BM * cfg.K + cfg.K * cfg.BN) * db;
    const double flopsPerCta = 2.0 * double(cfg.BM) * cfg.BN * cfg.K;
    const double arithmeticIntensity = flopsPerCta / globalBytesPerCta;
    const double hbmTFLOPS = hw.hbmBytesPerSec * arithmeticIntensity / 1e12;
    p.arithmeticIntensity = arithmeticIntensity;
    p.hbmTFLOPS = hbmTFLOPS;

    // --- Prediction and bottleneck ---
    //
    // When the compute side binds, the label names whichever derating factor is
    // furthest from 1.0, rather than testing them in a fixed order. A fixed
    // order silently reports the first thing below threshold even when another
    // factor is hurting twice as much, which would make the attribution a
    // lookup rather than a diagnosis.
    p.predictedTFLOPS = std::min(computeTFLOPS, hbmTFLOPS);
    if (hbmTFLOPS < computeTFLOPS) {
        p.bottleneck = Bottleneck::Hbm;
    } else {
        const double worst = std::min(std::min(smEfficiency, occFactor), waveEfficiency);
        if (worst >= 0.95)                    p.bottleneck = Bottleneck::TensorCore;
        else if (worst == waveEfficiency)     p.bottleneck = Bottleneck::Waves;
        else if (worst == smEfficiency)       p.bottleneck = Bottleneck::Smem;
        else                                  p.bottleneck = Bottleneck::Occupancy;
    }

    return p;
}

} // namespace ridge
