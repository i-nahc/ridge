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

    // --- CTA and warp geometry ---
    const int numWarps = (cfg.BM / cfg.warpM) * (cfg.BN / cfg.warpN);
    p.numWarps = numWarps;
    const int threadsPerCta = numWarps * 32;

    // --- Per K-step stage times, one CTA (SPEC 4.2) ---
    const double mmasPerWarpPerStep =
        double(cfg.warpM / cfg.mmaM) * (cfg.warpN / cfg.mmaN) * (cfg.BK / cfg.mmaK);
    const double mmasPerCtaPerStep = mmasPerWarpPerStep * numWarps;
    const double tCompute = mmasPerCtaPerStep * hw.mmaCyclesPerInst;

    // Shared memory READ traffic, not write traffic (PLAN.md Finding 8).
    //
    // The stage that competes for shared-memory bandwidth is what ldmatrix pulls
    // *out* to feed the tensor cores, not what cp.async writes *in*. Those are
    // different quantities and the second is smaller.
    //
    // Per warp per K-step, from the instruction shapes:
    //   A: (warpM/16) x (BK/16) ldmatrix.x4 at 512 B  = warpM * BK * dtypeBytes
    //   B: (warpN/8)  x (BK/16) ldmatrix.x2 at 256 B  = warpN * BK * dtypeBytes
    // so per CTA it is numWarps * (warpM + warpN) * BK * dtypeBytes.
    //
    // The old form was (BM*BK + BK*BN) * dtypeBytes, which does not mention the
    // warp tile at all. That was why the model gave identical smem behaviour to
    // configs that measure very differently: at 128x128x32 the read traffic is
    // 32768 B for a 64x64 warp tile and 65536 B for 32x32, while the old formula
    // returned 16384 B for both.
    const double smemBytesPerStep =
        double(numWarps) * (cfg.warpM + cfg.warpN) * cfg.BK * db;
    const double tSmem = smemBytesPerStep / hw.smemBytesPerCycle;

    // These add, they do not overlap (PLAN.md Finding 17, docs/MODEL.md section 3).
    //
    // The two memory paths into a CTA are not alike, and the difference is in the
    // ISA rather than in the tuning. cp.async moves global to shared
    // asynchronously, which is its whole purpose: the warp issues it and keeps
    // going. That path really does overlap compute and belongs under a max.
    //
    // ldmatrix moves shared to register synchronously and the very next mma
    // consumes the registers it just wrote, so the warp cannot issue that mma
    // until the ldmatrix feeding it has landed. Our kernel makes this concrete:
    // within a K-step a warp issues all its ldmatrix, then all its mma. Warps in
    // a CTA run the same code between the same barriers, so they enter those
    // phases together rather than covering for each other.
    //
    // Measured, not assumed. On the 336-row A100 sweep with every other term
    // held fixed, max gives MAPE 36.17% and the sum gives 30.15%.
    const double tStep = tCompute + tSmem;
    const double smEfficiency = tCompute / tStep;
    p.tComputeCycles = tCompute;
    p.tSmemCycles = tSmem;
    p.tStepCycles = tStep;
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

    // --- Envelope: prologue and epilogue (SPEC 4.6, docs/MODEL.md section 5) ---
    //
    // Everything above is a steady-state rate, but a kernel does not begin in
    // steady state. The software pipeline has to fill before the first mma can
    // issue, and after the last mma the accumulators still have to be written
    // out. TileSight writes this as
    //     T = T_pro + max(N - d, 0) * T_steady + T_epi
    // (docs/PAPERS.md section 2).
    //
    // stages-1 is how many tiles are in flight before the first
    // cp.async.wait_group can retire, so it is how many tile loads happen with
    // nothing to overlap them.
    //
    // There is deliberately no fixed DRAM-latency term here. An earlier draft had
    // one. It was a guess, nothing in bench/calibrate/ measures it, and sweeping
    // it from 0 to 2400 cycles made both MAPE and rank correlation slightly
    // worse. See PLAN.md anti-pattern 10.
    const double gmemBytesPerCycle =
        hw.hbmBytesPerSec / (hw.clockHz * double(hw.numSMs));
    const double tileBytes = double(cfg.BM * cfg.BK + cfg.BK * cfg.BN) * db;
    const double kSteps = double(cfg.K) / cfg.BK;
    const double tPrologue = double(cfg.stages - 1) * tileBytes / gmemBytesPerCycle;
    // Accumulators are fp32 regardless of the input dtype, so 4 bytes per element.
    const double tEpilogue = double(cfg.BM) * cfg.BN * 4.0 / gmemBytesPerCycle;
    const double tBody = kSteps * tStep;
    const double envelopeEfficiency =
        tBody / (tPrologue + tBody + tEpilogue);

    p.kSteps = kSteps;
    p.tPrologueCycles = tPrologue;
    p.tEpilogueCycles = tEpilogue;
    p.tBodyCycles = tBody;
    p.envelopeEfficiency = envelopeEfficiency;

    // --- Two ceilings (SPEC 4.4) ---
    // Compute ceiling. The SM tensor peak is fixed. Occupancy does not multiply
    // it, it only hides latency (occFactor). Do NOT scale by ctasPerSM.
    const double peakTensorTFLOPS =
        (double(mma.flops) / hw.mmaCyclesPerInst) * hw.clockHz * hw.numSMs / 1e12;
    const double computeTFLOPS = peakTensorTFLOPS * smEfficiency * occFactor *
                                 waveEfficiency * envelopeEfficiency;
    p.peakTensorTFLOPS = peakTensorTFLOPS;
    p.computeTFLOPS = computeTFLOPS;

    // HBM roofline, with L2 reuse across the wave (PLAN.md Known Issue 1, now
    // closed).
    //
    // baseIntensity assumes each CTA fetches its own A row-panel and B col-panel
    // from DRAM with no sharing, which is wrong. The CTAs resident at one instant
    // are a wave, and a wave covers a roughly square region of the output, about
    // sqrt(ctaSlots) tiles on a side. Every CTA in a tile-row of that region wants
    // the same A panel and every CTA in a tile-column wants the same B panel, so
    // one DRAM fetch serves about sqrt(ctaSlots) consumers through L2. That is a
    // geometric argument about the wave, not a fitted constant, and it is the same
    // working-set reasoning tritonBLAS uses to pick its group size
    // (docs/PAPERS.md section 3).
    //
    // Still an idealisation: it assumes the wave is square and that L2 holds the
    // panels as long as the wave needs them, and it has no capacity term, so it
    // is optimistic for very large tiles.
    //
    // This must not be applied without the envelope above. On its own it takes
    // MAPE from 36.17% to 141.16%, because the too-low HBM ceiling had been
    // standing in for the missing fill and drain cost. See PLAN.md Finding 17.
    const double globalBytesPerCta = double(cfg.BM * cfg.K + cfg.K * cfg.BN) * db;
    const double flopsPerCta = 2.0 * double(cfg.BM) * cfg.BN * cfg.K;
    const double baseIntensity = flopsPerCta / globalBytesPerCta;
    const double l2ReuseFactor = std::sqrt(ctaSlots);
    const double arithmeticIntensity = baseIntensity * l2ReuseFactor;
    const double hbmTFLOPS = hw.hbmBytesPerSec * arithmeticIntensity / 1e12;
    p.baseIntensity = baseIntensity;
    p.l2ReuseFactor = l2ReuseFactor;
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
        const double worst =
            std::min(std::min(smEfficiency, occFactor),
                     std::min(waveEfficiency, envelopeEfficiency));
        if (worst >= 0.95)                      p.bottleneck = Bottleneck::TensorCore;
        else if (worst == waveEfficiency)       p.bottleneck = Bottleneck::Waves;
        else if (worst == envelopeEfficiency)   p.bottleneck = Bottleneck::Envelope;
        else if (worst == smEfficiency)         p.bottleneck = Bottleneck::Smem;
        else                                    p.bottleneck = Bottleneck::Occupancy;
    }

    return p;
}

} // namespace ridge
