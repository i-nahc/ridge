#include "ridge/model.h"
#include "ridge/hardware.h"

#include <fstream>
#include <sstream>
#include <vector>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace ridge;

static void usage() {
    std::printf(
        "ridge-predict: predict tensor-core GEMM throughput and bottleneck.\n"
        "Usage:\n"
        "  ridge-predict [--gpu a100] [--m M --n N --k K]\n"
        "                [--bm BM --bn BN --bk BK --warpm WM --warpn WN]\n"
        "                [--stages S --regs R]\n"
        "Defaults: A100, 4096^3 GEMM, 128x128x32 tile, 64x64 warp, 3 stages.\n");
}

// Batch mode. Reads a CSV of configs and writes the same rows with the model's
// prediction columns appended.
//
// This exists so bench/validate.py never reimplements the model. A Python copy
// of the math would be a second source of truth that drifts silently, and the
// first symptom would be a validation result that describes a model nobody
// ships. The C++ model stays the only implementation and Python does statistics.
//
// Input needs the columns this reads by name. Extra columns are passed through
// untouched, so the measured CSV from bench/measure.cu can be fed in directly
// and comes back with predictions attached, already joined.
static int runBatch(const std::string& inPath, const std::string& outPath,
                    const HardwareModel& hw) {
    std::ifstream in(inPath);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", inPath.c_str());
        return 1;
    }
    std::ofstream out(outPath);
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", outPath.c_str());
        return 1;
    }

    std::string line;
    std::vector<std::string> header;
    bool haveHeader = false;
    int rows = 0;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') { out << line << "\n"; continue; }

        std::vector<std::string> cells;
        std::string cell;
        std::istringstream ss(line);
        while (std::getline(ss, cell, ',')) cells.push_back(cell);

        if (!haveHeader) {
            header = cells;
            haveHeader = true;
            out << line
                << ",pred_tflops,pred_bottleneck,pred_compute_ceiling,"
                   "pred_hbm_ceiling,pred_smem_eff,pred_occ_factor,"
                   "pred_arith_intensity,pred_ctas_per_sm\n";
            continue;
        }

        auto col = [&](const char* name, long long def) -> long long {
            for (size_t i = 0; i < header.size() && i < cells.size(); ++i) {
                if (header[i] == name) return std::atoll(cells[i].c_str());
            }
            return def;
        };

        GemmConfig cfg;
        cfg.M = col("M", cfg.M);
        cfg.N = col("N", cfg.N);
        cfg.K = col("K", cfg.K);
        cfg.BM = int(col("BM", cfg.BM));
        cfg.BN = int(col("BN", cfg.BN));
        cfg.BK = int(col("BK", cfg.BK));
        cfg.warpM = int(col("WM", cfg.warpM));
        cfg.warpN = int(col("WN", cfg.warpN));
        cfg.stages = int(col("stages", cfg.stages));
        cfg.regsPerThread = int(col("regsPerThread", cfg.regsPerThread));

        const Prediction p = predict(cfg, hw);
        out << line << ',' << p.predictedTFLOPS << ',' << toString(p.bottleneck)
            << ',' << p.computeTFLOPS << ',' << p.hbmTFLOPS << ','
            << p.smEfficiency << ',' << p.occFactor << ','
            << p.arithmeticIntensity << ',' << p.ctasPerSM << '\n';
        ++rows;
    }

    std::fprintf(stderr, "predicted %d rows -> %s (constants: %s)\n", rows,
                 outPath.c_str(),
                 hw.calibrated ? "CALIBRATED" : "PLACEHOLDER");
    return 0;
}

int main(int argc, char** argv) {
    GemmConfig cfg;
    std::string gpu = "a100";
    std::string hwPath;
    std::string batchIn, batchOut;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](long long def) -> long long {
            return (i + 1 < argc) ? std::atoll(argv[++i]) : def;
        };
        if (a == "--gpu")         gpu = (i + 1 < argc) ? argv[++i] : gpu;
        else if (a == "--hw")     hwPath = (i + 1 < argc) ? argv[++i] : hwPath;
        else if (a == "--batch")  batchIn = (i + 1 < argc) ? argv[++i] : batchIn;
        else if (a == "--out")    batchOut = (i + 1 < argc) ? argv[++i] : batchOut;
        else if (a == "--m")      cfg.M = next(cfg.M);
        else if (a == "--n")      cfg.N = next(cfg.N);
        else if (a == "--k")      cfg.K = next(cfg.K);
        else if (a == "--bm")     cfg.BM = int(next(cfg.BM));
        else if (a == "--bn")     cfg.BN = int(next(cfg.BN));
        else if (a == "--bk")     cfg.BK = int(next(cfg.BK));
        else if (a == "--warpm")  cfg.warpM = int(next(cfg.warpM));
        else if (a == "--warpn")  cfg.warpN = int(next(cfg.warpN));
        else if (a == "--stages") cfg.stages = int(next(cfg.stages));
        else if (a == "--regs")   cfg.regsPerThread = int(next(cfg.regsPerThread));
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(); return 1; }
    }

    HardwareModel hw;
    if (!hwPath.empty()) {
        std::string err;
        if (!HardwareModel::loadFromJson(hwPath, hw, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
    } else if (gpu == "a100") {
        hw = HardwareModel::a100();
    } else {
        std::fprintf(stderr, "unknown gpu: %s (only 'a100' in v1)\n", gpu.c_str());
        return 1;
    }

    if (!batchIn.empty()) {
        return runBatch(batchIn, batchOut.empty() ? "predictions.csv" : batchOut, hw);
    }

    const Prediction p = predict(cfg, hw);

    std::printf("GPU: %s\n", hw.name.c_str());
    // Say plainly whether these numbers rest on measurements or on placeholders.
    // A prediction from an uncalibrated model is indicative only, and the
    // difference should never have to be inferred from context.
    std::printf("  constants: %s\n",
                hw.calibrated ? "CALIBRATED (measured on this GPU)"
                              : "PLACEHOLDER (datasheet-derived, indicative only)");
    std::printf("GEMM %lldx%lldx%lld  tile %dx%dx%d  warp %dx%d  stages %d\n",
                (long long)cfg.M, (long long)cfg.N, (long long)cfg.K,
                cfg.BM, cfg.BN, cfg.BK, cfg.warpM, cfg.warpN, cfg.stages);

    if (p.bottleneck == Bottleneck::DoesNotFit) {
        std::printf("  DOES NOT FIT: shared memory or registers exceed the SM.\n");
        return 0;
    }

    std::printf("  predicted:    %8.1f TFLOP/s   bottleneck: %s\n",
                p.predictedTFLOPS, toString(p.bottleneck));
    std::printf("  peak tensor:  %8.1f   compute ceiling: %8.1f   HBM ceiling: %8.1f\n",
                p.peakTensorTFLOPS, p.computeTFLOPS, p.hbmTFLOPS);
    std::printf("  smem eff: %.2f   occ factor: %.2f   arith intensity: %.1f FLOP/byte\n",
                p.smEfficiency, p.occFactor, p.arithmeticIntensity);
    std::printf("  warps/CTA: %d   CTAs/SM: %d\n", p.numWarps, p.ctasPerSM);
    std::printf("  grid: %.0f CTAs in %.0f slots   waves: %.2f   wave eff: %.3f\n",
                p.totalCtas, p.ctaSlots, p.wavesExact, p.waveEfficiency);
    std::printf("  step: %.0f cyc = %.0f mma + %.0f ldmatrix (in series)\n",
                p.tStepCycles, p.tComputeCycles, p.tSmemCycles);
    std::printf("  envelope eff: %.3f   %.0f K-steps   prologue %.0f cyc   epilogue %.0f cyc\n",
                p.envelopeEfficiency, p.kSteps, p.tPrologueCycles, p.tEpilogueCycles);
    std::printf("  L2 reuse: %.1fx  (base intensity %.1f -> %.1f FLOP/byte)\n",
                p.l2ReuseFactor, p.baseIntensity, p.arithmeticIntensity);
    return 0;
}
