#include "ridge/model.h"
#include "ridge/hardware.h"

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

int main(int argc, char** argv) {
    GemmConfig cfg;
    std::string gpu = "a100";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](long long def) -> long long {
            return (i + 1 < argc) ? std::atoll(argv[++i]) : def;
        };
        if (a == "--gpu")         gpu = (i + 1 < argc) ? argv[++i] : gpu;
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
    if (gpu == "a100") {
        hw = HardwareModel::a100();
    } else {
        std::fprintf(stderr, "unknown gpu: %s (only 'a100' in v1)\n", gpu.c_str());
        return 1;
    }

    const Prediction p = predict(cfg, hw);

    std::printf("GPU: %s\n", hw.name.c_str());
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
    return 0;
}
