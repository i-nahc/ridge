#pragma once
#include <string>

namespace ridge {

// Per-GPU hardware model. Constants marked [CAL] in the spec should be replaced
// by microbenchmark calibration (Phase 3). The built-in factory below holds
// datasheet-derived placeholders so Phase 1 runs with no GPU.
struct HardwareModel {
    std::string name;
    int numSMs = 0;
    double clockHz = 0.0;

    // [CAL] effective SM-level cycles to issue one MMA instruction. Folds in all
    // tensor cores and warp schedulers on the SM.
    double mmaCyclesPerInst = 0.0;

    // [CAL] sustained shared-memory read bandwidth, bytes per cycle per SM.
    double smemBytesPerCycle = 0.0;

    // [CAL] sustained HBM bandwidth for the whole GPU, bytes per second.
    double hbmBytesPerSec = 0.0;

    // Occupancy limits.
    int regsPerSM = 0;
    int smemBytesPerSM = 0;
    int maxWarpsPerSM = 0;
    int maxCtasPerSM = 0;

    // [CAL] active warps per SM needed to hide MMA and memory latency.
    double warpsNeededToHide = 0.0;

    // Built-in placeholder model. Phase 3 replaces this by loading a calibrated
    // data/hardware/*.json.
    static HardwareModel a100();
};

} // namespace ridge
