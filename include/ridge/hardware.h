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
    //
    // WARNING: hbmBytesPerSec here is the A100 80GB figure while the project
    // measures on a 40GB card, so this model currently describes hardware that
    // is not under test. See PLAN.md Finding 10. Prefer loadFromJson.
    static HardwareModel a100();

    // Loads a calibrated model from a data/hardware/*.json written by
    // bench/calibrate/run-calibration.py.
    //
    // Returns false and leaves `out` untouched if the file is missing,
    // unparseable, or missing any required key. Partially loading a hardware
    // model would be worse than not loading one: the caller would get a mix of
    // measured and placeholder constants with no way to tell which was which,
    // and would then report predictions as calibrated. So this is all or
    // nothing, and `err` says what went wrong.
    static bool loadFromJson(const std::string& path, HardwareModel& out,
                             std::string& err);

    // True when every [CAL] constant came from a calibration file rather than
    // from the built-in placeholders. Predictions from an uncalibrated model are
    // indicative only and anything reporting results should say so.
    bool calibrated = false;
};

} // namespace ridge
