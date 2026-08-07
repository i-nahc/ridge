#include "ridge/Hardware.h"

namespace ridge {

// Datasheet-derived placeholders for A100 80GB. Phase 3 calibration replaces
// these with measured values. mmaCyclesPerInst is back-solved so peak FP16
// tensor throughput is ~312 TFLOPS dense:
//   4096 flops/mma / 2.0 cycles * 1.41e9 Hz * 108 SMs ~= 312e12 FLOP/s.
HardwareModel HardwareModel::a100() {
    HardwareModel h;
    h.name = "A100-80GB (datasheet placeholders, replace with calibration)";
    h.numSMs = 108;
    h.clockHz = 1.41e9;
    h.mmaCyclesPerInst = 2.0;
    h.smemBytesPerCycle = 128.0;   // 32 banks * 4 bytes
    h.hbmBytesPerSec = 2.0e12;     // HBM2e ~ 2.0 TB/s
    h.regsPerSM = 65536;
    h.smemBytesPerSM = 167936;     // 164 KB max shared per SM
    h.maxWarpsPerSM = 64;
    h.maxCtasPerSM = 32;
    h.warpsNeededToHide = 16.0;
    return h;
}

} // namespace ridge
