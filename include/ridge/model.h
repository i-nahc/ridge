#pragma once
#include "ridge/config.h"
#include "ridge/hardware.h"
#include "ridge/prediction.h"

namespace ridge {

// Predict sustained throughput and the bottleneck for one GEMM config on one
// GPU. Implements SPEC section 4 (the multi-stage roofline). No GPU needed.
Prediction predict(const GemmConfig& cfg, const HardwareModel& hw);

} // namespace ridge
