#pragma once
#include "ridge/Config.h"
#include "ridge/Hardware.h"
#include "ridge/Prediction.h"

namespace ridge {

// Predict sustained throughput and the bottleneck for one GEMM config on one
// GPU. Implements SPEC section 4 (the multi-stage roofline). No GPU needed.
Prediction predict(const GemmConfig& cfg, const HardwareModel& hw);

} // namespace ridge
