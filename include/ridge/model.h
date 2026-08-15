#pragma once
#include "ridge/config.h"
#include "ridge/hardware.h"
#include "ridge/prediction.h"

namespace ridge {

// Predict sustained throughput and the binding resource for one GEMM config on
// one GPU. Closed form, no GPU needed.
Prediction predict(const GemmConfig& cfg, const HardwareModel& hw);

} // namespace ridge
