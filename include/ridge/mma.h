#pragma once
#include <cstdint>
#include "ridge/config.h"

namespace ridge {

struct MmaShape {
    int m, n, k;
    int64_t flops; // 2 * m * n * k
};

// The warp-level MMA instruction shape for a dtype. v1: sm_80 mma.sync FP16.
MmaShape mmaShapeFor(DType dtype);

} // namespace ridge
