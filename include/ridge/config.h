#pragma once
#include <cstdint>

namespace ridge {

// Data types supported in v1. FP16 accumulate FP32 is the only one wired now.
enum class DType { FP16 };

inline int dtypeBytes(DType d) {
    switch (d) {
        case DType::FP16: return 2;
    }
    return 2;
}

// One tensor-core GEMM tile configuration. This is the flat v1 input. Phase 6
// refactors it into a general TilePlan (see SPEC section 3.5). Fields that are
// 0 for the MMA shape are filled from Mma.h at predict time.
struct GemmConfig {
    // Problem size.
    int64_t M = 4096, N = 4096, K = 4096;
    DType dtype = DType::FP16;

    // CTA-level tile.
    int BM = 128, BN = 128, BK = 32;

    // Warp-level tile. numWarps = (BM/warpM) * (BN/warpN).
    int warpM = 64, warpN = 64;

    // Warp-level MMA instruction shape. Left at 0 means "use the default for the
    // dtype" (filled by the model from Mma.h).
    int mmaM = 0, mmaN = 0, mmaK = 0;

    // Software-pipeline depth (number of shared-memory buffers).
    int stages = 3;

    // Registers per thread, from `ptxas -v` on the real kernel. 0 means unknown,
    // in which case register-based occupancy is skipped.
    int regsPerThread = 128;
};

} // namespace ridge
