#include "ridge/mma.h"

namespace ridge {

MmaShape mmaShapeFor(DType dtype) {
    switch (dtype) {
        case DType::FP16:
            // sm_80 mma.sync.aligned.m16n8k16.f16
            return MmaShape{16, 8, 16, 2LL * 16 * 8 * 16};
    }
    return MmaShape{16, 8, 16, 2LL * 16 * 8 * 16};
}

} // namespace ridge
