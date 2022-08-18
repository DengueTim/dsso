#pragma once

#include "util/NumType.h"

namespace dso {

inline bool MatEq(const MatXX &lhs, const MatXX &rhs) {
	return lhs.isApprox(rhs, 1e-4);
}

}
