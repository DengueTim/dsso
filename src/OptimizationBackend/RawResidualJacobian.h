/**
 * This file is part of DSO.
 * 
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <iostream>

#include "util/NumType.h"

namespace dso {
// Residual for a point. Although parts are the residuals on the pixels that belong to that point.
struct RawResidualJacobian {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;
	// ================== new structure: save independently =============.

	// resF, Jpdxi, Jpdc & Jpdd are linearised at the first estimate.  First Estimate Jacobian FEJ

	// Per pixel(N per point) Huber weighted residual (ab adjusted already). Assigned in PointFrameResidual::linearize()
	VecNRf resF;

	// the two rows of d[x,y]/d[xi].  How image point x & y change with pose estimate.
	Vec6f Jpdxi[2];			// 2x6

	// the two rows of d[x,y]/d[C].  How image point x & y change with Camera intrinsics estimate.
	VecCf Jpdc[2];			// 2x14

	// the two rows of d[x,y]/d[idepth].  How image point x & y change with inverse depth estimate.
	Vec2f Jpdd;				// 2x1

	// JIdx and JabF are linearised at the current estimate. (GN step?)

	// The two columns of d[r]/d[x,y]. How point pixel residuals change with (point/pixel?) x,y position. Basically the weighted image gradient around pixel.
	VecNRf JIdx[2];			// MAX_RES_PER_POINT(8)x2

	// = the two columns of d[r] / d[ab]. How point pixel residuals change with estimated frame a and b values(affine brightness).
	VecNRf JabF[2];			// MAX_RES_PER_POINT(8)x2

	// JIdx, JabJIdx & Jab2 are derived from Jidx & JabF and are just stored for efficiency.

	/* = JIdx^T * JIdx (inner product). Only as a shorthand.
	 * How sum of squared residuals changes with
	 * [ xx, xy ]
	 * [ yx, yy ]
	 * x & y being the point position in the image.
	 */
	Mat22f JIdx2;			// 2x2
	/* = Jab^T * JIdx (inner product). Only as a shorthand.
	 * How sum of squared residuals changes with
	 * [ xa, xb ]
	 * [ ya, yb ]
	 * x & y being the point position in the image.
	 * a & b being the affine brightness factors
	 */
	Mat22f JabJIdx;			// 2x2
	/* = Jab^T * Jab (inner product). Only as a shorthand.
	 * How sum of squared residuals changes with
	 * [ aa, ab ]
	 * [ ba, bb ]
	 *  * a & b being the affine brightness factors
	 */
	Mat22f Jab2;			// 2x2

	void print() {
		std::cout << "\tJ->resF" << resF.format(MatFormatInit) << "\n";
		std::cout << "\tJ->Jpdxi[0]" << Jpdxi[0].format(MatFormatInit) << "\n";
		std::cout << "\tJ->Jpdxi[1]" << Jpdxi[1].format(MatFormatInit) << "\n";
		std::cout << "\tJ->Jpdc[0]" << Jpdc[0].format(MatFormatInit) << "\n";
		std::cout << "\tJ->Jpdc[1]" << Jpdc[1].format(MatFormatInit) << "\n";
		std::cout << "\tJ->Jpdd" << Jpdd.format(MatFormatInit) << "\n";
		std::cout << "\tJ->JIdx[0]" << JIdx[0].format(MatFormatInit) << "\n";
		std::cout << "\tJ->JIdx[1]" << JIdx[1].format(MatFormatInit) << "\n";
		std::cout << "\tJ->JabF[0]" << JabF[0].format(MatFormatInit) << "\n";
		std::cout << "\tJ->JabF[1]" << JabF[1].format(MatFormatInit) << "\n";
		std::cout << "\tJ->JIdx2" << JIdx2.format(MatFormatInit) << "\n";
		std::cout << "\tJ->JabJIdx" << JabJIdx.format(MatFormatInit) << "\n";
		std::cout << "\tJ->Jab2" << Jab2.format(MatFormatInit) << "\n";
		std::cout << "\n";
	}
};
}

