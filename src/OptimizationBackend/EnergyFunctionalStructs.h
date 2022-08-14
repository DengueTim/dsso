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

#include "util/NumType.h"
#include "vector"
#include <math.h>
#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso {

class PointFrameResidual;
class CalibHessian;
class FrameHessian;
class PointHessianBase;

class EFResidual;
class EFPoint;
class EFFrame;
class EnergyFunctional;

class EFResidual {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;

	inline EFResidual(PointFrameResidual *org, EFPoint *point_, EFFrame *host_, EFFrame *target_) :
			data(org), point(point_), host(host_), target(target_) {
		isLinearized = false;
		isActive = false;
		J = new RawResidualJacobian();
		assert(((long )this) % 16 == 0);
		assert(((long )J) % 16 == 0);
	}
	inline ~EFResidual() {
		delete J;
	}

	void takeDataF();

	void fixLinearizationF(EnergyFunctional *ef);

	// structural pointers
	PointFrameResidual *data;
	int hostIDX, targetIDX;
	EFPoint *point;
	EFFrame *host;
	EFFrame *target;
	int idxInAll;

	RawResidualJacobian *J;

	VecNRf res_toZeroF; // Set when point is marginalized.
	Vec8f JpJdF; // How pose changes with depth?
	Vec8f JpJdAdH; // Above in host frames pose tangent space.
	Vec8f JpJdAdT; // Above in target...

	// status.
	bool isLinearized;

	// if residual is not OOB & not OUTLIER & should be used during accumulations
	bool isActive; // was isActiveAndIsGoodNEW.
};

enum EFPointStatus {
	PS_GOOD = 0, PS_MARGINALIZE, PS_DROP
};

class EFPoint {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;
	EFPoint(PointHessianBase *ph_, EFFrame *host_);

	PointHessianBase *ph;

	float priorF;
	float deltaF;

	// constant info (never changes in-between).
	int idxInPoints;
	EFFrame *host;

	/* contains all residuals.
	 * Elements are the residual between the points host frame and the n>0 target frames the point appears in.
	 */
	std::vector<EFResidual*> residualsAll;

	float bdSumF; // Sum of bd_accLF + bd_accAF (+ prior..)?
	/* Computed/set in Acc.SCHessian.addPoint().
	 * 1 / (Hdd_accAF + Hdd_accLF + priorF).  1 / "idepth_hessian"
	 * Used for weighting depth step. HdiF -> Hessian Inverse Depth Float
	 * Point diagonal element of Schur Comp. Hbb^-1 in the paper (eq.17)?
	 */
	float HdiF;
	float Hdd_accLF; // dRes^2/dDepth^2 diagonal element of depth/depth bit of Hessian for Linearized point.
	VecCf Hcd_accLF; // Depth to Camera params Hessian value Accumulator for Linearized point.
	float bd_accLF;  // Depth B(residual) value Accumulator for Linearized point.
	float Hdd_accAF; // dRes^2/dDepth^2 diagonal element of depth/depth bit of Hessian for Active point.
	VecCf Hcd_accAF; // Depth to Camera params Hessian value Accumulator for Active point.
	float bd_accAF;  // Depth B(residual) value Accumulator for Active point.

	EFPointStatus stateFlag;
};

class EFFrame {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;
	EFFrame(EnergyFunctional *ef_, FrameHessian *fh_);

	EnergyFunctional *ef;

	Vec8 prior;				// prior hessian (diagonal)
	Vec8 delta_prior;		// = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
	Vec8 delta;				// state - state_zero.

	std::vector<EFPoint*> points;
	FrameHessian *fh;
	int idxInFrames;

	int keyFrameID;
};

}

