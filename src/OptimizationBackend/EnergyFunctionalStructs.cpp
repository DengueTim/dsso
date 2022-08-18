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

#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

void EFResidual::takeDataF() {
	std::swap<RawResidualJacobian*>(J, data->J);

	Vec2f JI_JI_Jd = J->JIdx2 * J->Jpdd;

	bool leftToRight = hostIDX == targetIDX;
	const Vec6f &JpdxiX = leftToRight ? J->Jpdc[0].segment(CIPARS, 6) : J->Jpdxi[0];
	const Vec6f &JpdxiY = leftToRight ? J->Jpdc[1].segment(CIPARS, 6) : J->Jpdxi[1];

	for (int i = 0; i < 6; i++)
		JpJdF[i] = JpdxiX[i] * JI_JI_Jd[0] + JpdxiY[i] * JI_JI_Jd[1];

	JpJdF.segment<2>(6) = J->JabJIdx * J->Jpdd;

	EnergyFunctional *ef = host->ef;
	int ij = hostIDX + ef->nFrames * targetIDX;
#ifndef ADD_LR_RESIDUALS	
	assert (hostIDX != targetIDX);
#endif
	JpJdAdH = ef->adHostF[ij] * JpJdF;
	JpJdAdT = ef->adTargetF[ij] * JpJdF;

	// I saw a NAN once...
	assert(!std::isnan(JpJdAdH.sum()));
	assert(!std::isnan(JpJdAdT.sum()));
}

EFFrame::EFFrame(EnergyFunctional *ef_, FrameHessian *fh_) :
		ef(ef_), fh(fh_) {
	prior = fh->getPrior().head<8>();
	delta = fh->get_state_minus_stateZero().head<8>();
	delta_prior = (fh->get_state() - fh->getPriorZero()).head<8>();

//	Vec10 state_zero =  data->get_state_zero();
//	state_zero.segment<3>(0) = SCALE_XI_TRANS * state_zero.segment<3>(0);
//	state_zero.segment<3>(3) = SCALE_XI_ROT * state_zero.segment<3>(3);
//	state_zero[6] = SCALE_A * state_zero[6];
//	state_zero[7] = SCALE_B * state_zero[7];
//	state_zero[8] = SCALE_A * state_zero[8];
//	state_zero[9] = SCALE_B * state_zero[9];
//
//	std::cout << "state_zero: " << state_zero.transpose() << "\n";

	assert(fh->keyFrameID != -1);

	keyFrameID = fh->keyFrameID;
	idxInFrames = -1;
}

EFPoint::EFPoint(PointHessianBase *ph_, EFFrame *host_) :
		ph(ph_), host(host_) {

	if (ph->hasDepthPrior && !(setting_solverMode & SOLVER_REMOVE_POSEPRIOR )) {
		priorF = setting_idepthFixPrior * SCALE_IDEPTH * SCALE_IDEPTH;
	} else {
		priorF = 0;
	}

	deltaF = ph->idepth - ph->idepth_zero;
	stateFlag = EFPointStatus::PS_GOOD;
	HdiF = 0.0;
	Hdd_accAF = 0.0;
	Hdd_accLF = 0.0;
	bd_accAF = 0.0;
	bd_accLF = 0.0;
	bdSumF = 0.0;
	idxInPoints = -1;
}

void EFResidual::fixLinearizationF(EnergyFunctional *ef) {
	bool leftToRight = hostIDX == targetIDX;
	const Vec6f &JpdxiX = leftToRight ? J->Jpdc[0].segment(CIPARS, 6) : J->Jpdxi[0];
	const Vec6f &JpdxiY = leftToRight ? J->Jpdc[1].segment(CIPARS, 6) : J->Jpdxi[1];

	float dd = point->deltaF;
	Vec8f dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX];
	VecCf dc = ef->cDelta.cast<float>();

	float Jp_delta_x_1 = J->Jpdd[0] * dd;
	float Jp_delta_y_1 = J->Jpdd[1] * dd;

	// compute Jp*delta
	if (leftToRight) {
		Jp_delta_x_1 += J->Jpdc[0].dot(dc);
		Jp_delta_y_1 += J->Jpdc[1].dot(dc);
	} else {
		Jp_delta_x_1 += J->Jpdc[0].head<4>().dot(dc.head<4>());
		Jp_delta_x_1 += J->Jpdxi[0].dot(dp.head<6>());
		Jp_delta_y_1 += J->Jpdc[1].head<4>().dot(dc.head<4>());
		Jp_delta_y_1 += J->Jpdxi[1].dot(dp.head<6>());
	}

	__m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
	__m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
	__m128 delta_a = _mm_set1_ps((float) (dp[6]));
	__m128 delta_b = _mm_set1_ps((float) (dp[7]));

	for (int i = 0; i < patternNum; i += 4) {
		// PATTERN: rtz = resF - [JI*Jp Ja]*delta.
		__m128 rtz = _mm_load_ps(((float*) &J->resF) + i);
		rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*) (J->JIdx)) + i), Jp_delta_x));
		rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*) (J->JIdx + 1)) + i), Jp_delta_y));
		rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*) (J->JabF)) + i), delta_a));
		rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*) (J->JabF + 1)) + i), delta_b));
		_mm_store_ps(((float*) &res_toZeroF) + i, rtz);
	}

	isLinearized = true;
}

}
