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

#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace dso {

PointHessian::PointHessian(const ImmaturePoint *const rawPoint) {
	instanceCounter++;
	host = rawPoint->host;

	numGoodResiduals = 0;

	// set static values & initialization.
	u = rawPoint->u;
	v = rawPoint->v;
	assert(std::isfinite(rawPoint->idepth_max));
	//idepth_init = rawPoint->idepth_GT;

	my_type = rawPoint->my_type;

	setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min) * 0.5);
	setPointStatus(PointHessian::INACTIVE);

	int n = patternNum;
	memcpy(color, rawPoint->color, sizeof(float) * n);
	memcpy(weights, rawPoint->weights, sizeof(float) * n);
	energyTH = rawPoint->energyTH;
}

void PointHessian::release() {
	for (unsigned int i = 0; i < residuals.size(); i++)
		delete residuals[i];
	residuals.clear();
}

// Call when setting the Eval_PT. Takes current state as state_zero and updates nullspaces.
void FrameHessian::setStateZero() {
	assert(state.head<6>().squaredNorm() < 1e-20);

	this->state_zero = state;

	for (int i = 0; i < 6; i++) {
		Vec6 eps;
		eps.setZero();
		eps[i] = 1e-3;
		SE3 EepsP = Sophus::SE3d::exp(eps);
		SE3 EepsM = Sophus::SE3d::exp(-eps);
		SE3 w2c_leftEps_P_x0 = (worldToCam_evalPT * EepsP) * worldToCam_evalPT.inverse();
		SE3 w2c_leftEps_M_x0 = (worldToCam_evalPT * EepsM) * worldToCam_evalPT.inverse();
		nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);
	}
	//nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
	//nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

	// scale change
	SE3 w2c_leftEps_P_x0 = worldToCam_evalPT;
	w2c_leftEps_P_x0.translation() *= 1.00001;
	w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * worldToCam_evalPT.inverse();
	SE3 w2c_leftEps_M_x0 = worldToCam_evalPT;
	w2c_leftEps_M_x0.translation() /= 1.00001;
	w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * worldToCam_evalPT.inverse();
	nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);

	nullspaces_affine.setZero();
	nullspaces_affine.topLeftCorner<2, 1>() = Vec2(1, 0);
	assert(ab_exposure > 0);
	nullspaces_affine.topRightCorner<2, 1>() = Vec2(0, expf(aff_g2l_0().a) * ab_exposure);
}
;

void FrameHessian::release() {
	// DELETE POINT
	// DELETE RESIDUAL
	for (unsigned int i = 0; i < pointHessians.size(); i++)
		delete pointHessians[i];
	for (unsigned int i = 0; i < pointHessiansMarginalized.size(); i++)
		delete pointHessiansMarginalized[i];
	for (unsigned int i = 0; i < pointHessiansOut.size(); i++)
		delete pointHessiansOut[i];
	for (unsigned int i = 0; i < immaturePoints.size(); i++)
		delete immaturePoints[i];

	pointHessians.clear();
	pointHessiansMarginalized.clear();
	pointHessiansOut.clear();
	immaturePoints.clear();
}

template<bool makeRightPyramid>
void FrameHessian::makeImages(float *color, float *colorR, CalibHessian *HCalib) {

	for (int i = 0; i < pyrLevelsUsed; i++) {
		dIp[i] = new Eigen::Vector3f[wG[i] * hG[i]];
		if (makeRightPyramid || i == 0)
			dIrp[i] = new Eigen::Vector3f[wG[i] * hG[i]];
		absSquaredGrad[i] = new float[wG[i] * hG[i]];
	}

	dI = dIp[0];
	dIr = dIrp[0];

	// make d0
	int w = wG[0];
	int h = hG[0];

	// Dump image data for unit test..
//	std::ofstream os("LeftRightDump.bin", std::ios::binary);
//	for (int i = 0; i < w * h; i++) {
//		os.write((char*) &color[i], sizeof(float));
//		os.write((char*) &colorR[i], sizeof(float));
//	}
//	os.close();

	for (int i = 0; i < w * h; i++) {
		dI[i][0] = color[i];
		dIr[i][0] = colorR[i];
	}

	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
		int wl = wG[lvl], hl = hG[lvl];
		Eigen::Vector3f *dI_l = dIp[lvl];
		Eigen::Vector3f *dIr_l = dIrp[lvl];

		float *dabs_l = absSquaredGrad[lvl];
		if (lvl > 0) {
			int lvlm1 = lvl - 1;
			int wlm1 = wG[lvlm1];
			Eigen::Vector3f *dI_lm = dIp[lvlm1];

			for (int y = 0; y < hl; y++)
				for (int x = 0; x < wl; x++) {
					dI_l[x + y * wl][0] = 0.25f
							* (dI_lm[2 * x + 2 * y * wlm1][0] + dI_lm[2 * x + 1 + 2 * y * wlm1][0]
									+ dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] + dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);

					if (makeRightPyramid) {
						Eigen::Vector3f *dIr_lm = dIrp[lvlm1];
						dIr_l[x + y * wl][0] = 0.25f
								* (dIr_lm[2 * x + 2 * y * wlm1][0] + dIr_lm[2 * x + 1 + 2 * y * wlm1][0]
										+ dIr_lm[2 * x + 2 * y * wlm1 + wlm1][0] + dIr_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
					}
				}
		}

		for (int idx = wl; idx < wl * (hl - 1); idx++) {
			float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
			float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

			if (!std::isfinite(dx))
				dx = 0;
			if (!std::isfinite(dy))
				dy = 0;

			dI_l[idx][1] = dx;
			dI_l[idx][2] = dy;

			dabs_l[idx] = dx * dx + dy * dy;

			if (setting_gammaWeightsPixelSelect == 1 && HCalib != 0) {
				float gw = HCalib->getBGradOnly((float) (dI_l[idx][0]));
				dabs_l[idx] *= gw * gw;	// convert to gradient of original color space (before removing response).
			}

			if (makeRightPyramid || lvl == 0) {
				dx = 0.5f * (dIr_l[idx + 1][0] - dIr_l[idx - 1][0]);
				dy = 0.5f * (dIr_l[idx + wl][0] - dIr_l[idx - wl][0]);

				if (!std::isfinite(dx))
					dx = 0;
				if (!std::isfinite(dy))
					dy = 0;

				dIr_l[idx][1] = dx;
				dIr_l[idx][2] = dy;
			}
		}
	}
}

template void FrameHessian::makeImages<false>(float *color, float *colorR, CalibHessian *HCalib);
template void FrameHessian::makeImages<true>(float *color, float *colorR, CalibHessian *HCalib);

void FrameFramePrecalc::set(FrameHessian *host, FrameHessian *target, CalibHessian *HCalib) {
	this->host = host;
	this->target = target;

	if (host == target) {
		SE3 leftToRightZero = HCalib->getLeftToRightZero();
		PRE_RTll_0 = leftToRightZero.rotationMatrix().cast<float>();
		PRE_tTll_0 = leftToRightZero.translation().cast<float>();

		SE3 leftToRight = HCalib->getLeftToRight(); // SE3::exp(value_scaled.segment<6>(8));
		PRE_RTll = leftToRight.rotationMatrix().cast<float>();
		PRE_tTll = leftToRight.translation().cast<float>();
		distanceLL = leftToRight.translation().norm();

		Mat33f KL = Mat33f::Zero();
		KL(0, 0) = HCalib->fxl();
		KL(1, 1) = HCalib->fyl();
		KL(0, 2) = HCalib->cxl();
		KL(1, 2) = HCalib->cyl();
		KL(2, 2) = 1;

		Mat33f KR = Mat33f::Zero();
		KR(0, 0) = HCalib->fxlR();
		KR(1, 1) = HCalib->fylR();
		KR(0, 2) = HCalib->cxlR();
		KR(1, 2) = HCalib->cylR();
		KR(2, 2) = 1;

		PRE_KRKiTll = KR * PRE_RTll * KL.inverse();
		PRE_RKiTll = PRE_RTll * KL.inverse();
		PRE_KtTll = KR * PRE_tTll;

		// Not estimating exposure params between L/R images. Assuming they are the same.
		PRE_aff_mode = Vec2f(1.0, 0.0);
		PRE_b0_mode = 0; //host->aff_g2l_0().b;
	} else {
		// evalPT set when frame added and after GN optimisation.
		// The paper says the evalPT is fixed when any residual dependent on the transform is marginalised...
		// Guess if it's only set in the above cases it will be fixed when marginalizing points/frames??..
		SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
		PRE_RTll_0 = (hostToTarget.rotationMatrix()).cast<float>();
		PRE_tTll_0 = (hostToTarget.translation()).cast<float>();

		// PREcalculated transform set when evalPT updated and for every GN step.
		SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
		PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
		PRE_tTll = (leftToLeft.translation()).cast<float>();
		distanceLL = leftToLeft.translation().norm();

		Mat33f K = Mat33f::Zero();
		K(0, 0) = HCalib->fxl();
		K(1, 1) = HCalib->fyl();
		K(0, 2) = HCalib->cxl();
		K(1, 2) = HCalib->cyl();
		K(2, 2) = 1;
		PRE_KRKiTll = K * PRE_RTll * K.inverse();
		PRE_RKiTll = PRE_RTll * K.inverse();
		PRE_KtTll = K * PRE_tTll;

		PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<
				float>();
		PRE_b0_mode = host->aff_g2l_0().b;
	}
}

}

