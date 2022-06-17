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

/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include <fstream>
#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "util/nanoflann.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

CoarseInitializer::CoarseInitializer(int ww, int hh) :
		thisToNext_aff(0, 0), thisToNext(SE3()) {
	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
		points[lvl] = 0;
		numPoints[lvl] = 0;
	}

	JbBuffer = new Vec10f[ww * hh];
	JbBuffer_new = new Vec10f[ww * hh];

	frameID = -1;
	fixAffine = true;
	printDebug = false;

	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}

CoarseInitializer::~CoarseInitializer() {
	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
		if (points[lvl] != 0)
			delete[] points[lvl];
	}

	delete[] JbBuffer;
	delete[] JbBuffer_new;
}

bool CoarseInitializer::trackFrame(FrameHessian *newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps) {
	newFrame = newFrameHessian;

	for (IOWrap::Output3DWrapper *ow : wraps)
		ow->pushLiveFrame(newFrameHessian);

	int maxIterations[] = { 5, 5, 10, 30, 50 };

	alphaK = 2.5 * 2.5; //*freeDebugParam1*freeDebugParam1;
	alphaW = 150 * 150; //*freeDebugParam2*freeDebugParam2;
	regWeight = 0.8; //*freeDebugParam4;
	couplingWeight = 1; //*freeDebugParam5;

	if (!snapped) {
		thisToNext.translation().setZero();
		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
			int npts = numPoints[lvl];
			Pnt *ptsl = points[lvl];
			for (int i = 0; i < npts; i++) {
				ptsl[i].iR = 1;
				ptsl[i].idepth_new = 1;
				ptsl[i].lastHessian = 0;
			}
		}
	}

	SE3 refToNew_current = thisToNext;
	AffLight refToNew_aff_current = thisToNext_aff;

	if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0)
		refToNew_aff_current = AffLight(logf(newFrame->ab_exposure / firstFrame->ab_exposure), 0); // coarse approximation.

	// From rough to fine..
	for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--) {

		if (lvl < pyrLevelsUsed - 1)
			propagateDown(lvl + 1);

		Mat88f H, Hsc;
		Vec8f b, bsc;
		resetPoints(lvl);
		Vec3f resOld = calcResidualAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current);
		applyStep(lvl);

		float lambda = 0.1;
		float eps = 1e-4;
		int fails = 0;

		if (printDebug) {
			printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t", lvl, 0, lambda, "INITIA",
					sqrtf((float) (resOld[0] / resOld[2])), sqrtf((float) (resOld[1] / resOld[2])),
					sqrtf((float) (resOld[0] / resOld[2])), sqrtf((float) (resOld[1] / resOld[2])),
					(resOld[0] + resOld[1]) / resOld[2], (resOld[0] + resOld[1]) / resOld[2], 0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() << "\n";
		}

		int iteration = 0;
		while (true) {
			Mat88f Hl = H;
			for (int i = 0; i < 8; i++)
				Hl(i, i) *= (1 + lambda);
			Hl -= Hsc * (1 / (1 + lambda));
			Vec8f bl = b - bsc * (1 / (1 + lambda));

			Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl]));
			bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));

			Vec8f inc;
			if (fixAffine) {
				inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() * (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
				inc.tail<2>().setZero();
			} else
				inc = -(wM * (Hl.ldlt().solve(bl)));	//=-H^-1 * b.

			SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
			AffLight refToNew_aff_new = refToNew_aff_current;
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];
			doIdepthStepUpdate(lvl, lambda, inc);

			Mat88f H_new, Hsc_new;
			Vec8f b_new, bsc_new;
			Vec3f resNew = calcResidualAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new);
			Vec3f regEnergy = calcDepthRegularisationCouplingEnergy(lvl);

			float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
			float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);

			bool accept = eTotalOld > eTotalNew;

			if (printDebug) {
				printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t", lvl,
						iteration, lambda, (accept ? "ACCEPT" : "REJECT"), sqrtf((float) (resOld[0] / resOld[2])),
						sqrtf((float) (regEnergy[0] / regEnergy[2])), sqrtf((float) (resOld[1] / resOld[2])),
						sqrtf((float) (resNew[0] / resNew[2])), sqrtf((float) (regEnergy[1] / regEnergy[2])),
						sqrtf((float) (resNew[1] / resNew[2])), eTotalOld / resNew[2], eTotalNew / resNew[2], inc.norm());
				std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() << "\n";
			}

			if (accept) {

				if (resNew[1] == alphaK * numPoints[lvl])
					snapped = true;
				H = H_new;
				b = b_new;
				Hsc = Hsc_new;
				bsc = bsc_new;
				resOld = resNew;
				refToNew_aff_current = refToNew_aff_new;
				refToNew_current = refToNew_new;
				applyStep(lvl);
				updateIdepthRegularisation(lvl);
				lambda *= 0.5;
				fails = 0;
				if (lambda < 0.0001)
					lambda = 0.0001;
			} else {
				fails++;
				lambda *= 4;
				if (lambda > 10000)
					lambda = 10000;
			}

			if (!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2) {
				break;
			}

			iteration++;
		}
	}

	thisToNext = refToNew_current;
	thisToNext_aff = refToNew_aff_current;

	for (int i = 0; i < pyrLevelsUsed - 1; i++)
		propagateUp(i);

	frameID++;
	if (!snapped)
		snappedAt = 0;

	if (snapped && snappedAt == 0)
		snappedAt = frameID;

	debugPlot(0, wraps);

	return snapped && frameID > snappedAt + 5;
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps) {
	bool needCall = false;
	for (IOWrap::Output3DWrapper *ow : wraps)
		needCall = needCall || ow->needPushDepthImage();
	if (!needCall)
		return;

	Mat33f RrKi = (leftToRight.rotationMatrix().cast<float>() * Ki[lvl]);
	Vec3f tr = (leftToRight.translation()).cast<float>();

	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];
	Eigen::Vector3f *colorRight = firstFrame->dIrp[lvl];

	MinimalImageB3 imgLeft(wl, hl);
	MinimalImageB3 imgRight(wl, hl);

	for (int i = 0; i < wl * hl; i++) {
		float c = colorRef[i][0];
		if (c >= 256.0) {
			c = 255.0;
		}
		imgLeft.at(i) = Vec3b(c, c, c);

		c = colorRight[i][0];
		if (c >= 256.0) {
			c = 255.0;
		}
		imgRight.at(i) = Vec3b(c, c, c);
	}

	int npts = numPoints[lvl];

	float nid = 0, sid = 0;
	for (int i = 0; i < npts; i++) {
		Pnt *point = points[lvl] + i;
		if (point->isGood) {
			nid++;
			sid += point->iR;
		}
	}
	float fac = nid / sid;

	for (int i = 0; i < npts; i++) {
		Pnt *point = points[lvl] + i;

		if (point->isGood) {
			imgLeft.setPixel9(point->u + 0.5f, point->v + 0.5f, makeRainbow3B(point->iR * fac));

			Vec3f pt = RrKi * Vec3f(point->u, point->v, 1) + tr * point->iR;
			float u = pt[0] / pt[2];
			float v = pt[1] / pt[2];
			float Ku = fxr[lvl] * u + cxr[lvl];
			float Kv = fyr[lvl] * v + cyr[lvl];
			float idr = point->iR / pt[2];
			if (Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && idr > 0) {
				imgRight.setPixel9(Ku, Kv, makeRainbow3B(idr * fac));
			}
		} else {
			imgLeft.setPixel9(point->u + 0.5f, point->v + 0.5f, Vec3b(0, 0, 0));
		}
	}

	if (lvl == 0) {
		for (IOWrap::Output3DWrapper *ow : wraps)
			ow->pushDepthImage(&imgLeft, &imgRight);
	} else {
		// Make scaled up image.
		int w0 = w[0], h0 = h[0];
		MinimalImageB3 imgLeft0(w0, h0);
		MinimalImageB3 imgRight0(w0, h0);

		for (int v = 0; v < h0; v++) {
			for (int u = 0; u < w0; u++) {
				int i0 = v * w0 + u;
				int il = (v >> lvl) * wl + (u >> lvl);
				imgLeft0.at(i0) = imgLeft.at(il);
				imgRight0.at(i0) = imgRight.at(il);
			}
		}

		for (IOWrap::Output3DWrapper *ow : wraps)
			ow->pushDepthImage(&imgLeft0, &imgRight0);
	}
}

// calculates residual, Hessian and Hessian-block neede for re-substituting depth.
Vec3f CoarseInitializer::calcResidualAndGS(int lvl, Mat88f &H_out, Vec8f &b_out, Mat88f &H_out_sc, Vec8f &b_out_sc,
		const SE3 &refToNew, AffLight refToNew_aff) {
	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];
	Eigen::Vector3f *colorNew = newFrame->dIp[lvl];

	Mat33f RKi = refToNew.rotationMatrix().cast<float>() * Ki[lvl];
	Vec3f t = refToNew.translation().cast<float>();
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];

	Accumulator11 E;
	acc9.initialize();
	E.initialize();

	int npts = numPoints[lvl];
	Pnt *ptsl = points[lvl];
	for (int i = 0; i < npts; i++) {

		Pnt *point = ptsl + i;

		point->maxstep = 1e10;
		if (!point->isGood) {
			E.updateSingle((float) (point->energy[0]));
			point->energy_new = point->energy;
			point->isGood_new = false;
			continue;
		}

		VecNRf dp0;
		VecNRf dp1;
		VecNRf dp2;
		VecNRf dp3;
		VecNRf dp4;
		VecNRf dp5;
		VecNRf dp6;
		VecNRf dp7;
		VecNRf dd;
		VecNRf r;
		JbBuffer_new[i].setZero();

		// sum over all residuals.
		bool isGood = true;
		float energy = 0;
		for (int idx = 0; idx < patternNum; idx++) {
			int dx = patternP[idx][0];
			int dy = patternP[idx][1];

			Vec3f pt = RKi * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;
			float u = pt[0] / pt[2];
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl;
			float Kv = fyl * v + cyl;
			float new_idepth = point->idepth_new/pt[2];

			if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0)) {
				isGood = false;
				break;
			}

			Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
			float rlR = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl);

			if(!std::isfinite(rlR) || !std::isfinite((float)hitColor[0])) {
				isGood = false;
				break;
			}

			float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw *residual*residual*(2-hw);

			float dxdd = (t[0]-t[2]*u)/pt[2];
			float dydd = (t[1]-t[2]*v)/pt[2];

			if(hw < 1) hw = sqrtf(hw);
			float dxInterp = hw*hitColor[1]*fxl;
			float dyInterp = hw*hitColor[2]*fyl;
			dp0[idx] = new_idepth*dxInterp;						// dRes/dTransX - image gradient in x / depth
			dp1[idx] = new_idepth*dyInterp;// dRes/dTransY - image gradient in y / depth
			dp2[idx] = -new_idepth*(u*dxInterp + v*dyInterp);// dRes/dTransZ - ( x offset from image center / -depth) * image gradient in x .. + same for y
			dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp;// dRes/dRotX -
			dp4[idx] = (1+u*u)*dxInterp + u*v*dyInterp;// dRes/dRotY -
			dp5[idx] = -v*dxInterp + u*dyInterp;// dRes/dRotZ -
			dp6[idx] = - hw*r2new_aff[0] * rlR;// dPhotoAffineSomething...
			dp7[idx] = - hw*1;// dPhotoAffineSomething...
			dd[idx] = dxInterp * dxdd + dyInterp * dydd;// dRes/dInverseDepth
			r[idx] = hw*residual;// Huber weighted residual.


// Attempt at adding residual from Left to Right images.
//	Mat33f RlrKi = (leftToRight.rotationMatrix() * Ki[lvl]).cast<float>();
//	Vec3f tlr = leftToRight.translation().cast<float>();

//			Vec3f ptr = RlrKi * Vec3f(point->u+dx, point->v+dy, 1) + tlr*point->idepth_new;
//			float ur = ptr[0] / ptr[2];
//			float vr = ptr[1] / ptr[2];
//			float Kur = fxrl * ur + cxrl;
//			float Kvr = fyrl * vr + cyrl;
//			float new_idepth_r = point->idepth_new/ptr[2];
//
//			Vec3f hitColorRight;
//			hitColorRight[0] = -1.0;
//
//			if(Kur > 1 && Kvr > 1 && Kur < wl-2 && Kvr < hl-2 && new_idepth_r > 0) {
//				hitColorRight = getInterpolatedElement33(colorRight, Kur, Kvr, wl);
//				if (std::isfinite((float)hitColorRight[0])) {
//					residual += hitColorRight[0] - rlR;
//					residual /= 2;
//				}
//			}

//			if (hitColorRight[0] > -1.0) {
//				float dxddr = (tlr[0]-tlr[2]*ur)/ptr[2];
//				float dyddr = (tlr[1]-tlr[2]*vr)/ptr[2];
//				float dxrInterp = hw*hitColorRight[1]*fxrl;
//				float dyrInterp = hw*hitColorRight[2]*fyrl;
//				dp0[idx] += new_idepth_r*dxrInterp;
//				dp0[idx] /= 2;
//				dp1[idx] += new_idepth_r*dyrInterp;
//				dp1[idx] /= 2;
//				dp2[idx] += new_idepth_r*(ur*dxrInterp + vr*dyrInterp);;
//				dp2[idx] /= 2;
//				dd[idx] += dxrInterp * dxddr + dyrInterp * dyddr;
//				dd[idx] /= 2;
//			}



			float maxstep = 1.0f / Vec2f(dxdd*fxl, dydd*fyl).norm();
			if(maxstep < point->maxstep) point->maxstep = maxstep;

			// immediately compute dp*dd' and dd*dd' in JbBuffer1.
			JbBuffer_new[i][0] += dp0[idx]*dd[idx];
			JbBuffer_new[i][1] += dp1[idx]*dd[idx];
			JbBuffer_new[i][2] += dp2[idx]*dd[idx];
			JbBuffer_new[i][3] += dp3[idx]*dd[idx];
			JbBuffer_new[i][4] += dp4[idx]*dd[idx];
			JbBuffer_new[i][5] += dp5[idx]*dd[idx];
			JbBuffer_new[i][6] += dp6[idx]*dd[idx];
			JbBuffer_new[i][7] += dp7[idx]*dd[idx];
			JbBuffer_new[i][8] += r[idx]*dd[idx];
			JbBuffer_new[i][9] += dd[idx]*dd[idx];
		}

		if (!isGood || energy > point->outlierTH * 20) {
			E.updateSingle((float) (point->energy[0]));
			point->isGood_new = false;
			point->energy_new = point->energy;
			continue;
		}

		// add into energy.
		E.updateSingle(energy);
		point->isGood_new = true;
		point->energy_new[0] = energy;

		// update Hessian matrix.
		for (int i = 0; i + 3 < patternNum; i += 4)
			acc9.updateSSE(_mm_load_ps(((float*) (&dp0)) + i), _mm_load_ps(((float*) (&dp1)) + i),
					_mm_load_ps(((float*) (&dp2)) + i), _mm_load_ps(((float*) (&dp3)) + i), _mm_load_ps(((float*) (&dp4)) + i),
					_mm_load_ps(((float*) (&dp5)) + i), _mm_load_ps(((float*) (&dp6)) + i), _mm_load_ps(((float*) (&dp7)) + i),
					_mm_load_ps(((float*) (&r)) + i));

		for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
			acc9.updateSingle((float) dp0[i], (float) dp1[i], (float) dp2[i], (float) dp3[i], (float) dp4[i], (float) dp5[i],
					(float) dp6[i], (float) dp7[i], (float) r[i]);

	}

	E.finish();
	acc9.finish();

	float alphaEnergy = alphaW * (refToNew.translation().squaredNorm() * npts);

	//printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);

	// compute alpha opt.
	float alphaOpt;
	if (alphaEnergy > alphaK * npts) {
		// Snapped condition. There is enough camera translation(energy) initialise idepth estimates.
		alphaOpt = 0;
		alphaEnergy = alphaK * npts;
	} else {
		alphaOpt = alphaW;
	}

	acc9SC.initialize();
	for (int i = 0; i < npts; i++) {
		Pnt *point = ptsl + i;
		if (!point->isGood_new)
			continue;

		point->lastHessian_new = JbBuffer_new[i][9];

		if (alphaOpt != 0) {
			JbBuffer_new[i][8] += alphaOpt * (point->idepth_new - 1);
			JbBuffer_new[i][9] += alphaOpt;
		} else {
			// "snapped"
			JbBuffer_new[i][8] += couplingWeight * (point->idepth_new - point->iR);
			JbBuffer_new[i][9] += couplingWeight;
		}

		JbBuffer_new[i][9] = 1 / (1 + JbBuffer_new[i][9]);
		acc9SC.updateSingleWeighted((float) JbBuffer_new[i][0], (float) JbBuffer_new[i][1], (float) JbBuffer_new[i][2],
				(float) JbBuffer_new[i][3], (float) JbBuffer_new[i][4], (float) JbBuffer_new[i][5], (float) JbBuffer_new[i][6],
				(float) JbBuffer_new[i][7], (float) JbBuffer_new[i][8], (float) JbBuffer_new[i][9]);
	}
	acc9SC.finish();

	//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
	H_out = acc9.H.topLeftCorner<8, 8>();	// / acc9.num;
	b_out = acc9.H.topRightCorner<8, 1>();	// / acc9.num;
	H_out_sc = acc9SC.H.topLeftCorner<8, 8>();	// / acc9.num;
	b_out_sc = acc9SC.H.topRightCorner<8, 1>();	// / acc9.num;

	if (alphaOpt != 0) {
		// Not snapped..
		H_out(0, 0) += alphaOpt * npts;
		H_out(1, 1) += alphaOpt * npts;
		H_out(2, 2) += alphaOpt * npts;

		Vec3f tlog = refToNew.log().head<3>().cast<float>();
		b_out[0] += tlog[0] * alphaOpt * npts;
		b_out[1] += tlog[1] * alphaOpt * npts;
		b_out[2] += tlog[2] * alphaOpt * npts;
	}

	return Vec3f(E.A, alphaEnergy, E.num);
}

float CoarseInitializer::rescale(float factor) {
	float factori = 1.0f / factor;
	float factori2 = factori * factori;

	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
		int npts = numPoints[lvl];
		Pnt *ptsl = points[lvl];
		for (int i = 0; i < npts; i++) {
			ptsl[i].iR *= factor;
			ptsl[i].idepth_new *= factor;
			ptsl[i].lastHessian *= factori2;
		}
	}
	thisToNext.translation() *= factori;

	return thisToNext.translation().norm();
}

Vec3f CoarseInitializer::calcDepthRegularisationCouplingEnergy(int lvl) {
	if (!snapped)
		return Vec3f(0, 0, numPoints[lvl]);
	AccumulatorX<2> E;
	E.initialize();
	int npts = numPoints[lvl];
	for (int i = 0; i < npts; i++) {
		Pnt *point = points[lvl] + i;
		if (!point->isGood_new)
			continue;
		float rOld = (point->idepth - point->iR);
		float rNew = (point->idepth_new - point->iR);
		E.updateNoWeight(Vec2f(rOld * rOld, rNew * rNew));

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}

	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	return Vec3f(couplingWeight * E.A[0], couplingWeight * E.A[1], E.num);
}

/**
 * Sets point->iR on all good points to the mid iR(idepth) of nearest neighbours..
 */
void CoarseInitializer::updateIdepthRegularisation(int lvl) {
	int npts = numPoints[lvl];
	Pnt *ptsl = points[lvl];
	if (!snapped) {
		for (int i = 0; i < npts; i++)
			ptsl[i].iR = 1;
		return;
	}

	for (int i = 0; i < npts; i++) {
		Pnt *point = ptsl + i;
		if (!point->isGood)
			continue;

		float idnn[10];
		int nnn = 0;
		for (int j = 0; j < 10; j++) {
			if (point->neighbours[j] == -1)
				continue;
			Pnt *other = ptsl + point->neighbours[j];
			if (!other->isGood)
				continue;
			idnn[nnn] = other->iR; // !? The iRs are being updated.. idepth maybe?
			nnn++;
		}

		if (nnn > 2) {
			std::nth_element(idnn, idnn + nnn / 2, idnn + nnn);
			point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2];
		}
	}

}

void CoarseInitializer::propagateUp(int srcLvl) {
	assert(srcLvl + 1 < pyrLevelsUsed);
	// set idepth of target

	int nptss = numPoints[srcLvl];
	int nptst = numPoints[srcLvl + 1];
	Pnt *ptss = points[srcLvl];
	Pnt *ptst = points[srcLvl + 1];

	// set to zero.
	for (int i = 0; i < nptst; i++) {
		Pnt *parent = ptst + i;
		parent->iR = 0;
		parent->iRSumNum = 0;
	}

	for (int i = 0; i < nptss; i++) {
		Pnt *point = ptss + i;
		if (!point->isGood)
			continue;

		Pnt *parent = ptst + point->parent;
		parent->iR += point->iR * point->lastHessian;
		parent->iRSumNum += point->lastHessian;
	}

	for (int i = 0; i < nptst; i++) {
		Pnt *parent = ptst + i;
		if (parent->iRSumNum > 0) {
			parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
			parent->isGood = true;
		}
	}

	updateIdepthRegularisation(srcLvl + 1);
}

void CoarseInitializer::propagateDown(int srcLvl) {
	assert(srcLvl > 0);
	// set idepth of target

	int nptst = numPoints[srcLvl - 1];
	Pnt *ptss = points[srcLvl];
	Pnt *ptst = points[srcLvl - 1];

	for (int i = 0; i < nptst; i++) {
		Pnt *point = ptst + i;
		Pnt *parent = ptss + point->parent;

		if (!parent->isGood || parent->lastHessian < 0.1)
			continue;
		if (!point->isGood) {
			point->iR = point->idepth = point->idepth_new = parent->iR;
			point->isGood = true;
			point->lastHessian = 0;
		} else {
			float newiR = (point->iR * point->lastHessian * 2 + parent->iR * parent->lastHessian)
					/ (point->lastHessian * 2 + parent->lastHessian);
			point->iR = point->idepth = point->idepth_new = newiR;
		}
	}
	updateIdepthRegularisation(srcLvl - 1);
}

void CoarseInitializer::makeGradients(Eigen::Vector3f **data) {
	for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
		int lvlm1 = lvl - 1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		Eigen::Vector3f *dINew_l = data[lvl];
		Eigen::Vector3f *dINew_lm = data[lvlm1];

		for (int y = 0; y < hl; y++)
			for (int x = 0; x < wl; x++)
				dINew_l[x + y * wl][0] = 0.25f
						* (dINew_lm[2 * x + 2 * y * wlm1][0] + dINew_lm[2 * x + 1 + 2 * y * wlm1][0]
								+ dINew_lm[2 * x + 2 * y * wlm1 + wlm1][0] + dINew_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);

		for (int idx = wl; idx < wl * (hl - 1); idx++) {
			dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]);
			dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]);
		}
	}
}
void CoarseInitializer::setFirst(CalibHessian *HCalib, FrameHessian *newFrameHessian,
		std::vector<IOWrap::Output3DWrapper*> &wraps) {

	makeK(HCalib);
	firstFrame = newFrameHessian;

	PixelSelector sel(w[0], h[0]);

	char *statusMap = new char[w[0] * h[0]];
	bool *statusMapB = new bool[w[0] * h[0]];

	float densities[] = { 0.03, 0.05, 0.15, 0.5, 1 };
	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
		sel.currentPotential = 3;
		int npts;
		if (lvl == 0)
			npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0], 1, false, 2);
		else
			npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl] * w[0] * h[0]);

		if (points[lvl] != 0)
			delete[] points[lvl];
		points[lvl] = new Pnt[npts];

		// set idepth map to initially 1 everywhere.
		int wl = w[lvl], hl = h[lvl];
		Pnt *pl = points[lvl];
		int nl = 0;
		for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
			for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++) {
				//if(x==2) printf("y=%d!\n",y);
				if ((lvl != 0 && statusMapB[x + y * wl]) || (lvl == 0 && statusMap[x + y * wl] != 0)) {
					//assert(patternNum==9);
					pl[nl].u = x + 0.1;
					pl[nl].v = y + 0.1;
					pl[nl].idepth = 1;
					pl[nl].iR = 1;
					pl[nl].isGood = true;
					pl[nl].energy.setZero();
					pl[nl].lastHessian = 0;
					pl[nl].lastHessian_new = 0;
					pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];

//				Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
//				float sumGrad2=0;
//				for(int idx=0;idx<patternNum;idx++)
//				{
//					int dx = patternP[idx][0];
//					int dy = patternP[idx][1];
//					float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
//					sumGrad2 += absgrad;
//				}

//				float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
//				pl[nl].outlierTH = patternNum*gth*gth;
//

					pl[nl].outlierTH = patternNum * setting_outlierTH;

					nl++;
					assert(nl <= npts);
				}
			}

		numPoints[lvl] = nl;
	}
	delete[] statusMap;
	delete[] statusMapB;

	makeNN();

	thisToNext = SE3();
	snapped = false;
	frameID = snappedAt = 0;

	for (IOWrap::Output3DWrapper *ow : wraps)
		ow->pushLiveFrame(newFrameHessian);
}

void CoarseInitializer::resetPoints(int lvl) {
	Pnt *pts = points[lvl];
	int npts = numPoints[lvl];
	for (int i = 0; i < npts; i++) {
		pts[i].energy.setZero();
		pts[i].idepth_new = pts[i].idepth;

		if (lvl == pyrLevelsUsed - 1 && !pts[i].isGood) {
			float snd = 0, sn = 0;
			for (int n = 0; n < 10; n++) {
				if (pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood)
					continue;
				snd += pts[pts[i].neighbours[n]].iR;
				sn += 1;
			}

			if (sn > 0) {
				pts[i].isGood = true;
				pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd / sn;
			}
		}
	}
}

void CoarseInitializer::doIdepthStepUpdate(int lvl, float lambda, Vec8f inc) {
	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	Pnt *pts = points[lvl];
	int npts = numPoints[lvl];
	for (int i = 0; i < npts; i++) {
		if (!pts[i].isGood)
			continue;

		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		float step = -b * JbBuffer[i][9] / (1 + lambda);

		float maxstep = maxPixelStep * pts[i].maxstep;
		if (maxstep > idMaxStep)
			maxstep = idMaxStep;

		if (step > maxstep)
			step = maxstep;
		if (step < -maxstep)
			step = -maxstep;

		float newIdepth = pts[i].idepth + step;
		if (newIdepth < 1e-3)
			newIdepth = 1e-3;
		if (newIdepth > 50)
			newIdepth = 50;
		pts[i].idepth_new = newIdepth;
	}

}
void CoarseInitializer::applyStep(int lvl) {
	Pnt *pts = points[lvl];
	int npts = numPoints[lvl];
	for (int i = 0; i < npts; i++) {
		if (!pts[i].isGood) {
			pts[i].idepth = pts[i].idepth_new = pts[i].iR;
			continue;
		}
		pts[i].energy = pts[i].energy_new;
		pts[i].isGood = pts[i].isGood_new;
		pts[i].idepth = pts[i].idepth_new;
		pts[i].lastHessian = pts[i].lastHessian_new;
	}
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

void CoarseInitializer::makeK(CalibHessian *HCalib) {
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	fxr[0] = HCalib->fxlR();
	fyr[0] = HCalib->fylR();
	cxr[0] = HCalib->cxlR();
	cyr[0] = HCalib->cylR();

	for (int level = 1; level < pyrLevelsUsed; ++level) {
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level - 1] * 0.5;
		fy[level] = fy[level - 1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int) 1 << level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int) 1 << level) - 0.5;

		fxr[level] = fxr[level - 1] * 0.5;
		fyr[level] = fyr[level - 1] * 0.5;
		cxr[level] = (cxr[0] + 0.5) / ((int) 1 << level) - 0.5;
		cyr[level] = (cyr[0] + 0.5) / ((int) 1 << level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++level) {
		K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Kr[level] << fxr[level], 0.0, cxr[level], 0.0, fyr[level], cyr[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		Kri[level] = Kr[level].inverse();
		fxi[level] = Ki[level](0, 0);
		fyi[level] = Ki[level](1, 1);
		cxi[level] = Ki[level](0, 2);
		cyi[level] = Ki[level](1, 2);
	}

	leftToRight = HCalib->getLeftToRight();
}

void CoarseInitializer::makeNN() {
	const float NNDistFactor = 0.05;

	typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud>, FLANNPointcloud, 2> KDTree;

	// build indices
	FLANNPointcloud pcs[PYR_LEVELS];
	KDTree *indexes[PYR_LEVELS];
	for (int i = 0; i < pyrLevelsUsed; i++) {
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5));
		indexes[i]->buildIndex();
	}

	const int nn = 10;

	// find NN & parents
	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
		Pnt *pts = points[lvl];
		int npts = numPoints[lvl];

		int ret_index[nn];
		float ret_dist[nn];
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for (int i = 0; i < npts; i++) {
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			Vec2f pt = Vec2f(pts[i].u, pts[i].v);
			indexes[lvl]->findNeighbors(resultSet, (float*) &pt, nanoflann::SearchParams());
			int myidx = 0;
			float sumDF = 0;
			for (int k = 0; k < nn; k++) {
				pts[i].neighbours[myidx] = ret_index[k];
				float df = expf(-ret_dist[k] * NNDistFactor);
				sumDF += df;
				pts[i].neighboursDist[myidx] = df;
				assert(ret_index[k] >= 0 && ret_index[k] < npts);
				myidx++;
			}
			for (int k = 0; k < nn; k++)
				pts[i].neighboursDist[k] *= 10 / sumDF;

			if (lvl < pyrLevelsUsed - 1) {
				resultSet1.init(ret_index, ret_dist);
				pt = pt * 0.5f - Vec2f(0.25f, 0.25f);
				indexes[lvl + 1]->findNeighbors(resultSet1, (float*) &pt, nanoflann::SearchParams());

				pts[i].parent = ret_index[0];
				pts[i].parentDist = expf(-ret_dist[0] * NNDistFactor);

				assert(ret_index[0] >= 0 && ret_index[0] < numPoints[lvl + 1]);
			} else {
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}

	// done.

	for (int i = 0; i < pyrLevelsUsed; i++)
		delete indexes[i];
}

float CoarseInitializer::computeRescale() {
	const Mat33 R = leftToRight.rotationMatrix();
	const Vec3f t = leftToRight.translation().cast<float>();

	float rescale = 1.0;

	for (int lvl = 4; lvl >= 0; lvl--) {
		const Mat33f RKi = R.cast<float>() * Ki[lvl];
		const float fxrl = fxr[lvl];
		const float fyrl = fyr[lvl];
		const float cxrl = cxr[lvl];
		const float cyrl = cyr[lvl];
		const int wl = w[lvl];
		const int hl = h[lvl];

		Vec3f *leftImage = firstFrame->dIp[lvl];
		Vec3f *rightImage = firstFrame->dIrp[lvl];

		for (int it = 0; it < 25; it++) {

			float lvlAbsRes = 0.0;
			float lvlEnergy = 0.0;
			float H = 1.0;
			float b = 0.0;
			int nGood = 0;

			for (int pntIdx = 0; pntIdx < numPoints[lvl]; pntIdx++) {
				Pnt *pnt = points[lvl] + pntIdx;

//				if (pnt->iR > 1) { // only points that are further away.
//					continue;
//				}

				float absRes = 0.0;
				float pntEnergy = 0.0;
				bool isGood = true;
				VecNRf dRdS;
				VecNRf r;
				float Hp = 0.0;
				float bp = 0.0;

				for (int patIdx = 0; patIdx < patternNum; patIdx++) {
					int dx = patternP[patIdx][0];
					int dy = patternP[patIdx][1];

					Vec3f pt = RKi * Vec3f(pnt->u+dx, pnt->v+dy, 1) + t * (pnt->iR * rescale);
					float u = pt[0] / pt[2];
					float v = pt[1] / pt[2];
					float rightU = fxrl * u + cxrl;
					float rightV = fyrl * v + cyrl;
					float rightIR = (rescale * pnt->iR)/pt[2];

					if(!(rightU > 1 && rightV > 1 && rightU < wl-2 && rightV < hl-2 && rightIR > 0)) {
						isGood = false;
						break;
					}

					Vec3f rightColor = getInterpolatedElement33(rightImage, rightU, rightV, wl);
					if (!isfinite(rightColor[0])) {
						isGood = false;
						break;
					}

					float leftColor = getInterpolatedElement31(leftImage, pnt->u+dx, pnt->v+dy, wl);
					if (!isfinite(leftColor)) {
						isGood = false;
						break;
					}

					float residual = leftColor - rightColor[0];
					absRes += fabs(residual);
					float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
					pntEnergy += hw *residual*residual*(2-hw);

					float dxdd = (t[0]-t[2]*u)/pt[2];
					float dydd = (t[1]-t[2]*v)/pt[2];

					if(hw < 1) hw = sqrtf(hw);
					float dxInterp = rightColor[1]*fxrl;
					float dyInterp = rightColor[2]*fyrl;
					dRdS[patIdx] = (dxInterp * dxdd + dyInterp * dydd) * -rescale; // dRes/dRescale
					r[patIdx] = residual;// Huber weighted residual.

					Hp += dRdS[patIdx]*dRdS[patIdx];
					bp += r[patIdx]*dRdS[patIdx];
				}

				if (!isGood || pntEnergy > pnt->outlierTH * 20) {
					//			E.updateSingle((float) (point->energy[0]));
					//			point->isGood_new = false;
					//			point->energy_new = point->energy;
					continue;
				}

				lvlAbsRes += absRes;
				lvlEnergy += pntEnergy;
				H += Hp;
				b += bp;
				nGood++;
			}

			float rescaleDelta = -b / H;

			if (true) {
				printf("lvl %i\tit %i\trescale %f\t rescaleDelta %f\tlvlAbsRes %f\tnGood %d\tlvlAbsResPP %f\n", lvl, it, rescale,
						rescaleDelta, lvlAbsRes, nGood, lvlAbsRes / nGood);
			}

			rescale += rescaleDelta;
		}
	}

	return rescale;
}

//void epeLine() {
//	//	// skew-symmetric(-t)
//	//	const Mat33 tSkew = (Mat33() << 0.0, t[2], -t[1], -t[2], 0.0, t[0], t[1], -t[0], 0.0).finished();
//	// std::cout << "Ri:\n" << Ri << "\ntSkew:\n" << tSkew << "\n";
//
//
//	// Fundamental.transpose()...
//	const Mat33 Ft = (Ki[lvl].transpose() * tSkew * Ri * Kri[lvl]).transpose();
//
//	const Mat33 KrRKi = Kr[lvl] * R * Ki[lvl];
//	const Mat22f Rplane = KrRKi.topLeftCorner(2, 2).cast<float>();
//	Vec2f rotatetPattern[MAX_RES_PER_POINT];
//	for (int idx = 0; idx < patternNum; idx++)
//		rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);
//
//	float point_buf[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
//
//	float line_buf[w[lvl]];
//	for (int i = 0; i < w[lvl]; i++)
//		line_buf[i] = 0;
//
//	float ssdErrors[w[lvl] - 4];
//	for (int i = 0; i < w[lvl] - 4; i++)
//		line_buf[i] = 0;
//
//	float errSum = 0;
//	int errCnt = 0;
//	for (int pIdx = 0; pIdx < numPoints[lvl]; pIdx++) {
//		Pnt *pntLeft = points[lvl] + pIdx;
//		const float u_left = pntLeft->u;
//		const float v_left = pntLeft->v;
//
//		const Vec3 pointLeft = Vec3(u_left, v_left, 1);
//		const Vec3 lineRight = Ft * pointLeft;
//
//		// Left point at infinite depth projected to right.
//		Vec3 pointRightFar = R * Ki[lvl] * pointLeft;
//		assert(pointRightFar[2] > 0);
//		pointRightFar /= pointRightFar[2];
//		pointRightFar = Kr[lvl] * pointRightFar;
//
//		// Calculate bi-linear pixel values for patch around left point.
//		for (int u = 0; u < 5; u++) {
//			const Vec3f ival = getInterpolatedElement33(firstFrame->dIp[lvl], u_left + u - 2, v_left, w[lvl]);
//			point_buf[u] = ival(0);
//		}
//
//		// Calculate linear in v pixel values along epipolar line for integer u pixel locations.
//		const int epFarLimitIdx = std::min((int) (pointRightFar(0) + 0.5) + 2, w[lvl]);
//		const int epNearLimitIdx = std::max(0, epFarLimitIdx - (24 << (4 - lvl)));
//		for (int u = epNearLimitIdx; u < epFarLimitIdx; u++) {
//			float y = (lineRight[0] * u + lineRight[2]) / -lineRight[1] + 0.5;
//			if (y <= 1.0 || y >= h[lvl]) {
//				line_buf[u] = std::numeric_limits<int>::min();
//			} else {
//				int v = (int) y;
//				float dv = y - v;
//				const Eigen::Vector3f *pixel_uv = firstFrame->dIrp[lvl] + u + v * w[lvl];
//				const Eigen::Vector3f *pixel_uv1 = pixel_uv - w[lvl];
//				line_buf[u] = pixel_uv->coeff(0) * dv + pixel_uv1->coeff(0) * (1 - dv);
//			}
//		}
//
//		for (int idx = 0; idx < patternNum; idx++) {
//			point_buf[idx] = getInterpolatedElement31(firstFrame->dIp[lvl], u_left + patternP[idx][0], v_left+patternP[idx][1], w[lvl]);
//		}
//
//		float bestErr1 = std::numeric_limits<float>::max(), bestErr2 = std::numeric_limits<float>::max();
//		int bestIdx1 = std::numeric_limits<int>::min(), bestIdx2 = std::numeric_limits<int>::min();
//
//		// Sum of Squared errors over 5 pixels.
//		for (int u = epNearLimitIdx; u < epFarLimitIdx - 4; u++) {
//			ssdErrors[u] = 0;
////			for (int i = 0; i < 5; i++) {
////				float e = point_buf[i] - line_buf[u + i];
////				ssdErrors[u] += e * e;
////			}
//
//			float v = (lineRight[0] * u + lineRight[2]) / -lineRight[1];
//
//			for (int idx = 0; idx < patternNum; idx++) {
//				float hitColor = getInterpolatedElement31(firstFrame->dIrp[lvl], (float)(u + rotatetPattern[idx][0]), v + rotatetPattern[idx][1], w[lvl]);
//
//				if (std::isfinite(hitColor)) {
//					float residual = hitColor - point_buf[idx];
//					float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
//					ssdErrors[u] += hw * residual * residual * (2 - hw);
//				} else {
//					ssdErrors[u] += 1e5;
//				}
//			}
//
//			if (ssdErrors[u] < bestErr1) {
//				bestErr1 = ssdErrors[u];
//				bestIdx1 = u; // offset from u in right image by 2
//			}
//		}
//
//		// Find second best not near best.
//		for (int u = epNearLimitIdx; u < epFarLimitIdx - 4; u++) {
//			// Only update 2nd best if new 1st best is not within one pixel.
//			if ((u < (bestIdx1 - 1) || u > (bestIdx1 + 1)) && ssdErrors[u] < bestErr2) {
//				bestErr2 = ssdErrors[u];
//				bestIdx2 = u;
//			}
//		}
//
////		if (pIdx % 20 == 0) {
////			std::cout << "pntLeft(x,y):" << u_left << "," << v_left << "\t";
////			std::cout << "Best:" << bestErr1 << "@" << bestIdx1 << "," << bestErr2 << "@" << bestIdx2 << "\t";
////			std::cout << "1st/2nd:" << bestErr1 / bestErr2 << "\n";
////		}
//
//		if (bestErr1 < 500 && bestErr1 / bestErr2 < 0.8) {
//			pntLeft->idepth = (u_left - bestIdx1) / (t.norm());
//			pntLeft->idepth_new = pntLeft->idepth;
//			pntLeft->iR = pntLeft->idepth;
//			std::cout << "pntLeft(x,y):(" << u_left << "," << v_left << ") " << pntLeft->idepth << "\t";
//			std::cout << "Best:" << bestIdx1 << "(" << bestErr1 << "), " << bestIdx2 << "(" << bestErr2 << ")\t";
//			std::cout << "1st/2nd:" << bestErr1 / bestErr2 << "\n";
//		} else {
//			std::cout << "pntLeft(x,y):(" << u_left << "," << v_left << ") " << pntLeft->idepth << "\n";
//			pntLeft->isGood = false;
//		}
//
//		// Translate point from left to right camera plane. Right point not scaled to z = 1.
//		Vec3 pointRight = R * Ki[lvl] * pointLeft + t;
//		assert(pointRight[2] > 0);
//		// Scale x,y to camera plane. z = 1.
//		pointRight /= pointRight[2];
//
//		// Camera plane to image plane.
//		pointRight = K[lvl] * pointRight;
//
//		// std::cout << "pointLeft:\n" << pointLeft << "\nlineRight:\n" << lineRight << "\npointRight:\n" << pointRight << "\n";
//
//		// std::cout << "pointRightFar:\n" << pointRightFar << "\n";
//
//		errSum += lineRight.transpose() * pointRightFar;
//		errCnt++;
//
////		if (pIdx % 132 == 0) {
////			std::cout << pntLeft->u << "," << pntLeft->v << "\n";
////		}
//	}
//	std::cout << "Ft:\n" << Ft << "\nerrSum:" << errSum << "\nerrCnt:" << errCnt << "\nAvgErr:" << (errSum / errCnt) << "\n";
//	return 1.0;
//}
}

