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

CoarseInitializer::CoarseInitializer(CalibHessian *HCalib, int ww, int hh) :
		thisToNext_aff(0, 0), thisToNext(SE3()) {
	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
		points[lvl] = 0;
		numPoints[lvl] = 0;
	}

	JbBuffer = new Vec10f[ww * hh];
	JbBuffer_new = new Vec10f[ww * hh];

	frameID = -1;
	fixAffine = true;
	printDebug = true;

	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;

	makeK(HCalib);

	lrR = leftToRight.rotationMatrix().cast<float>();
	lrt = leftToRight.translation().cast<float>();

	// skew-symmetric(-t)
	const Mat33f tSkew = (Mat33f() << 0.0, lrt[2], -lrt[1], -lrt[2], 0.0, lrt[0], lrt[1], -lrt[0], 0.0).finished();
	std::cout << "R:\n" << lrR << "\ntSkew:\n" << tSkew << "\n";

	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
		// Fundamental.transpose()...
		Ftl[lvl] = (Ki[lvl].transpose() * tSkew * lrR.transpose() * Kri[lvl]).transpose();
		KrRKil[lvl] = Kr[lvl] * lrR * Ki[lvl];

		Mat22f Rplane = KrRKil[lvl].topLeftCorner(2, 2);
		for (int idx = 0; idx < patternNum; idx++) {
			rotatedPattern[lvl][idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);
		}
	}
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

	if (!snapped) {
		thisToNext.translation().setZero();
		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
			int npts = numPoints[lvl];
			Pnt *ptsl = points[lvl];
			for (int i = 0; i < npts; i++) {
				if (ptsl[i].idepthLr >= 0) {
					ptsl[i].iR = ptsl[i].idepthLr;
					ptsl[i].idepth_new = ptsl[i].idepthLr;
				} else {
					ptsl[i].iR = 1;
					ptsl[i].idepth_new = 1;
					ptsl[i].isGood = true;
				}
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
				if (resNew[1] == alphaK * numPoints[lvl]) {
					if (printDebug && snapped == false)
						printf("Snapped!\n");
					snapped = true;
				}
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
	else if (snappedAt == 0)
		snappedAt = frameID;

	debugPlot(wraps);

	float lrTransNorm = leftToRight.translation().norm();
	float ttnTransNorm = thisToNext.translation().norm();
	return snapped && (ttnTransNorm > lrTransNorm && frameID > snappedAt + 5);
}

void CoarseInitializer::debugPlot(std::vector<IOWrap::Output3DWrapper*> &wraps, int lvl) {
	bool needCall = false;
	for (IOWrap::Output3DWrapper *ow : wraps)
		needCall = needCall || ow->needPushDepthImage();
	if (!needCall)
		return;

	int w0 = w[0], h0 = h[0];
	int wl = w[lvl], hl = h[lvl];

	Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];
	Eigen::Vector3f *colorRight = firstFrame->dIrp[lvl];

	MinimalImageB3 imgLeft(w0, h0);
	MinimalImageB3 imgRight(w0, h0);

	for (int v = 0; v < h0; v++) {
		for (int u = 0; u < w0; u++) {
			int i0 = v * w0 + u;
			// Make scaled image.
			int il = (v >> lvl) * wl + (u >> lvl);

			float c = colorRef[il][0];
			if (c >= 256.0) {
				c = 255.0;
			}
			imgLeft.at(i0) = Vec3b(c, c, c);

			c = colorRight[il][0];
			if (c >= 256.0) {
				c = 255.0;
			}
			imgRight.at(i0) = Vec3b(c, c, c);
		}
	}

	Mat33f RrKi = (leftToRight.rotationMatrix().cast<float>() * Ki[lvl]);
	Vec3f tr = (leftToRight.translation()).cast<float>();

	int npts = numPoints[lvl];

	float nid = 0, sid = 0;
	for (int i = 0; i < npts; i++) {
		Pnt *point = points[lvl] + i;
		if (point->isGood) {
			nid++;
			sid += point->iR;
		}
	}
	float fac = 1; //nid / sid;

	int lvlScale = 1 << lvl;

	for (int i = 0; i < npts; i++) {
		Pnt *point = points[lvl] + i;

		Vec3b idepthColor;
		if (point->isGood) {
			idepthColor = makeRainbow3B(point->iR * fac);

			Vec3f pt = RrKi * Vec3f(point->u, point->v, 1) + tr * point->iR;
			float u = pt[0] / pt[2];
			float v = pt[1] / pt[2];
			float Ku = fxr[lvl] * u + cxr[lvl];
			float Kv = fyr[lvl] * v + cyr[lvl];
			if (Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && point->iR > 0) {
				imgRight.setPixelSizedForLevel(lvl, lvlScale * Ku + 0.5f, lvlScale * Kv + 0.5f, idepthColor);
			}
		} else {
			idepthColor = Vec3b(0, 0, 0);
		}
		imgLeft.setPixelSizedForLevel(lvl, lvlScale * point->u + 0.5f, lvlScale * point->v + 0.5f, idepthColor);
	}

	for (IOWrap::Output3DWrapper *ow : wraps)
		ow->pushDepthImage(&imgLeft, &imgRight);
}

// calculates residual, Hessian and Hessian-block needed for re-substituting depth.
Vec3f CoarseInitializer::calcResidualAndGS(int lvl, Mat88f &H_out, Vec8f &b_out, Mat88f &H_out_sc, Vec8f &b_out_sc,
		const SE3 &refToNew, AffLight refToNew_aff) {
	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];
	Eigen::Vector3f *colorFirstRight = firstFrame->dIrp[lvl];
	Eigen::Vector3f *colorNew = newFrame->dIp[lvl];

	Mat33f RKi = refToNew.rotationMatrix().cast<float>() * Ki[lvl];
	Vec3f t = refToNew.translation().cast<float>();
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

	Mat33f RlrKi = leftToRight.rotationMatrix().cast<float>() * Ki[lvl];
	Vec3f tlr = leftToRight.translation().cast<float>();

	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];

	float fxrl = fxr[lvl];
	float fyrl = fyr[lvl];
	float cxrl = cxr[lvl];
	float cyrl = cyr[lvl];

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

		VecNRf dTx;
		VecNRf dTy;
		VecNRf dTz;
		VecNRf dRx;
		VecNRf dRy;
		VecNRf dRz;
		VecNRf dA;
		VecNRf dB;
		VecNRf dId;
		VecNRf r;
		JbBuffer_new[i].setZero();

		// sum over all residuals.
		bool isGood = true;
		float energy = 0;
		for (int idx = 0; idx < patternNum; idx++) {
			int dx = patternP[idx][0];
			int dy = patternP[idx][1];
			Vec3f ptRef(point->u+dx, point->v+dy, 1);

			Vec3f pt = RKi * ptRef + t*point->idepth_new;
			float u = pt[0] / pt[2];
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl;
			float Kv = fyl * v + cyl;
			float new_idepth = point->idepth_new/pt[2];

			if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0)) {
				isGood = false;
				break;
			}

			float refColor = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl);
			Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);

			// Attempt at adding residual from Left to Right images.
			Vec3f ptr = RlrKi * ptRef + tlr*point->idepth_new;
			float ur = ptr[0] / ptr[2];
			float vr = ptr[1] / ptr[2];
			float Kur = fxrl * ur + cxrl;
			float Kvr = fyrl * vr + cyrl;
			float new_idepth_r = point->idepth_new/ptr[2];

			if(!(Kur > 1 && Kvr > 1 && Kur < wl-2 && Kvr < hl-2 && new_idepth_r > 0)) {
				isGood = false;
				break;
			}

			Vec3f hitColorRight = getInterpolatedElement33(colorFirstRight, Kur, Kvr, wl);

			if(!std::isfinite(refColor) || !std::isfinite(hitColor[0]) || !std::isfinite(hitColorRight[0])) {
				isGood = false;
				break;
			}

			float residual = hitColor[0] - r2new_aff[0] * refColor - r2new_aff[1];
			residual += hitColorRight[0] - refColor; // No photometric correction between left/right images.

			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw *residual*residual*(2-hw);

			float dxdd = (t[0]-t[2]*u)/pt[2];
			float dydd = (t[1]-t[2]*v)/pt[2];

			if(hw < 1) hw = sqrtf(hw);
			float dxInterp = hw*hitColor[1]*fxl;
			float dyInterp = hw*hitColor[2]*fyl;
			dTx[idx] = new_idepth*dxInterp;// dRes/dTransX - image gradient in x / depth
			dTy[idx] = new_idepth*dyInterp;// dRes/dTransY - image gradient in y / depth
			dTz[idx] = -new_idepth*(u*dxInterp + v*dyInterp);// dRes/dTransZ - ( x offset from image center / -depth) * image gradient in x .. + same for y
			dRx[idx] = -u*v*dxInterp - (1+v*v)*dyInterp;// dRes/dRotX -
			dRy[idx] = (1+u*u)*dxInterp + u*v*dyInterp;// dRes/dRotY -
			dRz[idx] = -v*dxInterp + u*dyInterp;// dRes/dRotZ -
			dA[idx] = - hw*r2new_aff[0] * refColor;// dPhotoAffineSomething...
			dB[idx] = - hw*1;// dPhotoAffineSomething...
			dId[idx] = dxInterp * dxdd + dyInterp * dydd;// dRes/dInverseDepth

			// Left/right residual only depends on depth.
			float dxddr = (tlr[0]-tlr[2]*ur)/ptr[2];
			float dyddr = (tlr[1]-tlr[2]*vr)/ptr[2];
			dId[idx] += hw*hitColorRight[1] * fxrl * dxddr + hw*hitColorRight[2] * fyrl * dyddr;

			r[idx] = hw*residual;// Huber weighted residual.

			float maxstep = 1.0f / Vec2f(dxdd*fxl, dydd*fyl).norm();
			if(maxstep < point->maxstep) point->maxstep = maxstep;

			// immediately compute dp*dd' and dd*dd' in JbBuffer1.
			JbBuffer_new[i][0] += dTx[idx]*dId[idx];
			JbBuffer_new[i][1] += dTy[idx]*dId[idx];
			JbBuffer_new[i][2] += dTz[idx]*dId[idx];
			JbBuffer_new[i][3] += dRx[idx]*dId[idx];
			JbBuffer_new[i][4] += dRy[idx]*dId[idx];
			JbBuffer_new[i][5] += dRz[idx]*dId[idx];
			JbBuffer_new[i][6] += dA[idx]*dId[idx];
			JbBuffer_new[i][7] += dB[idx]*dId[idx];
			JbBuffer_new[i][8] += r[idx]*dId[idx];
			JbBuffer_new[i][9] += dId[idx]*dId[idx];
		}

		if (!isGood || energy > (patternNum * setting_outlierTH) * 10) {
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
			acc9.updateSSE(_mm_load_ps(((float*) (&dTx)) + i), _mm_load_ps(((float*) (&dTy)) + i),
					_mm_load_ps(((float*) (&dTz)) + i), _mm_load_ps(((float*) (&dRx)) + i), _mm_load_ps(((float*) (&dRy)) + i),
					_mm_load_ps(((float*) (&dRz)) + i), _mm_load_ps(((float*) (&dA)) + i), _mm_load_ps(((float*) (&dB)) + i),
					_mm_load_ps(((float*) (&r)) + i));

		for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
			acc9.updateSingle((float) dTx[i], (float) dTy[i], (float) dTz[i], (float) dRx[i], (float) dRy[i], (float) dRz[i],
					(float) dA[i], (float) dB[i], (float) r[i]);

	}

	E.finish();
	acc9.finish();

	assert(E.num == npts);

	float alphaEnergy = alphaW * (refToNew.translation().squaredNorm() * npts);

	//printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);

	// Snapped condition. There is enough camera translation(energy) to initialise idepth estimates.
	bool snapped = alphaEnergy > alphaK * npts;
	if (snapped) {
		alphaEnergy = alphaK * npts;
	}

	acc9SC.initialize();
	for (int i = 0; i < npts; i++) {
		Pnt *point = ptsl + i;
		if (!point->isGood_new)
			continue;

		point->lastHessian_new = JbBuffer_new[i][9];

		if (snapped) {
			JbBuffer_new[i][8] += (point->idepth_new - point->iR);
			JbBuffer_new[i][9] += 1;
		} else {
			JbBuffer_new[i][8] += alphaW * (point->idepth_new - 1);
			JbBuffer_new[i][9] += alphaW;
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

	if (!snapped) {
		H_out(0, 0) += alphaW * npts;
		H_out(1, 1) += alphaW * npts;
		H_out(2, 2) += alphaW * npts;

		Vec3f tlog = refToNew.log().head<3>().cast<float>();
		b_out[0] += tlog[0] * alphaW * npts;
		b_out[1] += tlog[1] * alphaW * npts;
		b_out[2] += tlog[2] * alphaW * npts;
	}

	return Vec3f(E.A, alphaEnergy, E.num);  // Accumulation of point residual error, translation * npoints, npoints.
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
	return Vec3f(E.A[0], E.A[1], E.num);
}

/**
 * Sets point->iR on all good points to the mid iR(idepth) of nearest neighbours..
 */
void CoarseInitializer::updateIdepthRegularisation(int lvl) {
	const float regWeight = 0.7; //*freeDebugParam4;

	int npts = numPoints[lvl];
	Pnt *ptsl = points[lvl];
	if (!snapped) {
		for (int i = 0; i < npts; i++) {
			Pnt *point = ptsl + i;
			if (point->idepthLr >= 0) {
				point->iR = point->idepthLr;
			} else {
				// ptsl[i].iR = 1;
				float idnn[10];
				int nnn = 0;
				for (int j = 0; j < 10; j++) {
					if (point->neighbours[j] == -1)
						continue;
					Pnt *other = ptsl + point->neighbours[j];
					if (!other->isGood || other->idepthLr < 0) // (!other->isGood)
						continue;
					idnn[nnn] = other->idepthLr;
					nnn++;
				}

				if (nnn > 2) {
					std::nth_element(idnn, idnn + nnn / 2, idnn + nnn);
					point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2];
					assert(point->iR >= 0);
				}
			}
		}
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
			assert(point->iR >= 0);
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
			assert(parent->iR >= 0);
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
			assert(point->iR >= 0);
		} else {
			float newiR = (point->iR * point->lastHessian * 2 + parent->iR * parent->lastHessian)
					/ (point->lastHessian * 2 + parent->lastHessian);
			point->iR = point->idepth = point->idepth_new = newiR;
			assert(point->iR >= 0);
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
void CoarseInitializer::setFirst(FrameHessian *newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps) {
	firstFrame = newFrameHessian;

	for (IOWrap::Output3DWrapper *ow : wraps)
		ow->pushLiveFrame(newFrameHessian);

	PixelSelector sel(w[0], h[0]);

	int *pointsWithIdepthLrEstimate;
	int pointsWithIdepthLrEstimateCounter = 0;

	char *statusMap = new char[w[0] * h[0]];
	bool *statusMapB = new bool[w[0] * h[0]];

	float densities[] = { 0.03, 0.05, 0.15, 0.5, 1 };
	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
		int wl = w[lvl], hl = h[lvl];

		sel.currentPotential = 3;
		int npts;
		if (lvl == 0) {
			npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * wl * hl, 1, false, 2);
			pointsWithIdepthLrEstimate = new int[npts];
		} else {
			npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, wl, hl, densities[lvl] * wl * hl);
		}

		if (points[lvl] != 0)
			delete[] points[lvl];
		points[lvl] = new Pnt[npts];

		// set idepth map to initially 1 everywhere.
		Pnt *pl = points[lvl];
		int nl = 0;
		for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++) {
			for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++) {
				//if(x==2) printf("y=%d!\n",y);
				if ((lvl != 0 && statusMapB[x + y * wl]) || (lvl == 0 && statusMap[x + y * wl] != 0)) {
					pl[nl].u = x + 0.1;
					pl[nl].v = y + 0.1;
					pl[nl].idepthLr = -1;
					pl[nl].idepth = 1;
					pl[nl].iR = 1;
					pl[nl].isGood = false;
					pl[nl].energy.setZero();
					pl[nl].lastHessian = 0;
					pl[nl].lastHessian_new = 0;
					pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];

					if (lvl == 0 && idepthLrEstimate(lvl, &pl[nl])) {
						pointsWithIdepthLrEstimate[pointsWithIdepthLrEstimateCounter++] = nl;
					}

					nl++;
					assert(nl <= npts);
				}
			}
		}

		numPoints[lvl] = nl;
	}
	delete[] statusMap;
	delete[] statusMapB;

	makeNN();

	// Only use idepthLr estimates that are similar to their neighbours.
	const float fBound = 0.90f;
	for (int estimateI = 0; estimateI < pointsWithIdepthLrEstimateCounter; estimateI++) {
		int pi = pointsWithIdepthLrEstimate[estimateI];
		Pnt *p = &points[0][pi];
		assert(p->idepthLr >= 0);

		int nnWithIDepth = 0;
		int nnWithSimilarIDepth = 0;

		// First neighbour is self.
		for (int i = 1; i < 10; i++) {
			if (p->neighbours[i] == -1)
				continue;

			Pnt *n = &points[0][p->neighbours[i]];
			if (n->idepthLr < 0)
				continue;

			nnWithIDepth++;

			float f = p->idepthLr / n->idepthLr;
			if (fBound < f && f < 1 / fBound) {
				nnWithSimilarIDepth++;
			}
		}

		if (nnWithIDepth >= 3 && nnWithSimilarIDepth / nnWithIDepth >= 0.6) {
			// LR depth estimate agrees with neighbours.
			p->isGood = true;
		}
	}

	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
		float idepthLrSum = 0;
		float idepthLrSqSum = 0;
		int idepthLrCounter = 0;
		int goodIdepthLr = 0;
		float bestParentDist[(lvl + 1) < pyrLevelsUsed ? numPoints[lvl + 1] : 0];

		for (int i = 0; i < numPoints[lvl]; i++) {
			Pnt *p = &points[lvl][i];
			if (p->idepthLr < 0)
				continue;

			idepthLrCounter++;
			if (p->isGood) {
				p->idepth = p->idepthLr;
				p->iR = p->idepthLr;

				if (p->parent >= 0) {
					int parenti = p->parent;
					Pnt *parent = &points[lvl + 1][parenti];
					if (parent->idepthLr < 0 || bestParentDist[parenti] < p->parentDist) {
						bestParentDist[parenti] = p->parentDist;
						parent->idepthLr = p->idepthLr;
						parent->isGood = true;
					}
				}

				idepthLrSum += p->idepthLr;
				idepthLrSqSum += p->idepthLr * p->idepthLr;
				goodIdepthLr++;
			}
		}

		if (printDebug) {
			float idMu = idepthLrSum / idepthLrCounter;
			float idSigma = idepthLrSqSum / idepthLrCounter - idMu * idMu;

			printf("Points at lvl %d idMu:%f\tidSigma:%f\t(All/With IDepthLr/With good IDepthLr) %d/%d/%d\n", lvl, idMu, idSigma,
					numPoints[lvl], idepthLrCounter, goodIdepthLr);
		}
	}

	thisToNext = SE3();
	snapped = false;
	frameID = snappedAt = 0;

	debugPlot(wraps);
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
				assert(pts[i].iR >= 0);
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
	FLANNPointcloud pcs[MAX_PYR_LEVELS];
	KDTree *indexes[MAX_PYR_LEVELS];
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

//		int npts1 = (lvl + 1) < pyrLevelsUsed ? numPoints[lvl + 1] : 0;
//		float parentsSumDf[npts1];
//		int parentsSumDfCounter[npts1];
//		for (int i = 0 ; i < npts1 ; i++) {
//			parentsSumDf[i] = 0;
//			parentsSumDfCounter[i] = 0;
//		}

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
//				parentsSumDf[ret_index[0]] += pts[i].parentDist;
//				parentsSumDfCounter[ret_index[0]]++;

				assert(ret_index[0] >= 0 && ret_index[0] < numPoints[lvl + 1]);
			} else {
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}

//		for (int i = 0 ; i < npts ; i++) {
//			int parentIdx = pts[i].parent;
//			if (parentIdx >= 0) {
//				pts[i].parentDist /= parentsSumDf[parentIdx];
//			}
//		}
	}

	// done.

	for (int i = 0; i < pyrLevelsUsed; i++)
		delete indexes[i];
}

bool CoarseInitializer::idepthLrEstimate(int lvl, Pnt *pntLeft) {
	const float uLeft = pntLeft->u;
	const float vLeft = pntLeft->v;

	const Vec3f pointLeft = Vec3f(uLeft, vLeft, 1);
	const Vec3f lineRight = Ftl[lvl] * pointLeft;

	// Left point at infinite depth projected to right.
	Vec3f pointRightFar = lrR * Ki[lvl] * pointLeft;
	assert(pointRightFar[2] > 0);
	pointRightFar /= pointRightFar[2];
	pointRightFar = Kr[lvl] * pointRightFar;

	float uRight = pointRightFar(0);
	float vRight = pointRightFar(1);

	if (uRight < 2.1f || vRight < 2.1f || uRight > (w[lvl] - 3.1) || vRight > (h[lvl] - 3.1)) {
		// Point projected at infinita depth is out of bounds in right image.
		return false;
	}

	// Calculate bi-linear pixel values for patch around left point.
	float leftPointPatch[patternNum]; // TODO maybe use pattern?
	for (int idx = 0; idx < patternNum; idx++) {
		leftPointPatch[idx] = getInterpolatedElement31(firstFrame->dIp[lvl], uLeft + patternP[idx][0], vLeft + patternP[idx][1], w[lvl]);
	}

	// Left point at inverse depth of 4 projected to right;
	Vec3f pointRightNear = lrR * Ki[lvl] * pointLeft + lrt * 3;
	assert(pointRightNear[2] > 0);
	pointRightNear /= pointRightNear[2];
	pointRightNear = Kr[lvl] * pointRightNear;

	const int numberOfSamples = 1 << (8 - (lvl / 2));

	float uInc = (pointRightFar(0) - pointRightNear(0)) / (numberOfSamples - 1);
	float vInc = (pointRightFar(1) - pointRightNear(1)) / (numberOfSamples - 1);

	int sample = 0;
	float ssdErrors[numberOfSamples];
	do {
		ssdErrors[sample] = 0;

		for (int idx = 0; idx < patternNum; idx++) {
			float hitColor = getInterpolatedElement31(firstFrame->dIrp[lvl], uRight + rotatedPattern[lvl][idx][0],
					vRight + rotatedPattern[lvl][idx][1], w[lvl]);
			if (std::isfinite(hitColor)) {
				float residual = hitColor - leftPointPatch[idx];
				float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
				ssdErrors[sample] += hw * residual * residual * (2 - hw);
			} else {
				ssdErrors[sample] += 1e5;
			}
		}

		sample++;
		uRight -= uInc;
		vRight -= vInc;
	} while (sample < numberOfSamples && uRight > 2.1f && (vRight > 2.1f && vRight < (h[lvl] - 3.1)));

	float smoothSsdErrors[sample];

	smoothSsdErrors[0] = (ssdErrors[0] + ssdErrors[0] + ssdErrors[1]) / 3;
	float sum3 = ssdErrors[0] + ssdErrors[1];
	for (int idx = 1; idx < (sample - 1); idx++) {
		sum3 += ssdErrors[idx + 1];
		smoothSsdErrors[idx] = sum3 / 3;
		sum3 -= ssdErrors[idx - 1];
	}
	smoothSsdErrors[sample - 1] = (sum3 + ssdErrors[sample - 1]) / 3;

	float bestErr1 = std::numeric_limits<float>::max(), bestErr2 = std::numeric_limits<float>::max();
	int bestIdx1 = std::numeric_limits<int>::min(), bestIdx2 = std::numeric_limits<int>::min();

	for (int idx = 0; idx < sample; idx++) {
		if (smoothSsdErrors[idx] < bestErr1) {
			bestErr1 = smoothSsdErrors[idx];
			bestIdx1 = idx;
		}
	}

	// Find second best not near best.
	for (int idx = 0; idx < sample; idx++) {
		if ((idx < (bestIdx1 - 1) || idx > (bestIdx1 + 1)) && smoothSsdErrors[idx] < bestErr2) {
			bestErr2 = smoothSsdErrors[idx];
			bestIdx2 = idx;
		}
	}

	if (bestErr1 < 2000 && (bestErr1 / bestErr2 < 0.75 || abs(bestIdx1 - bestIdx2) < 3)) {
		float disparity = uInc * bestIdx1; // Ignore v for now.
		float baseline = lrt.norm();
		float focalLength = Kr[lvl](0, 0);
		float idepth = disparity / (baseline * focalLength);
		pntLeft->idepthLr = idepth;
		assert(idepth >= 0);
		return true;
	}

	return false;
}
}

