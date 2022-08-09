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
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "util/settings.h"

namespace dso {

EIGEN_STRONG_INLINE float derive_idepth(const Vec3f &t, const float &u, const float &v, const float &dxInterp,
		const float &dyInterp, const float &drescale) {
	return (dxInterp * drescale * (t[0] - t[2] * u) + dyInterp * drescale * (t[1] - t[2] * v)) * SCALE_IDEPTH;
}

EIGEN_STRONG_INLINE bool projectPoint(const float &u_pt, const float &v_pt, const float &idepth, const Mat33f &KRKi,
		const Vec3f &Kt, float &Ku, float &Kv) {
	Vec3f ptp = KRKi * Vec3f(u_pt, v_pt, 1) + Kt * idepth;
	Ku = ptp[0] / ptp[2];
	Kv = ptp[1] / ptp[2];
	return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G;
}

EIGEN_STRONG_INLINE bool projectPoint(const float u_pt, const float v_pt, const float idepth, CalibHessian *const&HCalib,
		const Mat33f &R, const Vec3f &t, float &drescale, float &u, float &v, float &Ku, float &Kv, Vec3f &KliP, float &new_idepth,
		bool leftToRight) {
	// inv(K) * (Homogeneous left pixel/image point) -> Point in left camera plane.
	KliP = Vec3f((u_pt - HCalib->cxl()) * HCalib->fxli(), (v_pt - HCalib->cyl()) * HCalib->fyli(), 1);

	Vec3f ptp = R * KliP + t * idepth;
	drescale = 1.0f / ptp[2];

	if (!(drescale > 0))
		return false;

	new_idepth = idepth * drescale;
	// Scale x,y to camera plane. z = 1.
	u = ptp[0] * drescale;
	v = ptp[1] * drescale;

	// Camera plane to image plane.
	if (leftToRight) {
		Ku = u * HCalib->fxlR() + HCalib->cxlR();
		Kv = v * HCalib->fylR() + HCalib->cylR();
	} else {
		Ku = u * HCalib->fxl() + HCalib->cxl();
		Kv = v * HCalib->fyl() + HCalib->cyl();
	}

	// OOB/Out of image check.
	return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G;
}

EIGEN_STRONG_INLINE bool projectPointLR(float ul, float vl, float idepthl, CalibHessian *const&HCalib, float &ur, float &vr, float &idepthr) {
	Vec3f KliP = Vec3f((ul - HCalib->cxl()) * HCalib->fxli(), (vl - HCalib->cyl()) * HCalib->fyli(), 1);

	SE3 leftToRight = HCalib->getLeftToRight();
	Mat33f R = leftToRight.rotationMatrix().cast<float>();
	Vec3f t = leftToRight.translation().cast<float>();

	Vec3f ptp = R * KliP + t * idepthl;
	float drescale = 1.0f / ptp[2];

	if (!(drescale > 0))
		return false;

	idepthr = idepthl * drescale;
	ur = ptp[0] * drescale * HCalib->fxlR() + HCalib->cxlR();
	vr = ptp[1] * drescale * HCalib->fylR() + HCalib->cylR();

	// OOB/Out of image check.
	return ur > 1.1f && vr > 1.1f && ur < wM3G && vr < hM3G;
}
}

