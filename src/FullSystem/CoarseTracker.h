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
#include "util/settings.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"

namespace dso {
struct CalibHessian;
struct FrameHessian;
struct PointFrameResidual;

class CoarseTracker {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;

	CoarseTracker(int w, int h);
	~CoarseTracker();

	bool trackNewestCoarse(FrameHessian *newFrameHessian, SE3 &lastToNew_out, AffLight &aff_g2l_out, int coarsestLvl,
			Vec5 minResForAbort, IOWrap::Output3DWrapper *wrap = 0);

	void setCoarseTrackingRef(std::vector<FrameHessian*> frameHessians);

	void makeK(CalibHessian *HCalib);

	bool debugPrint, debugPlot;

	Mat33f K[MAX_PYR_LEVELS];
	Mat33f Kr[MAX_PYR_LEVELS];
	Mat33f Ki[MAX_PYR_LEVELS];
	Mat33f Kri[MAX_PYR_LEVELS];
	double fx[MAX_PYR_LEVELS];
	double fy[MAX_PYR_LEVELS];
	double fxi[MAX_PYR_LEVELS];
	double fyi[MAX_PYR_LEVELS];
	double cx[MAX_PYR_LEVELS];
	double cy[MAX_PYR_LEVELS];
	double cxi[MAX_PYR_LEVELS];
	double cyi[MAX_PYR_LEVELS];
	double fxr[MAX_PYR_LEVELS];
	double fyr[MAX_PYR_LEVELS];
	double cxr[MAX_PYR_LEVELS];
	double cyr[MAX_PYR_LEVELS];
	int w[MAX_PYR_LEVELS];
	int h[MAX_PYR_LEVELS];
	SE3 leftToRight;

	void debugPlotIDepthMap(float *minID, float *maxID, std::vector<IOWrap::Output3DWrapper*> &wraps);
	void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps);

	FrameHessian *lastRef;
	AffLight lastRef_aff_g2l;
	FrameHessian *newFrame;
	int refFrameID;

	// act as pure ouptut
	Vec5 lastResiduals;
	Vec3 lastFlowIndicators;
	double firstCoarseRMSE;
private:

	void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians);
	float *idepth[MAX_PYR_LEVELS];
	float *weightSums[MAX_PYR_LEVELS];
	float *weightSums_bak[MAX_PYR_LEVELS];

	Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
	void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);

	// pc buffers
	float *pc_u[MAX_PYR_LEVELS];
	float *pc_v[MAX_PYR_LEVELS];
	float *pc_idepth[MAX_PYR_LEVELS];
	float *pc_color[MAX_PYR_LEVELS];
	int pc_n[MAX_PYR_LEVELS];

	// warped buffers
	float *buf_warped_idepth;
	float *buf_warped_u;
	float *buf_warped_v;
	float *buf_warped_dx;
	float *buf_warped_dy;
	float *buf_warped_residual;
	float *buf_warped_weight;
	float *buf_warped_refColor;
	int buf_warped_n;

	std::vector<float*> ptrToDelete;

	Accumulator9 acc;
};

class CoarseDistanceMap {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;

	CoarseDistanceMap(int w, int h);
	~CoarseDistanceMap();

	void makeDistanceMap(std::vector<FrameHessian*> frameHessians, FrameHessian *frame);

	void makeInlierVotes(std::vector<FrameHessian*> frameHessians);

	void makeK(CalibHessian *HCalib);

	float *fwdWarpedIDDistFinal;

	Mat33f K[MAX_PYR_LEVELS];
	Mat33f Ki[MAX_PYR_LEVELS];
	float fx[MAX_PYR_LEVELS];
	float fy[MAX_PYR_LEVELS];
	float fxi[MAX_PYR_LEVELS];
	float fyi[MAX_PYR_LEVELS];
	float cx[MAX_PYR_LEVELS];
	float cy[MAX_PYR_LEVELS];
	float cxi[MAX_PYR_LEVELS];
	float cyi[MAX_PYR_LEVELS];
	int w[MAX_PYR_LEVELS];
	int h[MAX_PYR_LEVELS];

	void addIntoDistFinal(int u, int v);

private:

	PointFrameResidual **coarseProjectionGrid;
	int *coarseProjectionGridNum;
	Eigen::Vector2i *bfsList1;
	Eigen::Vector2i *bfsList2;

	void growDistBFS(int bfsNum);
};

}

