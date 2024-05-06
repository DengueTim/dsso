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

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

Eigen::IOFormat MatlabFmt(8, 0, ", ", "\n", "[", "]");


struct PointBlock {
	Eigen::Matrix<float,patternNum,8> jp;
	Eigen::Matrix<float,patternNum,1> jl;
	Eigen::Matrix<float,patternNum,1> r;

	Eigen::Matrix<float,patternNum,8> qTjp;
	Eigen::Matrix<float,patternNum,1> R0;
	Eigen::Matrix<float,patternNum,1> qTr;

	Eigen::Matrix<float,patternNum+1,8> qTjpD;
	Eigen::Matrix<float,patternNum+1,1> R0D;
	Eigen::Matrix<float,patternNum+1,1> qTrD;

	PointBlock() {
		reset();
	}

	void reset(){
		jp.setZero();
		jl.setZero();
		r.setZero();
		qTjp.setZero();
		R0.setZero();
		qTr.setZero();
		qTjpD.setZero();
		R0D.setZero();
		qTrD.setZero();
	}

	void addResidual(int index, float dTx, float dTy, float dTz, float dRa, float dRb, float dRc, float dAa, float dAb, float dD, float res) {
		assert(jp.row(index).isZero());
		jp.row(index) = Eigen::Matrix<float, 1, 8>(dTx, dTy, dTz, dRa, dRb, dRc, dAa, dAb);
		jl[index] = dD;
		r[index] = res;
	}
	
	void qrMarginalise() {
		assert(!jp.row(7).isZero());
		Eigen::HouseholderQR<Eigen::Matrix<float,patternNum,1>> qr(jl);
		Eigen::Matrix<float,patternNum,patternNum> Q = qr.householderQ();
		const Eigen::Matrix<float, patternNum, patternNum> &qT = Q.transpose();
		qTjp = qT * jp;
		R0 = qr.matrixQR().triangularView<Eigen::Upper>();
		qTr = qT * r;

		assert(jl.isApprox(Q * R0));

//		Eigen::Matrix<float,patternNum,10> lb;
//		lb.leftCols<8>() = jp;
//		lb.col(8) = jl;
//		lb.col(9) = r;
//		std::cerr << "\nlb:\n" << lb.format(MatlabFmt) << "\n";
//
//		std::cerr << "Q:\n" << Q.format(MatlabFmt) << "\n";
//		std::cerr << "R0:\n" << R0.format(MatlabFmt) << "\n";
//
//		Eigen::Matrix<float,patternNum,10> lbQr;
//		lbQr.leftCols<8>() = qTjp;
//		lbQr.col(8) = R0;
//		lbQr.col(9) = qTr;
//		std::cerr << "\nlbQr:\n" << lbQr.format(MatlabFmt) << "\n";

	}

	void applyDamping(float lambda) {
		qTjpD.topRows(patternNum) = qTjp;
		qTjpD.bottomRows(1).setZero();
		R0D.topRows(patternNum) = R0;
		R0D[patternNum] = 1/(1+lambda);
		qTrD.topRows(patternNum) = qTr;
		qTrD.bottomRows(1).setZero();

//		Eigen::Matrix<float,patternNum+1,10> lbQrD;
//		lbQrD.leftCols<8>() = qTjpD;
//		lbQrD.col(8) = R0D;
//		lbQrD.col(9) = qTrD;
//		std::cerr << "\nlbQrD:\n" << lbQrD.format(MatlabFmt) << "\n";

		Eigen::Matrix<float,patternNum+1,8> t(qTjpD);
		Eigen::Matrix<float,patternNum+1,1> r2(R0D);
		Eigen::HouseholderQR<Eigen::Matrix<float,patternNum+1,1>> qr(R0D);
		Eigen::Matrix<float,patternNum+1,patternNum+1> Q = qr.householderQ();
		const Eigen::Matrix<float, patternNum+1, patternNum+1> &qT = Q.transpose();
		auto d = qT * qTjpD;
		qTjpD = d;
		R0D = qr.matrixQR().triangularView<Eigen::Upper>();
		auto t2 = qT * qTrD;
		qTrD = t2;

//		if (true ||!qTjpD.isApprox(t)) {
//			std::cerr << "\nR0D pre QR:\n" << r2.format(MatlabFmt) << "\n";
//
//			std::cerr << "\nqTjpD:\n" << qTjpD.format(MatlabFmt) << "\n";
//			std::cerr << "\nt:\n" << t.format(MatlabFmt) << "\n";
//			std::cerr << "\nQ:\n" << Q.format(MatlabFmt) << "\n";
//			abort();
//		}
	}

	void addPoseContribution(Eigen::Matrix<float, 8, 8> &Hpp, Eigen::Matrix<float, 8, 1> &bp) {
		const Eigen::Matrix<float, 8, 8> &q2TjpD = qTjpD.bottomRows<8>();
		const Eigen::Matrix<float, 8, 1> &q2TrD = qTrD.bottomRows<8>();
		Eigen::DenseBase<Eigen::Matrix<float, 8, 8, 0, 8, 8>>::ConstTransposeReturnType &q2TjpDT = q2TjpD.transpose();

		Hpp += q2TjpDT * q2TjpD;
		bp += q2TjpDT * q2TrD;
	}

	float getLandmarkIncFromPoseInt(Eigen::Matrix<float, 8, 1> poseInc) {
		float x = (qTjpD.topRows(1) * poseInc)[0];
		return -1/R0D[0] * (qTrD[0] + x);
	}
};


CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0,0), thisToNext(SE3())
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		points[lvl] = 0;
		numPoints[lvl] = 0;
	}

	JbBuffer = new Vec10f[ww*hh];
	JbBuffer_new = new Vec10f[ww*hh];

	pBlocks = new PointBlock[ww*hh];
	pBlocksNew = new PointBlock[ww*hh];


	frameID=-1;
	fixAffine=true;
	printDebug=false;

	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_TRANS;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_ROT;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer()
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		if(points[lvl] != 0) delete[] points[lvl];
	}

	delete[] pBlocks;
	delete[] pBlocksNew;
	delete[] JbBuffer;
	delete[] JbBuffer_new;
}


bool CoarseInitializer::trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
	newFrame = newFrameHessian;

    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushLiveFrame(newFrameHessian);

	int maxIterations[] = {5,5,10,30,50};

	alphaK = 2.5*2.5;//*freeDebugParam1*freeDebugParam1;
	alphaW = 150*150;//*freeDebugParam2*freeDebugParam2;
	regWeight = 0.8;//*freeDebugParam4;
	couplingWeight = 1;//*freeDebugParam5;

	if(!snapped)
	{
		thisToNext.translation().setZero();
		for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
		{
			int npts = numPoints[lvl];
			Pnt* ptsl = points[lvl];
			for(int i=0;i<npts;i++)
			{
				ptsl[i].iR = 1;
				ptsl[i].idepth_new = 1;
				ptsl[i].lastHessian = 0;
			}
		}
	}


	SE3 refToNew_current = thisToNext;
	AffLight refToNew_aff_current = thisToNext_aff;

	//refToNew_current = SE3::exp(Vec6(0.0,0.0,0.0001,0.0,0.0,0.0)) * refToNew_current;

	if(firstFrame->ab_exposure>0 && newFrame->ab_exposure>0)
		refToNew_aff_current = AffLight(logf(newFrame->ab_exposure /  firstFrame->ab_exposure),0); // coarse approximation.


	Vec3f latestRes = Vec3f::Zero();
	for(int lvl=pyrLevelsUsed-1; lvl>=0; lvl--)
	{
		if(lvl<pyrLevelsUsed-1) {
			//abort();
			propagateDown(lvl + 1);
		}

		float lambda = 0.1;

		Mat88f H,Hsc; Vec8f b,bsc;
		resetPoints(lvl);
		Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, lambda, false);
		applyStep(lvl);

		float eps = 1e-4;
		int fails=0;

		if(printDebug)
		{
			printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
					lvl, 0, lambda,
					"INITIA",
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					(resOld[0]+resOld[1]) / resOld[2],
					(resOld[0]+resOld[1]) / resOld[2],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() <<"\n";
		}

		int iteration=0;
		while(true)
		{
			std::cerr << "\nlambda:" << lambda << "\n";
			Mat88f Hl = H;
			for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda);
			for(int i=0;i<8;i++) Hpp(i,i) *= (1+lambda);
			Hl -= Hsc*(1/(1+lambda));
			Vec8f bl = b - bsc*(1/(1+lambda));

			Hl = wM * Hl * wM * (0.01f/(w[lvl]*h[lvl]));
			bl = wM * bl * (0.01f/(w[lvl]*h[lvl]));

			Vec8f incSc;
			Vec8f incQr;
			if(fixAffine)
			{
//				if (!Hl.isApprox(Hpp)) {
//					std::cerr << "\nHpp, QR:\n" << Hpp.format(MatlabFmt);
//					std::cerr << "\nHl, SC:\n" << Hl.format(MatlabFmt);
//					std::cerr << "\nH QR/SC:\n" << (Hpp.array() / Hl.array()).format(MatlabFmt);
//				}
//
//				if (!bl.isApprox(bp)) {
//					std::cerr << "\nbp, QR:\n" << bp.format(MatlabFmt);
//					std::cerr << "\nbl, SC:\n" << bl.format(MatlabFmt);
//					std::cerr << "\nb QR/SC:\n" << (bp.array() / bl.array()).format(MatlabFmt);
//				}
				incSc.head<6>() = - (wM.toDenseMatrix().topLeftCorner<6,6>() * (Hl.topLeftCorner<6,6>().ldlt().solve(bl.head<6>())));
				incSc.tail<2>().setZero();
				
								
				incQr.head<6>() = - wM.toDenseMatrix().topLeftCorner<6,6>() * (Hpp.topLeftCorner<6,6>().ldlt().solve(bp.head<6>()));
				incQr.tail<2>().setZero();

//				if (!incQr.isApprox(incSc, 0.001)) {
//					std::cerr << "\nincQr:\n" << incQr.format(MatlabFmt);
//					std::cerr << "\nincSc:\n" << incSc.format(MatlabFmt);
//					std::cerr << "\ninc QR/SC:\n" << (incQr.array() / incSc.array()).format(MatlabFmt);
//					abort();
//				}
			} else {
				incSc = - (wM * (Hl.ldlt().solve(bl)));	//=-H^-1 * b.
			}

			Vec8f inc(incQr);

			SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
			AffLight refToNew_aff_new = refToNew_aff_current;
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];
			doStep(lvl, lambda, inc);

			Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
			Mat88f HppOld(Hpp);
			Vec8f bpOld(bp);
			Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, lambda, false);
			Vec3f regEnergy = calcEC(lvl);

			float eTotalNew = (resNew[0]+resNew[1]+regEnergy[1]);
			float eTotalOld = (resOld[0]+resOld[1]+regEnergy[0]);

			bool accept = eTotalOld > eTotalNew;

			if(printDebug)
			{
				printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						(accept ? "ACCEPT" : "REJECT"),
						sqrtf((float)(resOld[0] / resOld[2])),
						sqrtf((float)(regEnergy[0] / regEnergy[2])),
						sqrtf((float)(resOld[1] / resOld[2])),
						sqrtf((float)(resNew[0] / resNew[2])),
						sqrtf((float)(regEnergy[1] / regEnergy[2])),
						sqrtf((float)(resNew[1] / resNew[2])),
						eTotalOld / resNew[2],
						eTotalNew / resNew[2],
						inc.norm());
				std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() <<"\n";
			}

			if(accept)
			{

				if(resNew[1] == alphaK*numPoints[lvl])
					snapped = true;
				H = H_new;
				b = b_new;
				Hsc = Hsc_new;
				bsc = bsc_new;
				resOld = resNew;
				refToNew_aff_current = refToNew_aff_new;
				refToNew_current = refToNew_new;
				applyStep(lvl);
				optReg(lvl);
				lambda *= 0.5;
				fails=0;
				if(lambda < 0.0001) lambda = 0.0001;
			}
			else
			{
				fails++;
				lambda *= 4;
				if(lambda > 10000) lambda = 10000;
				Hpp = HppOld;
				bp = bpOld;
			}

			if(!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2)
				 break;

			iteration++;
		}
		latestRes = resOld;

	}



	thisToNext = refToNew_current;
	thisToNext_aff = refToNew_aff_current;

	for(int i=0;i<pyrLevelsUsed-1;i++)
		propagateUp(i);




	frameID++;
	if(!snapped) snappedAt=0;

	if(snapped && snappedAt==0)
		snappedAt = frameID;



    debugPlot(0,wraps);



	return snapped && frameID > snappedAt+5;
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    bool needCall = false;
    for(IOWrap::Output3DWrapper* ow : wraps)
        needCall = needCall || ow->needPushDepthImage();
    if(!needCall) return;


	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

	MinimalImageB3 iRImg(wl,hl);

	for(int i=0;i<wl*hl;i++)
		iRImg.at(i) = Vec3b(colorRef[i][0],colorRef[i][0],colorRef[i][0]);


	int npts = numPoints[lvl];

	float nid = 0, sid=0;
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(point->isGood)
		{
			nid++;
			sid += point->iR;
		}
	}
	float fac = nid / sid;



	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;

		if(!point->isGood)
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));

		else
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,makeRainbow3B(point->iR*fac));
	}


	//IOWrap::displayImage("idepth-R", &iRImg, false);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImage(&iRImg);
}


/**
 * Calculates residual, Hessian and Hessian-block needed for re-substituting depth. Returns accumulated photometric error/residual for points.
 */
Vec3f CoarseInitializer::calcResAndGS(
		int lvl, Mat88f &H_out, Vec8f &b_out,
		Mat88f &H_out_sc, Vec8f &b_out_sc,
		const SE3 &refToNew, AffLight refToNew_aff,
		float lambda,
		bool plot)
{
	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
	Eigen::Vector3f* colorNew = newFrame->dIp[lvl];

	Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
	Vec3f t = refToNew.translation().cast<float>();
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];


	Accumulator11 E;
	acc9.initialize();
	E.initialize();

	Hpp.setZero();
	bp.setZero();

	std::cerr << "\nrefToNew:\n" << refToNew.matrix().format(MatlabFmt) << "\n";

	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;

		point->maxstep = 1e10;
		if(!point->isGood)
		{
			E.updateSingle((float)(point->energy[0]));
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
		pBlocksNew[i].reset();

		// sum over all residuals.
		bool isGood = true;
		float energy=0;
		for(int idx=0;idx<patternNum;idx++)
		{
			int dx = patternP[idx][0];
			int dy = patternP[idx][1];


			Vec3f pt = RKi * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;
			float u = pt[0] / pt[2];
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl;
			float Kv = fyl * v + cyl;
			float new_idepth = point->idepth_new/pt[2];

			if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0))
			{
				isGood = false;
				break;
			}

			Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
			//Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

			//float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
			float rlR = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl);

			if(!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
			{
				isGood = false;
				break;
			}


			float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			hw = 1;
			energy += hw *residual*residual*(2-hw);

			float dxdd = (t[0]-t[2]*u)/pt[2];
			float dydd = (t[1]-t[2]*v)/pt[2];

			if(hw < 1) hw = sqrtf(hw);
			float dxInterp = hw*hitColor[1]*fxl;
			float dyInterp = hw*hitColor[2]*fyl;
			dp0[idx] = new_idepth*dxInterp;
			dp1[idx] = new_idepth*dyInterp;
			dp2[idx] = -new_idepth*(u*dxInterp + v*dyInterp);
			dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp;
			dp4[idx] = (1+u*u)*dxInterp + u*v*dyInterp;
			dp5[idx] = -v*dxInterp + u*dyInterp;
			dp6[idx] = - hw*r2new_aff[0] * rlR;
			dp7[idx] = - hw*1;
			dd[idx] = dxInterp * dxdd + dyInterp * dydd;
			r[idx] = hw*residual;

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

		if(!isGood || energy > point->outlierTH*20)
		{
			E.updateSingle((float)(point->energy[0]));
			point->isGood_new = false;
			point->energy_new = point->energy;
			continue;
		}


		// add into energy.
		E.updateSingle(energy);
		point->isGood_new = true;
		point->energy_new[0] = energy;

		// update Hessian matrix.
		for(int idx=0;idx+3<patternNum;idx+=4) {
			acc9.updateSSE(
				_mm_load_ps(((float *)(&dp0)) + idx),
				_mm_load_ps(((float *)(&dp1)) + idx),
				_mm_load_ps(((float *)(&dp2)) + idx),
				_mm_load_ps(((float *)(&dp3)) + idx),
				_mm_load_ps(((float *)(&dp4)) + idx),
				_mm_load_ps(((float *)(&dp5)) + idx),
				_mm_load_ps(((float *)(&dp6)) + idx),
				_mm_load_ps(((float *)(&dp7)) + idx),
				_mm_load_ps(((float *)(&r)) + idx));
		}


		for(int idx=((patternNum>>2)<<2); idx < patternNum; idx++)
			acc9.updateSingle(
					(float)dp0[idx],(float)dp1[idx],(float)dp2[idx],(float)dp3[idx],
					(float)dp4[idx],(float)dp5[idx],(float)dp6[idx],(float)dp7[idx],
					(float)r[idx]);

		for(int idx=0; idx<patternNum; idx++) {
			pBlocksNew[i].addResidual(idx,dp0[idx],dp1[idx],dp2[idx],dp3[idx],dp4[idx],dp5[idx],dp6[idx],dp7[idx],dd[idx],r[idx]);
		}

		pBlocksNew[i].qrMarginalise();
		pBlocksNew[i].applyDamping(lambda);
		pBlocksNew[i].addPoseContribution(Hpp, bp);

		//break;
	}

	acc9.finish();

	// Update depth regularisation energy and accumulate.
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
		{
			E.updateSingle((float)(point->energy[1]));
		}
		else
		{
			point->energy_new[1] = (point->idepth_new-1)*(point->idepth_new-1);
			E.updateSingle((float)(point->energy_new[1]));
		}
	}

	// calculate alpha energy, and decide if we cap it.
	float alphaEnergy = alphaW*(refToNew.translation().squaredNorm() * npts);

	//printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);


	// compute alpha opt.
	float alphaOpt;
	if(alphaEnergy > alphaK*npts)
	{
		alphaOpt = 0;
		alphaEnergy = alphaK*npts;
	}
	else
	{
		alphaOpt = alphaW;
	}


	acc9SC.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
			continue;

		point->lastHessian_new = JbBuffer_new[i][9];

		if(alphaOpt==0) {
			JbBuffer_new[i][8] += couplingWeight*(point->idepth_new - point->iR);
			JbBuffer_new[i][9] += couplingWeight;
		} else {
			JbBuffer_new[i][8] += alphaOpt*(point->idepth_new - 1);
			JbBuffer_new[i][9] += alphaOpt;
		}

		JbBuffer_new[i][9] = 1/(1+JbBuffer_new[i][9]);
		acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
				(float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],
				(float)JbBuffer_new[i][8],(float)JbBuffer_new[i][9]);
	}
	acc9SC.finish();


	//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
	H_out = acc9.H.topLeftCorner<8,8>();// / acc9.num;
	b_out = acc9.H.topRightCorner<8,1>();// / acc9.num;
	H_out_sc = acc9SC.H.topLeftCorner<8,8>();// / acc9.num;
	b_out_sc = acc9SC.H.topRightCorner<8,1>();// / acc9.num;

//	H_out(0,0) += alphaOpt*npts;
//	H_out(1,1) += alphaOpt*npts;
//	H_out(2,2) += alphaOpt*npts;
//
//	Vec3f tlog = refToNew.log().head<3>().cast<float>();
//	b_out[0] += tlog[0]*alphaOpt*npts;
//	b_out[1] += tlog[1]*alphaOpt*npts;
//	b_out[2] += tlog[2]*alphaOpt*npts;

	E.finish();

	return Vec3f(E.A, alphaEnergy ,E.num);
}

float CoarseInitializer::rescale()
{
	float factor = 20*thisToNext.translation().norm();
//	float factori = 1.0f/factor;
//	float factori2 = factori*factori;
//
//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
//	{
//		int npts = numPoints[lvl];
//		Pnt* ptsl = points[lvl];
//		for(int i=0;i<npts;i++)
//		{
//			ptsl[i].iR *= factor;
//			ptsl[i].idepth_new *= factor;
//			ptsl[i].lastHessian *= factori2;
//		}
//	}
//	thisToNext.translation() *= factori;

	return factor;
}


Vec3f CoarseInitializer::calcEC(int lvl)
{
	if(!snapped) return Vec3f(0,0,numPoints[lvl]);
	AccumulatorX<2> E;
	E.initialize();
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(!point->isGood_new) continue;
		float rOld = (point->idepth-point->iR);
		float rNew = (point->idepth_new-point->iR);
		E.updateNoWeight(Vec2f(rOld*rOld,rNew*rNew));

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}
	E.finish();

	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	return Vec3f(couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], E.num);
}

/**
 * Update iR (inverse depth local average/regularisation.)
 * @param lvl
 */
void CoarseInitializer::optReg(int lvl)
{
	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	if(!snapped)
	{
		for(int i=0;i<npts;i++)
			ptsl[i].iR = 1;
		return;
	}


	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood) continue;

		float idnn[10];
		int nnn=0;
		for(int j=0;j<10;j++)
		{
			if(point->neighbours[j] == -1) continue;
			Pnt* other = ptsl+point->neighbours[j];
			if(!other->isGood) continue;
			idnn[nnn] = other->iR;
			nnn++;
		}

		if(nnn > 2)
		{
			std::nth_element(idnn,idnn+nnn/2,idnn+nnn);
			point->iR = (1-regWeight)*point->idepth + regWeight*idnn[nnn/2];
		}
	}

}



void CoarseInitializer::propagateUp(int srcLvl)
{
	assert(srcLvl+1<pyrLevelsUsed);
	// set idepth of target

	int nptss= numPoints[srcLvl];
	int nptst= numPoints[srcLvl+1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl+1];

	// set to zero.
	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		parent->iR=0;
		parent->iRSumNum=0;
	}

	for(int i=0;i<nptss;i++)
	{
		Pnt* point = ptss+i;
		if(!point->isGood) continue;

		Pnt* parent = ptst + point->parent;
		parent->iR += point->iR * point->lastHessian;
		parent->iRSumNum += point->lastHessian;
	}

	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		if(parent->iRSumNum > 0)
		{
			parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
			parent->isGood = true;
		}
	}

	optReg(srcLvl+1);
}

void CoarseInitializer::propagateDown(int srcLvl)
{
	assert(srcLvl>0);
	// set idepth of target

	int nptst= numPoints[srcLvl-1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl-1];

	for(int i=0;i<nptst;i++)
	{
		Pnt* point = ptst+i;
		Pnt* parent = ptss+point->parent;

		if(!parent->isGood || parent->lastHessian < 0.1) continue;
		if(!point->isGood)
		{
			point->iR = point->idepth = point->idepth_new = parent->iR;
			point->isGood=true;
			point->lastHessian=0;
		}
		else
		{
			float newiR = (point->iR*point->lastHessian*2 + parent->iR*parent->lastHessian) / (point->lastHessian*2+parent->lastHessian);
			point->iR = point->idepth = point->idepth_new = newiR;
		}
	}
	optReg(srcLvl-1);
}


void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
{
	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		Eigen::Vector3f* dINew_l = data[lvl];
		Eigen::Vector3f* dINew_lm = data[lvlm1];

		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
				dINew_l[x + y*wl][0] = 0.25f * (dINew_lm[2*x   + 2*y*wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1][0] +
													dINew_lm[2*x   + 2*y*wlm1+wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1+wlm1][0]);

		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			dINew_l[idx][1] = 0.5f*(dINew_l[idx+1][0] - dINew_l[idx-1][0]);
			dINew_l[idx][2] = 0.5f*(dINew_l[idx+wl][0] - dINew_l[idx-wl][0]);
		}
	}
}
void CoarseInitializer::setFirst(	CalibHessian* HCalib, FrameHessian* newFrameHessian)
{

	makeK(HCalib);
	firstFrame = newFrameHessian;

	PixelSelector<FrameHessian> sel(w[0],h[0]);

	float* statusMap = new float[w[0]*h[0]];
	bool* statusMapB = new bool[w[0]*h[0]];

	float densities[] = {0.03,0.05,0.15,0.5,1};
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		sel.currentPotential = 3;
		int npts;
		if(lvl == 0)
			npts = sel.makeMaps(firstFrame, statusMap,densities[lvl]*w[0]*h[0],1,false,2);
		else
			npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);



		if(points[lvl] != 0) delete[] points[lvl];
		points[lvl] = new Pnt[npts];

		// set idepth map to initially 1 everywhere.
		int wl = w[lvl], hl = h[lvl];
		Pnt* pl = points[lvl];
		int nl = 0;
		for(int y=patternPadding+1;y<hl-patternPadding-2;y++)
		for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
		{
			//if(x==2) printf("y=%d!\n",y);
			if((lvl!=0 && statusMapB[x+y*wl]) || (lvl==0 && statusMap[x+y*wl] != 0))
			{
				//assert(patternNum==9);
				pl[nl].u = x+0.1;
				pl[nl].v = y+0.1;
				pl[nl].idepth = 1;
				pl[nl].iR = 1;
				pl[nl].isGood=true;
				pl[nl].energy.setZero();
				pl[nl].lastHessian=0;
				pl[nl].lastHessian_new=0;
				pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];

//				Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
//				float sumGrad2=0;
//				for(int idx=0;idx<patternNum;idx++)
//				{
//					int dx = patternP[idx][0];
//					int dy = patternP[idx][1];
//					float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
//					sumGrad2 += absgrad;
//				}
//
//				float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
//				pl[nl].outlierTH = patternNum*gth*gth;

				pl[nl].outlierTH = patternNum*setting_outlierTH;

				nl++;
				assert(nl <= npts);
			}
		}


		numPoints[lvl]=nl;
	}
	delete[] statusMap;
	delete[] statusMapB;

	makeNN();

	thisToNext=SE3();
	snapped = false;
	frameID = snappedAt = 0;
}

void CoarseInitializer::resetPoints(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		pts[i].energy.setZero();
		pts[i].idepth_new = pts[i].idepth;


		if(lvl==pyrLevelsUsed-1 && !pts[i].isGood)
		{
			float snd=0, sn=0;
			for(int n = 0;n<10;n++)
			{
				if(pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
				snd += pts[pts[i].neighbours[n]].iR;
				sn += 1;
			}

			if(sn > 0)
			{
				pts[i].isGood=true;
				pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd/sn;
			}
		}
	}
}

/**
 * Updates good points idepth_new values using inc.
 * @param lvl
 * @param lambda
 * @param inc
 */
void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
{

	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood) continue;


//		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
//		float step = - b * JbBuffer[i][9] / (1+lambda);
		float step = pBlocks[i].getLandmarkIncFromPoseInt(inc);

		float maxstep = maxPixelStep*pts[i].maxstep;
		if(maxstep > idMaxStep) maxstep=idMaxStep;

		if(step >  maxstep) step = maxstep;
		if(step < -maxstep) step = -maxstep;

		float newIdepth = pts[i].idepth + step;
		if(newIdepth < 1e-3 ) newIdepth = 1e-3;
		if(newIdepth > 50) newIdepth = 50;
		pts[i].idepth_new = newIdepth;
	}

}
/**
 * Apply _new values to values.
 * @param lvl
 */
void CoarseInitializer::applyStep(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood)
		{
			pts[i].idepth = pts[i].idepth_new = pts[i].iR;
			continue;
		}
		pts[i].energy = pts[i].energy_new;
		pts[i].isGood = pts[i].isGood_new;
		pts[i].idepth = pts[i].idepth_new;
		pts[i].lastHessian = pts[i].lastHessian_new;
	}
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
	std::swap<PointBlock*>(pBlocks, pBlocksNew);
}

void CoarseInitializer::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}




void CoarseInitializer::makeNN()
{
	const float NNDistFactor=0.05;

	typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
			FLANNPointcloud,2> KDTree;

	// build indices
	FLANNPointcloud pcs[PYR_LEVELS];
	KDTree* indexes[PYR_LEVELS];
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
		indexes[i]->buildIndex();
	}

	const int nn=10;

	// find NN & parents
	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	{
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		int ret_index[nn];
		float ret_dist[nn];
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for(int i=0;i<npts;i++)
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			Vec2f pt = Vec2f(pts[i].u,pts[i].v);
			indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
			int myidx=0;
			float sumDF = 0;
			for(int k=0;k<nn;k++)
			{
				pts[i].neighbours[myidx]=ret_index[k];
				float df = expf(-ret_dist[k]*NNDistFactor);
				sumDF += df;
				pts[i].neighboursDist[myidx]=df;
				assert(ret_index[k]>=0 && ret_index[k] < npts);
				myidx++;
			}
			for(int k=0;k<nn;k++)
				pts[i].neighboursDist[k] *= 10/sumDF;


			if(lvl < pyrLevelsUsed-1 )
			{
				resultSet1.init(ret_index, ret_dist);
				pt = pt*0.5f-Vec2f(0.25f,0.25f);
				indexes[lvl+1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

				pts[i].parent = ret_index[0];
				pts[i].parentDist = expf(-ret_dist[0]*NNDistFactor);

				assert(ret_index[0]>=0 && ret_index[0] < numPoints[lvl+1]);
			}
			else
			{
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}



	// done.

	for(int i=0;i<pyrLevelsUsed;i++)
		delete indexes[i];
}
}

