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

Eigen::IOFormat MatlabFmt(16, 0, ", ", "\n", "\t", ";", "[", "\n]");

const float EPS = 1;

template <typename Scalar>
struct StepState {
	Eigen::Matrix<Scalar,8,8> QH;
	Eigen::Matrix<Scalar,8,1> Qb;
	Eigen::Matrix<Scalar,8,8> QHpp;
	Eigen::Matrix<Scalar,8,1> Qbpp;


	Mat88 SHpp;
	Vec8 Sbp;
	Mat88 SHsc;
	Vec8 Sbsc;

	Mat88f H,Hsc; Vec8f b,bsc;

	void reset() {
		QH.setZero();
		Qb.setZero();
		QHpp.setZero();
		Qbpp.setZero();

		SHpp.setZero();
		Sbp.setZero();
		SHsc.setZero();
		Sbsc.setZero();

		H.setZero();
		Hsc.setZero();
		b.setZero();
		bsc.setZero();
	}
};

template <typename Scalar>
struct PointBlock {
	Eigen::Matrix<Scalar,patternNum,8> jp;
	Eigen::Matrix<Scalar,patternNum,1> jl;
	Eigen::Matrix<Scalar,patternNum,1> r;

	Scalar Hll;
	Scalar HllSqrt;

	Eigen::Matrix<Scalar,patternNum,8> qTjp;
	Eigen::Matrix<Scalar,patternNum,1> R0;
	Eigen::Matrix<Scalar,patternNum,1> qTr;

	Eigen::Matrix<Scalar,patternNum+1,8> qTjpD;
	Eigen::Matrix<Scalar,patternNum+1,1> R0D;
	Eigen::Matrix<Scalar,patternNum+1,1> qTrD;

	float HllDampingAdd_, alphaW_;

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
		jp.row(index) = Eigen::Matrix<Scalar, 1, 8>(dTx, dTy, dTz, dRa, dRb, dRc, dAa, dAb);
		jl[index] = dD;
		r[index] = res;
	}

	void qrMarginalise() {
		assert(!jp.row(7).isZero());
		Hll = jl.template cast<Scalar>().squaredNorm();
		if (Hll == 0) {
			// No information from point depth. Q == I
			HllSqrt = 0;
			qTjp = jp;
			R0 = jl;
			qTr = r;
		} else {
			HllSqrt = sqrt(Hll);
			Eigen::HouseholderQR<Eigen::Matrix<Scalar,patternNum,1>> qr(jl / HllSqrt);
			Eigen::Matrix<Scalar,patternNum,patternNum> Q = qr.householderQ();
			R0 = qr.matrixQR().template triangularView<Eigen::Upper>();
			R0(0) *= HllSqrt;

			assert(R0.tail(patternNum - 1).isZero());
			assert(abs(1 - R0(0) * R0(0) / Hll) < 0.000001);

//			if (R0(0) < 0) {
//				// Does the sign matter?
//				R0(0) = -R0(0);
//				Q = -Q;
//			}

			const Eigen::Matrix<Scalar, patternNum, patternNum> qT = Q.transpose();
			qTjp = qT * jp;
			qTr = qT * r;
		}

//		Eigen::Matrix<float,patternNum,10> lb;
//		lb.leftCols<8>() = jp;
//		lb.col(8) = jl;
//		lb.col(9) = r;
//		std::cerr << "\nlb=\n" << lb.format(MatlabFmt) << "\n";

//		std::cerr << "Q=" << Q.format(MatlabFmt) << "\n";
//		std::cerr << "qT=" << qT.format(MatlabFmt) << "\n";
//		std::cerr << "R0=" << R0.format(MatlabFmt) << "\n";
//		std::cerr << "jp=" << jp.format(MatlabFmt) << "\n";
//		std::cerr << "jl=" << jl.format(MatlabFmt) << "\n";
//		std::cerr << "qTjp=" << qTjp.format(MatlabFmt) << "\n";
//
//		Eigen::Matrix<float,patternNum,10> lbQr;
//		lbQr.leftCols<8>() = qTjp;
//		lbQr.col(8) = R0;
//		lbQr.col(9) = qTr;
//		std::cerr << "\nlbQr:\n" << lbQr.format(MatlabFmt) << "\n";

	}

	void applyJlDamping(float dampingAdd) {
		HllDampingAdd_ = dampingAdd;

		qTjpD.topRows(patternNum) = qTjp;
		qTjpD.bottomRows(1).setZero();
		R0D.topRows(patternNum) = R0;
		R0D[patternNum] = sqrt(dampingAdd);

		qTrD.topRows(patternNum) = qTr;
		qTrD.bottomRows(1).setZero();

		Eigen::HouseholderQR<Eigen::Matrix<Scalar,patternNum+1,1>> qr(R0D);
		Eigen::Matrix<Scalar,patternNum+1,patternNum+1> Q = qr.householderQ();
		const Eigen::Matrix<Scalar, patternNum+1, patternNum+1> &qT = Q.transpose();
		qTjpD = qT * qTjpD;
		R0D = qr.matrixQR().template triangularView<Eigen::Upper>();
		qTrD = qT * qTrD;
	}

	void addPoseContribution(StepState<Scalar> &ss, float alphaW) {
		const Eigen::Matrix<Scalar, 8, 8> q2TjpD = qTjpD.template bottomRows<8>();
		const Eigen::Matrix<Scalar, 8, 8> jpTq2D = q2TjpD.transpose();

		const Eigen::Matrix<Scalar, 8, 1> q2TrD = qTrD.template bottomRows<8>();

		const Eigen::Matrix<Scalar, 8, 1> jpTq1D = qTjpD.template topRows<1>().transpose();

		ss.QH += jpTq2D * q2TjpD;
		ss.Qb += jpTq2D * q2TrD - jpTq1D * (alphaW / R0D[0]);

		ss.QHpp += (jp.transpose() * jp);
		ss.Qbpp += (jp.transpose() * r);

//		// Un dampened..
//		const Eigen::Matrix<float, 7, 8> &q2Tjp = qTjp.template bottomRows<7>().template cast<float>();
//		const Eigen::Matrix<float, 7, 1> &q2Tr = qTr.template bottomRows<7>().template cast<float>();
//		Eigen::DenseBase<Eigen::Matrix<float, 7, 8, 0, 7, 8>>::ConstTransposeReturnType &jpTq2 = q2Tjp.transpose();
//
//		Hpp += jpTq2 * q2Tjp;
//		bp += jpTq2 * q2Tr;
	}

	void addPoseScContribution(StepState<Scalar> &ss, float alphaW) {
		alphaW_ = alphaW;

		ss.SHpp += (jp.transpose() * jp).template cast<double>();
		ss.Sbp += (jp.transpose() * r).template cast<double>();

		const Eigen::Matrix<float, 1, 8> q1TjpD = qTjpD.template topRows<1>().template cast<float>();
		const Eigen::Matrix<float, 8, 1> jpTq1D = q1TjpD.transpose();
		const Eigen::Matrix<float, 8, 1> Hpl = (jp.transpose() * jl).template cast<float>();
		const Eigen::Matrix<float, 8, 1> HplHlli = Hpl * 1/(HllDampingAdd_ + Hll);

		auto A = HplHlli * Hpl.transpose();
		auto B = jpTq1D * q1TjpD;
		ss.SHsc += A.template cast<double>();
		if (!A.isApprox(B, 0.05)) {
			std::cerr << "\nA:\n" << A.format(MatlabFmt);
			std::cerr << "\nB:\n" << B.format(MatlabFmt);
			std::cerr << "\nA/B:\n" << (A.array() / B.array()).format(MatlabFmt);
			abort();
		}


		auto C = HplHlli * (jl.transpose() * r + alphaW);
		auto D = jpTq1D * qTrD(0); // Todo include alphaW here.
		ss.Sbsc += C.template cast<double>();
//		if (!C.isApprox(D)) {
//			std::cerr << "\nC:\n" << C.format(MatlabFmt);
//			std::cerr << "\nD:\n" << D.format(MatlabFmt);
//			std::cerr << "\nC/D:\n" << (C.array() / D.array()).format(MatlabFmt);
//			abort();
//		}
	}

	float getLandmarkIncFromPoseInc(Eigen::Matrix<float, 8, 1> poseInc) {
		float x = (qTjpD.topRows(1).template cast<float>() * poseInc)[0];
		x = -1/R0D[0] * (qTrD[0] + x);
		if (Hll == 0) {
			assert (x == 0);
		}
		x -= alphaW_ / (R0D[0] * R0D[0]);
		return x;
	}

	float getScLandmarkIncFromPoseInc(Eigen::Matrix<float, 8, 1> poseInc) {
		const Eigen::Matrix<float, 8, 1> Hlp = (jp.transpose() * jl).template cast<float>().transpose();

		float b = (jl.transpose() * r + alphaW_) + Hlp.dot(poseInc);
		float stepSc = - b * 1/(HllDampingAdd_ + Hll);

		return stepSc;
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

	pBlocks = new PointBlock<QR_PRECISION>[ww*hh];
	pBlocksNew = new PointBlock<QR_PRECISION>[ww*hh];


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

//	refToNew_current = SE3::exp(Vec6(0.0012,-0.0016,0.0021,0.1,0.0,0.0)) * refToNew_current;
//	refToNew_current = SE3::exp(Vec6(0.001,0.0,0.0,0.0,0.0,0.0)) * refToNew_current;


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

		StepState<QR_PRECISION> ss;

		resetPoints(lvl);
		Vec3f resOld = calcResAndGS(lvl, ss, refToNew_current, refToNew_aff_current, false);
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
			Mat88f H = ss.H;
			Mat88f H2 = ss.SHpp.cast<float>();
			Mat88f QH = ss.QH.cast<float>();
			// Pose damping, add diag(Hpp) * lambda to Hpp
			for(int i=0;i<8;i++) H(i,i) *= (1+lambda);
			for(int i=0;i<8;i++) H2(i,i) *= (1+lambda);
			for(int i=0;i<8;i++) QH(i,i) += ss.QHpp(i,i) * lambda;
			// point damping.

			H -= ss.Hsc*(1/(1+lambda));
			H2 -= ss.SHsc.cast<float>()*(1/(1+lambda));
			Mat88f QHsc = (ss.QHpp - ss.QH).cast<float>();
			QH += QHsc*(lambda/(1+lambda));
			Vec8f b = ss.b - ss.bsc*(1/(1+lambda));
			Vec8f b2 = ss.Sbp.cast<float>() - ss.Sbsc.cast<float>()*(1/(1+lambda));
			Vec8f Qbsc = (ss.Qbpp - ss.Qb).cast<float>();
			Vec8f Qb = ss.Qb.cast<float>() + Qbsc*(lambda/(1+lambda));

			H = wM * H * wM * (0.01f/(w[lvl]*h[lvl]));
			b = wM * b * (0.01f/(w[lvl]*h[lvl]));

			H2 = wM * H2 * wM * (0.01f/(w[lvl]*h[lvl]));
			b2 = wM * b2 * (0.01f/(w[lvl]*h[lvl]));

			QH = wM * QH * wM * (0.01f/(w[lvl]*h[lvl]));
			Qb = wM * Qb * (0.01f/(w[lvl]*h[lvl]));

			if (!ss.H.isApprox(ss.QHpp.cast<float>(),0.00001)) {
				std::cerr << "\nQHpp:\n" << ss.QHpp.format(MatlabFmt);
				std::cerr << "\nH:\n" << ss.H.format(MatlabFmt);
				std::cerr << "\nQHpp/H:\n" << (ss.QHpp.cast<float>().array() / ss.H.array()).format(MatlabFmt);
				abort();
			}


//			if (!ss.Hsc.isApprox(QHsc, 0.0001)) {
//				std::cerr << "\nQHpp:\n" << ss.QHpp.format(MatlabFmt);
//				std::cerr << "\nQH:\n" << ss.QH.format(MatlabFmt);
//				std::cerr << "\nQHsc:\n" << QHsc.format(MatlabFmt);
//				std::cerr << "\nHsc:\n" << ss.Hsc.format(MatlabFmt);
//				std::cerr << "\nQHsc/Hsc:\n" << (QHsc.array() / ss.Hsc.array()).format(MatlabFmt);
//				abort();
//			}

			Vec8f incSc;
			Vec8f incSc2;
			Vec8f incQr;
			if(fixAffine)
			{
				if (!H.isApprox(H2)) {
					std::cerr << "\nH2:\n" << H2.format(MatlabFmt);
					std::cerr << "\nH:\n" << H.format(MatlabFmt);
					std::cerr << "\nH2/H:\n" << (H2.array() / H.array()).format(MatlabFmt);
				}
				if (!b.isApprox(b2)) {
					std::cerr << "\nb2:\n" << b2.format(MatlabFmt);
					std::cerr << "\nb:\n" << b.format(MatlabFmt);
					std::cerr << "\nb2/b:\n" << (b2.array() / b.array()).format(MatlabFmt);
				}

				if (!H.isApprox(QH)) {
					std::cerr << "\nQH:\n" << QH.format(MatlabFmt);
					std::cerr << "\nH:\n" << H.format(MatlabFmt);
					std::cerr << "\nQH/H:\n" << (QH.array() / H.array()).format(MatlabFmt);
				}

				if (!b.isApprox(Qb)) {
					std::cerr << "\nQb:\n" << Qb.format(MatlabFmt);
					std::cerr << "\nb:\n" << b.format(MatlabFmt);
					std::cerr << "\nQb/b:\n" << (Qb.array() / b.array()).format(MatlabFmt);
				}

				if (!ss.b.isApprox(ss.Sbp.cast<float>())) {
					std::cerr << "\nSbp, SC:\n" << ss.Sbp.format(MatlabFmt);
					std::cerr << "\nb, SC:\n" << ss.b.format(MatlabFmt);
					std::cerr << "\nb Sbp/b:\n" << (ss.Sbp.cast<float>().array() / ss.b.array()).format(MatlabFmt);
				}

				if (!ss.bsc.isApprox(ss.Sbsc.cast<float>())) {
					std::cerr << "\nSbl, SC:\n" << ss.Sbsc.format(MatlabFmt);
					std::cerr << "\nbsc, SC:\n" << ss.bsc.format(MatlabFmt);
					std::cerr << "\nb Sbl/bsc:\n" << (ss.Sbsc.cast<float>().array() / ss.bsc.array()).format(MatlabFmt);
				}

				incSc.head<6>() = - (wM.toDenseMatrix().topLeftCorner<6,6>() * (H.topLeftCorner<6,6>().ldlt().solve(b.head<6>())));
				incSc.tail<2>().setZero();

				incSc2.head<6>() = - (wM.toDenseMatrix().topLeftCorner<6,6>() * (H2.topLeftCorner<6,6>().ldlt().solve(b2.head<6>())));
				incSc2.tail<2>().setZero();

				incQr.head<6>() = - (wM.toDenseMatrix().topLeftCorner<6,6>() * (QH.topLeftCorner<6,6>().ldlt().solve(Qb.head<6>())));
				incQr.tail<2>().setZero();

				if (!incSc2.isApprox(incSc, 0.0001)) {
					std::cerr << "\nincSc2:\n" << incSc2.format(MatlabFmt);
					std::cerr << "\nincSc:\n" << incSc.format(MatlabFmt);
					std::cerr << "\ninc SC2/SC:\n" << (incSc2.array() / incSc.array()).format(MatlabFmt);
				}
//				if (!incQr.isApprox(incSc, 0.0001)) {
//					std::cerr << "\nincQr:\n" << incQr.format(MatlabFmt);
//					std::cerr << "\nincSc:\n" << incSc.format(MatlabFmt);
//					std::cerr << "\ninc QR/SC:\n" << (incQr.array() / incSc.array()).format(MatlabFmt);
//				}
//				if (!incQr.isApprox(incSc, 0.2)) {
//					abort();
//				}
			} else {
				incSc = - (wM * (H.ldlt().solve(b)));	//=-H^-1 * b.
			}

			Vec8f inc(incQr);

			SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
			AffLight refToNew_aff_new = refToNew_aff_current;
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];
			doStep(lvl, lambda, inc);

			StepState<QR_PRECISION> ss_new;

			Vec3f resNew = calcResAndGS(lvl, ss_new, refToNew_new, refToNew_aff_new, false);
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
				ss = ss_new;
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
		int lvl, StepState<QR_PRECISION> &ss,
		const SE3 &refToNew, AffLight refToNew_aff,
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

	ss.reset();

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
	float alphaEnergy = alphaW*(refToNew.translation().squaredNorm());
	if(alphaEnergy > alphaK) {
		alphaEnergy = alphaK;
		//std::cerr << "alphaEnergy > alphaK\n";
	}

	acc9SC.initialize();

	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new) {
			continue;
		}

		point->lastHessian_new = JbBuffer_new[i][9];

		float HllDampingAdd = EPS;
		float alphaWPnt = 0;

		// Does this hold ID values around 1 until there is sufficient translation to estimate them...
		if(alphaEnergy == alphaK) {
			alphaWPnt += couplingWeight*(point->idepth_new - point->iR);
			HllDampingAdd += couplingWeight;
		} else {
			alphaWPnt += alphaW*(point->idepth_new - 1);
			HllDampingAdd += alphaW;
		}

		pBlocksNew[i].qrMarginalise();
		pBlocksNew[i].applyJlDamping(HllDampingAdd);
		pBlocksNew[i].addPoseContribution(ss, alphaWPnt);
		pBlocksNew[i].addPoseScContribution(ss, alphaWPnt);

		JbBuffer_new[i][8] += alphaWPnt;
		JbBuffer_new[i][9] = 1/(HllDampingAdd + JbBuffer_new[i][9]);
		acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
				(float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],
				(float)JbBuffer_new[i][8],(float)JbBuffer_new[i][9]);
	}
	acc9SC.finish();

	//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
	ss.H = acc9.H.topLeftCorner<8,8>();// / acc9.num;
	ss.b = acc9.H.topRightCorner<8,1>();// / acc9.num;
	ss.Hsc = acc9SC.H.topLeftCorner<8,8>();// / acc9.num;
	ss.bsc = acc9SC.H.topRightCorner<8,1>();// / acc9.num;

	if (alphaEnergy != alphaK) {
		ss.H(0, 0) += alphaW * npts;
		ss.H(1, 1) += alphaW * npts;
		ss.H(2, 2) += alphaW * npts;
		ss.SHpp(0, 0) += alphaW * npts;
		ss.SHpp(1, 1) += alphaW * npts;
		ss.SHpp(2, 2) += alphaW * npts;

		ss.QH(0, 0) += alphaW * npts;
		ss.QH(1, 1) += alphaW * npts;
		ss.QH(2, 2) += alphaW * npts;
		ss.QHpp(0, 0) += alphaW * npts;
		ss.QHpp(1, 1) += alphaW * npts;
		ss.QHpp(2, 2) += alphaW * npts;

		Vec3f tlog = refToNew.log().head<3>().cast<float>();
		ss.b[0] += tlog[0] * alphaW * npts;
		ss.b[1] += tlog[1] * alphaW * npts;
		ss.b[2] += tlog[2] * alphaW * npts;
		ss.Sbp[0] += tlog[0] * alphaW * npts;
		ss.Sbp[1] += tlog[1] * alphaW * npts;
		ss.Sbp[2] += tlog[2] * alphaW * npts;

		ss.Qb[0] += tlog[0] * alphaW * npts;
		ss.Qb[1] += tlog[1] * alphaW * npts;
		ss.Qb[2] += tlog[2] * alphaW * npts;
		ss.Qbpp[0] += tlog[0] * alphaW * npts;
		ss.Qbpp[1] += tlog[1] * alphaW * npts;
		ss.Qbpp[2] += tlog[2] * alphaW * npts;
	}

	E.finish();

	return Vec3f(E.A, alphaEnergy*npts, E.num);
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


		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		float stepSc = - b * JbBuffer[i][9] / (1+lambda);
		float stepSc2 = pBlocks[i].getScLandmarkIncFromPoseInc(inc) / (1+lambda);
		float stepQr = pBlocks[i].getLandmarkIncFromPoseInc(inc) / (1+lambda);

		assert (stepSc2 !=0 || stepSc == 0);
		float step = stepQr;

		if (abs(stepSc - step) > 0.001) {
			//if (stepSc == 0 && stepQr != 0) {
			std::cerr << "Depth step SC=" << stepSc << " new=" << step << "\n";
			abort();
		}


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
	std::swap<PointBlock<QR_PRECISION> *>(pBlocks, pBlocksNew);
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

