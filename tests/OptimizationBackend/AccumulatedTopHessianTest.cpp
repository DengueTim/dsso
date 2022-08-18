/*
 * AccumulatedTopHessianTest.cpp
 *
 *  Created on: 9 Aug 2022
 *      Author: tp
 */

#include <gtest/gtest.h>

#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/HessianBlocks.h"

#include "TestUtils.h"

using namespace dso;

class AccumulatedTopHessianTest: public ::testing::Test {
protected:
	AccumulatedTopHessianSSE *acc;
	Mat18f *adHTdeltaF = 0;
	VecC cDelta;
	Mat88 *adHost;
	Mat88 *adTarget;
	MatXX HResult;
	VecX bResult;

	void SetUp() override {
		acc = new AccumulatedTopHessianSSE();
		cDelta.Zero();
		adHost = 0;
		adTarget = 0;
	}

	void setNFrames(int n) {
		int n2 = n * n;
		acc->setZero(n);
		adHost = new Mat88[n2];
		adTarget = new Mat88[n2];

		for (int i = 0; i < n2; i++) {
			adHost[i].setIdentity();
			adTarget[i].setIdentity();
		}
	}

	void addResToJ(EFResidual &r, MatXX &J) {
		RawResidualJacobian *rJ = r.J;
		int hIdx = r.hostIDX;
		int tIdx = r.targetIDX;

		int cols = J.cols();
		int oldRows = J.rows();
		MatXX JTmp = J;
		J.resize(oldRows + 8, cols);
		J.topRows(oldRows) = JTmp;
		J.bottomRows(8) = MatXX::Zero(8, cols);

		MatXX dIdP(8, 2);
		dIdP.leftCols(1) = rJ->JIdx[0].cast<double>();
		dIdP.rightCols(1) = rJ->JIdx[1].cast<double>();

		MatXX dPdC(2, 14);
		dPdC.topRows(1) = rJ->Jpdc[0].transpose().cast<double>();
		dPdC.bottomRows(1) = rJ->Jpdc[1].transpose().cast<double>();
		J.block<8, 14>(oldRows, 0) = dIdP * dPdC;

		if (hIdx != tIdx) {
			hIdx = 14 + hIdx * 8;
			tIdx = 14 + tIdx * 8;
			MatXX dPdXi(2, 6);
			dPdXi.topRows(1) = rJ->Jpdxi[0].transpose().cast<double>();
			dPdXi.bottomRows(1) = rJ->Jpdxi[1].transpose().cast<double>();
			J.block<8, 6>(oldRows, hIdx) = dIdP * dPdXi;
			J.block<8, 6>(oldRows, tIdx) = dIdP * dPdXi;

			MatXX dPdAb(8, 2);
			dPdAb.leftCols(1) = rJ->JabF[0].cast<double>();
			dPdAb.rightCols(1) = rJ->JabF[1].cast<double>();
			J.block<8, 2>(oldRows, hIdx + 6) = dPdAb;
			J.block<8, 2>(oldRows, tIdx + 6) = dPdAb;
		}
	}

	void TearDown() override {
		if (adHost)
			delete adHost;
		if (adTarget)
			delete adTarget;
	}
};

TEST_F(AccumulatedTopHessianTest, zero) {
	setNFrames(3);
	acc->stitchDoubleMT(0, HResult, bResult, adHost, adTarget, false);

	EXPECT_EQ(HResult.norm(), 0);
	EXPECT_EQ(bResult.norm(), 0);
}

TEST_F(AccumulatedTopHessianTest, oneLeftLeftResidual) {
	setNFrames(2);

	EFResidual r = EFResidual(0, 0, 0, 0);
	r.isActive = true;
	r.hostIDX = 0;
	r.targetIDX = 1;

	RawResidualJacobian *rJ = r.J;
	//J->resF << -3.40619, -1.06463, -0.774745, 3.60521, 7.50899, -0.775642, 5.21856, 8.58308;
	rJ->resF << 1, 1, 1, 1, 1, 1, 1, 1;
	rJ->Jpdxi[0] << 133.221, 0, 19.6512, -6.50697, 468.632, 44.1125;
	rJ->Jpdxi[1] << 0, 132.827, 12.7751, -461.526, 6.48773, -67.455;
	rJ->Jpdc[0] << -0.508134, -0.000141843, -1.03484, -0.503665, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	rJ->Jpdc[1] << -0.0121198, -4.82333, 0.090069, -1.13272, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	rJ->Jpdd << -0.704304, -74.2505;
	rJ->JIdx[0] << -0.736503, 8.31152, 11.5026, 2.71132, 24.6621, 6.55246, 16.7477, 18.9202;
	rJ->JIdx[1] << 7.0005, -0.129556, 24.5396, -2.11575, 6.98762, 15.0123, -3.53676, -4.6037;
	rJ->JabF[0] << 44.9032, 35.7146, 53.4572, 27.9331, 56.1396, 108.597, 28.0283, 38.2034;
	rJ->JabF[1] << 0.990189, 0.974635, 0.846627, 0.994076, 0.872015, 0.968303, 0.89612, 0.857008;
	rJ->JIdx2 << 1498.9, 394.662, 394.662, 963.593;
	rJ->JabJIdx << 4242.72, 3310.01, 78.8782, 38.9933;
	rJ->Jab2 << 24119.8, 364.264, 364.264, 6.87087;

	PointHessianBase ph = PointHessianBase();
	EFPoint p = EFPoint(&ph, 0);
	p.residualsAll.push_back(&r);

	acc->addPoint<0>(&p, adHTdeltaF, &cDelta);  // <0> - Active point.

	acc->stitchDoubleMT(0, HResult, bResult, adHost, adTarget, false);

	MatXX J = MatXX::Zero(0, 30);
	addResToJ(r, J);

	VecX bExpected = J.transpose() * rJ->resF.cast<double>();
	ASSERT_PRED2(MatEq, bResult, bExpected);

	MatXX HExpected = J.transpose() * J;
	ASSERT_PRED2(MatEq, HResult, HExpected);
}

TEST_F(AccumulatedTopHessianTest, oneLeftRightResidual) {
	setNFrames(2);

	EFResidual r = EFResidual(0, 0, 0, 0);
	r.isActive = true;
	r.hostIDX = 0;
	r.targetIDX = 0;

	RawResidualJacobian *rJ = r.J;
//	rJ->resF << 9.92223, 8.7972, 10.9996, 6.33175, 8.08464, -4.79276, 8.08829, -7.77803;
	rJ->resF << 1, 1, 1, 1, 1, 1, 1, 1;
	rJ->Jpdxi[0] << 0, 0, 0, 0, 0, 0;
	rJ->Jpdxi[1] << 0, 0, 0, 0, 0, 0;
	rJ->Jpdc[0] << 16.0111, 0.123961, -50.2193, 0.335251, -30.5502, 0, 50, 0, 1221.26, 0, 746.192, 108.528, 628.362, -177.623;
	rJ->Jpdc[1] << -0.03008, -18.6761, 0.0943469, -50.5094, 0, 19.4103, 0, 50, 0, 1217.54, -472.654, -524.898, -108.197, -278.712;
	rJ->Jpdd << -50.56, 0.224778;
	rJ->JIdx[0] << 5.79476, 6.96991, 1.51123, 4.52278, 3.62904, -3.04868, 1.90134, -2.95721;
	rJ->JIdx[1] << 4.79284, 5.05582, 1.76634, 6.03098, 1.00303, -3.40171, 2.51659, -4.7374;
	rJ->JabF[0] << 0, 0, 0, 0, 0, 0, 0, 0;
	rJ->JabF[1] << 0, 0, 0, 0, 0, 0, 0, 0;
	rJ->JIdx2 << 139.723, 125.763, 125.763, 129.379;
	rJ->JabJIdx << 0, 0, 0, 0;
	rJ->Jab2 << 0, 0, 0, 0;

	PointHessianBase ph = PointHessianBase();
	EFPoint p = EFPoint(&ph, 0);
	p.residualsAll.push_back(&r);

	acc->addPoint<0>(&p, adHTdeltaF, &cDelta);  // <0> - Active point.

	acc->stitchDoubleMT(0, HResult, bResult, adHost, adTarget, false);

	MatXX J = MatXX::Zero(0, 30);
	addResToJ(r, J);

	VecX bExpected = J.transpose() * rJ->resF.cast<double>();
	ASSERT_PRED2(MatEq, bResult, bExpected);

	MatXX HExpected = J.transpose() * J;
	ASSERT_PRED2(MatEq, HResult, HExpected);
}

TEST_F(AccumulatedTopHessianTest, leftLeftAndLeftRightResidual) {
	setNFrames(2);

	EFResidual llr = EFResidual(0, 0, 0, 0);
	llr.isActive = true;
	llr.hostIDX = 0;
	llr.targetIDX = 1;

	RawResidualJacobian *llrJ = llr.J;
	//J->resF << -3.40619, -1.06463, -0.774745, 3.60521, 7.50899, -0.775642, 5.21856, 8.58308;
	llrJ->resF << 1, 1, 1, 1, 1, 1, 1, 1;
	llrJ->Jpdxi[0] << 133.221, 0, 19.6512, -6.50697, 468.632, 44.1125;
	llrJ->Jpdxi[1] << 0, 132.827, 12.7751, -461.526, 6.48773, -67.455;
	llrJ->Jpdc[0] << -0.508134, -0.000141843, -1.03484, -0.503665, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	llrJ->Jpdc[1] << -0.0121198, -4.82333, 0.090069, -1.13272, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	llrJ->Jpdd << -0.704304, -74.2505;
	llrJ->JIdx[0] << -0.736503, 8.31152, 11.5026, 2.71132, 24.6621, 6.55246, 16.7477, 18.9202;
	llrJ->JIdx[1] << 7.0005, -0.129556, 24.5396, -2.11575, 6.98762, 15.0123, -3.53676, -4.6037;
	llrJ->JabF[0] << 44.9032, 35.7146, 53.4572, 27.9331, 56.1396, 108.597, 28.0283, 38.2034;
	llrJ->JabF[1] << 0.990189, 0.974635, 0.846627, 0.994076, 0.872015, 0.968303, 0.89612, 0.857008;
	llrJ->JIdx2 << 1498.9, 394.662, 394.662, 963.593;
	llrJ->JabJIdx << 4242.72, 3310.01, 78.8782, 38.9933;
	llrJ->Jab2 << 24119.8, 364.264, 364.264, 6.87087;

	EFResidual lrr = EFResidual(0, 0, 0, 0);
	lrr.isActive = true;
	lrr.hostIDX = 0;
	lrr.targetIDX = 0;

	RawResidualJacobian *lrrJ = lrr.J;
//	rJ->resF << 9.92223, 8.7972, 10.9996, 6.33175, 8.08464, -4.79276, 8.08829, -7.77803;
	lrrJ->resF << 1, 1, 1, 1, 1, 1, 1, 1;
	lrrJ->Jpdxi[0] << 0, 0, 0, 0, 0, 0;
	lrrJ->Jpdxi[1] << 0, 0, 0, 0, 0, 0;
	lrrJ->Jpdc[0] << 16.0111, 0.123961, -50.2193, 0.335251, -30.5502, 0, 50, 0, 1221.26, 0, 746.192, 108.528, 628.362, -177.623;
	lrrJ->Jpdc[1] << -0.03008, -18.6761, 0.0943469, -50.5094, 0, 19.4103, 0, 50, 0, 1217.54, -472.654, -524.898, -108.197, -278.712;
	lrrJ->Jpdd << -50.56, 0.224778;
	lrrJ->JIdx[0] << 5.79476, 6.96991, 1.51123, 4.52278, 3.62904, -3.04868, 1.90134, -2.95721;
	lrrJ->JIdx[1] << 4.79284, 5.05582, 1.76634, 6.03098, 1.00303, -3.40171, 2.51659, -4.7374;
	lrrJ->JabF[0] << 0, 0, 0, 0, 0, 0, 0, 0;
	lrrJ->JabF[1] << 0, 0, 0, 0, 0, 0, 0, 0;
	lrrJ->JIdx2 << 139.723, 125.763, 125.763, 129.379;
	lrrJ->JabJIdx << 0, 0, 0, 0;
	lrrJ->Jab2 << 0, 0, 0, 0;

	PointHessianBase ph = PointHessianBase();
	EFPoint p = EFPoint(&ph, 0);
	p.residualsAll.push_back(&llr);
	p.residualsAll.push_back(&lrr);

	acc->addPoint<0>(&p, adHTdeltaF, &cDelta);  // <0> - Active point.

	acc->stitchDoubleMT(0, HResult, bResult, adHost, adTarget, false);

	MatXX J = MatXX::Zero(0, 30);
	addResToJ(llr, J);
	addResToJ(lrr, J);
	//std::cout << "J:\n" << J << "\n";

	VecXf r(16);
	r << llrJ->resF, lrrJ->resF;

	VecX bExpected = J.transpose() * r.cast<double>();
	//ASSERT_PRED2(MatEq, bResult, bExpected);

	MatXX HExpected = J.transpose() * J;
	ASSERT_PRED2(MatEq, HResult, HExpected);
}

