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

using namespace dso;

class AccumulatedTopHessianTest: public ::testing::Test {
protected:
	AccumulatedTopHessianSSE *acc;
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

	void TearDown() override {
		if (adHost)
			delete adHost;
		if (adTarget)
			delete adTarget;
	}
};

bool MatEq(const MatXX &lhs, const MatXX &rhs) {
	return lhs.isApprox(rhs, 1e-4);
}

TEST_F(AccumulatedTopHessianTest, zero) {
	setNFrames(3);
	acc->stitchDoubleMT(0, HResult, bResult, adHost, adTarget, false);

	EXPECT_EQ(HResult.norm(), 0);
	EXPECT_EQ(bResult.norm(), 0);
}

TEST_F(AccumulatedTopHessianTest, twoFrames) {
	setNFrames(2);

	Mat18f *adHTdeltaF = 0;

	PointHessianBase ph = PointHessianBase();
	EFPoint p = EFPoint(&ph, 0);

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

	p.residualsAll.push_back(&r);

	acc->addPoint<0>(&p, adHTdeltaF, &cDelta);  // <0> - Active point.

	acc->stitchDoubleMT(0, HResult, bResult, adHost, adTarget, false);

	MatXX J = MatXX::Zero(8, 30);

	MatXX dIdP(8, 2);
	dIdP.leftCols(1) = rJ->JIdx[0].cast<double>();
	dIdP.rightCols(1) = rJ->JIdx[1].cast<double>();

	MatXX dPdC(2, 14);
	dPdC.topRows(1) = rJ->Jpdc[0].transpose().cast<double>();
	dPdC.bottomRows(1) = rJ->Jpdc[1].transpose().cast<double>();
	J.block<8,14>(0, 0) = dIdP * dPdC;

	MatXX dPdXi(2,6);
	dPdXi.topRows(1) = rJ->Jpdxi[0].transpose().cast<double>();
	dPdXi.bottomRows(1) = rJ->Jpdxi[1].transpose().cast<double>();
	J.block<8,6>(0, 14) = dIdP * dPdXi;
	J.block<8,6>(0, 22) = dIdP * dPdXi;

	MatXX dPdAb(8,2);
	dPdAb.leftCols(1) = rJ->JabF[0].cast<double>();
	dPdAb.rightCols(1) = rJ->JabF[1].cast<double>();
	J.block<8,2>(0, 20) = dPdAb;
	J.block<8,2>(0, 28) = dPdAb;

	std::cout << "J:\n" << J << "\n\n";

	VecX bExpected = J.transpose() * rJ->resF.cast<double>();
	ASSERT_PRED2(MatEq, bResult, bExpected);

	MatXX HExpected = J.transpose() * J;
	ASSERT_PRED2(MatEq, HResult, HExpected);
}

