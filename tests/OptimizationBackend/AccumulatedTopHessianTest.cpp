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
	MatXX H;
	VecX b;

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
	acc->stitchDoubleMT(0, H, b, adHost, adTarget, false);

	EXPECT_EQ(H.norm(), 0);
	EXPECT_EQ(b.norm(), 0);
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

	RawResidualJacobian *J = r.J;
	//J->resF << -3.40619, -1.06463, -0.774745, 3.60521, 7.50899, -0.775642, 5.21856, 8.58308;
	J->resF << 1, 1, 1, 1, 1, 1, 1, 1;
	J->Jpdxi[0] << 133.221, 0, 19.6512, -6.50697, 468.632, 44.1125;
	J->Jpdxi[1] << 0, 132.827, 12.7751, -461.526, 6.48773, -67.455;
	J->Jpdc[0] << -0.508134, -0.000141843, -1.03484, -0.503665, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	J->Jpdc[1] << -0.0121198, -4.82333, 0.090069, -1.13272, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	J->Jpdd << -0.704304, -74.2505;
	J->JIdx[0] << -0.736503, 8.31152, 11.5026, 2.71132, 24.6621, 6.55246, 16.7477, 18.9202;
	J->JIdx[1] << 7.0005, -0.129556, 24.5396, -2.11575, 6.98762, 15.0123, -3.53676, -4.6037;
	J->JabF[0] << 44.9032, 35.7146, 53.4572, 27.9331, 56.1396, 108.597, 28.0283, 38.2034;
	J->JabF[1] << 0.990189, 0.974635, 0.846627, 0.994076, 0.872015, 0.968303, 0.89612, 0.857008;
	J->JIdx2 << 1498.9, 394.662, 394.662, 963.593;
	J->JabJIdx << 4242.72, 3310.01, 78.8782, 38.9933;
	J->Jab2 << 24119.8, 364.264, 364.264, 6.87087;

	p.residualsAll.push_back(&r);

	acc->addPoint<0>(&p, adHTdeltaF, &cDelta);  // <0> - Active point.

	acc->stitchDoubleMT(0, H, b, adHost, adTarget, false);

	// Given resFs are all 1
	float JIdxSumX = J->JIdx[0].sum();
	float JIdxSumY = J->JIdx[1].sum();

	VecX bExpected = VecX::Zero(30);
	bExpected(0) = JIdxSumX * J->Jpdc[0](0) + JIdxSumY * J->Jpdc[1](0);
	bExpected(1) = JIdxSumX * J->Jpdc[0](1) + JIdxSumY * J->Jpdc[1](1);
	bExpected(2) = JIdxSumX * J->Jpdc[0](2) + JIdxSumY * J->Jpdc[1](2);
	bExpected(3) = JIdxSumX * J->Jpdc[0](3) + JIdxSumY * J->Jpdc[1](3);

	bExpected(14) = bExpected(22) = 0;
	bExpected(15) = bExpected(23) = 0;
	bExpected(16) = bExpected(24) = 0;
	bExpected(17) = bExpected(25) = 0;
	bExpected(18) = bExpected(26) = 0;
	bExpected(19) = bExpected(27) = 0;

	bExpected(20) = bExpected(28) = 0;
	bExpected(21) = bExpected(29) = 0;

	ASSERT_PRED2(MatEq, b, bExpected);
	ASSERT_PRED2(MatEq, H, MatXX::Zero(30, 30));
}

