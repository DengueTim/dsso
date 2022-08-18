/*
 * AccumulatedTopHessianTest.cpp
 *
 *  Created on: 9 Aug 2022
 *      Author: tp
 */

#include <gtest/gtest.h>

#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/HessianBlocks.h"

#include "TestUtils.h"

using namespace dso;

class AccumulatedSCHessianTest: public ::testing::Test {
protected:
	AccumulatedSCHessianSSE *acc;
	MatXX HResult;
	VecX bResult;

	void SetUp() override {
		acc = new AccumulatedSCHessianSSE();
	}

	void setNFrames(int n) {
		int n2 = n * n;
		acc->setZero(n);
	}

	// Schur Complement blocks..
	// Haa' = Haa - Hba.transpose() * Hbb.inverse() * Hba
	void __attribute__((optimize(0))) addPointToHsc(EFPoint &p, MatXX &Hab, VecX &Hbb) {
		int oldCols = Hab.cols();
		int rows = Hab.rows();
		MatXX HabTmp = Hab;
		Hab.resize(rows, oldCols + 1);
		Hab.leftCols(oldCols) = HabTmp;
		Hab.rightCols(1) = MatXX::Zero(rows, 1);

		//Hab.block<14,1>(0, oldCols) = (p.Hcd_accAF + p.Hcd_accLF).cast<double>(); // Why not this...
		Hab.block<8, 1>(0, oldCols) = (p.Hcd_accAF + p.Hcd_accLF).head(8).cast<double>();

		for (EFResidual *r : p.residualsAll) {
			RawResidualJacobian *rJ = r->J;
			int hIdx = r->hostIDX;
			int tIdx = r->targetIDX;
			if (hIdx != tIdx) {
				Hab.block<8, 1>(14 + hIdx * 8, oldCols) = r->JpJdAdH.cast<double>();
				Hab.block<8, 1>(14 + tIdx * 8, oldCols) = r->JpJdAdT.cast<double>();
			} else {
				Hab.block<6, 1>(8, oldCols) = r->JpJdAdT.head(6).cast<double>();
			}
		}

		VecX HbbTmp = Hbb;
		Hbb.resize(oldCols + 1, 1);
		Hbb.head(oldCols) = HbbTmp;
		Hbb.tail(1) << (1.0 / (p.Hdd_accAF + p.Hdd_accLF + p.priorF));
	}

	void TearDown() override {
	}
};

TEST_F(AccumulatedSCHessianTest, zero) {
	setNFrames(3);
	acc->stitchDoubleMT(0, HResult, bResult, false);

	EXPECT_EQ(HResult.norm(), 0);
	EXPECT_EQ(bResult.norm(), 0);
}

TEST_F(AccumulatedSCHessianTest, oneLeftLeftResidual) {
	setNFrames(2);

	EFResidual r = EFResidual(0, 0, 0, 0);
	r.isActive = true;
	r.hostIDX = 0;
	r.targetIDX = 1;
	r.JpJdF << 4.97099e+06, -2.98301e+07, -1.07979e+07, 1.48794e+08, -1.08763e+07, 9.85469e+07, -1.51849e+06, -9739.62;
	r.JpJdAdH << -2.48163e+06, 1.51443e+07, 4.71978e+06, -1.49976e+08, 5.51855e+06, -9.65638e+07, -1.51841e+07, -9.7391e+06;
	r.JpJdAdT << 2.48549e+06, -1.4915e+07, -5.39893e+06, 1.48794e+08, -1.08763e+07, 9.85469e+07, 1.51841e+07, 9.73962e+06;

	RawResidualJacobian *rJ = r.J;
//	rJ->resF << -8.62783, 8.89152, -7.27327, 10.3212, -1.45104, -5.30127, 5.6949, 7.37228;
	rJ->resF << 1, 1, 1, 1, 1, 1, 1, 1;
	rJ->Jpdxi[0] << 107.865, 0, 75.2097, -152.918, 681.633, 219.315;
	rJ->Jpdxi[1] << 0, 107.547, 51.4258, -561.857, 152.466, -318.853;
	rJ->Jpdc[0] << -0.352246, 0.716056, -2.45949, -1.8834, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	rJ->Jpdc[1] << 0.0528305, -3.69366, -0.0803077, -3.17016, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	rJ->Jpdd << -17.8332, -86.9848;
	rJ->JIdx[0] << -17.2931, -2.4871, -16.2502, 12.8386, -14.3551, -18.3865, 22.0021, 4.29439;
	rJ->JIdx[1] << 17.8779, 18.862, 19.1852, 11.0276, 34.0093, 19.343, 17.9203, 18.846;
	rJ->JabF[0] << 79.8932, 80.2839, 64.1482, 82.1432, 119.028, 77.7925, 153.177, 220.734;
	rJ->JabF[1] << 0.784726, 0.667254, 0.627502, 0.62383, 0.702476, 0.79612, 0.808131, 0.895142;
	rJ->JIdx2 << 1780.8, -894.897, -894.897, 3372.17;
	rJ->JabJIdx << -389.949, 17536.9, -20.515, 116.175;
	rJ->Jab2 << 116097, 674.682, 674.682, 4.42558;

	PointHessianBase ph = PointHessianBase();
	EFPoint p = EFPoint(&ph, 0);
	p.priorF = 2500;
	p.deltaF = 0;
	p.bdSumF = 10417.5;
	p.HdiF = 1.41385e-08;
	p.Hdd_accLF = 0;
	p.Hcd_accLF << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	p.bd_accLF = 0;
	p.Hdd_accAF = 7.12484e+07;
	p.Hcd_accAF << -3.99909e+06, 3.20848e+06, 5.71125e+06, 1.05225e+06, 4.12991e+06, -351317, -6.07674e+06, 969084, -1.2496e+07, 1.98644e+06, -7.77244e+06, 3.6995e+06, -7.91213e+07, -2.61692e+07;
	p.bd_accAF = 3755.8;
	p.residualsAll.push_back(&r);

	acc->addPoint(&p, false);

	acc->stitchDoubleMT(0, HResult, bResult, false);

	MatXX Hab = MatXX::Zero(30, 0);
	VecX Hbb = VecX::Zero(0);
	addPointToHsc(p, Hab, Hbb);

	MatXX HExpected = Hab * Hbb.asDiagonal() * Hab.transpose();
	ASSERT_PRED2(MatEq, HResult, HExpected);

	VecX bExpected = Hab * Hbb.asDiagonal() * p.bdSumF;
	ASSERT_PRED2(MatEq, bResult, bExpected);

}

TEST_F(AccumulatedSCHessianTest, oneLeftRightResidual) {
	setNFrames(2);

	EFResidual r = EFResidual(0, 0, 0, 0);
	r.isActive = true;
	r.hostIDX = 0;
	r.targetIDX = 0;
	r.JpJdF << -480370, 1.54192e+06, 590735, -1.92051e+06, -505254, -242908, -0, -0;
	r.JpJdAdH << 241705, -765600, -307813, 1.92063e+06, 580680, 63863.8, 0, 0;
	r.JpJdAdT << -240185, 770962, 295367, -1.92051e+06, -505254, -242908, 0, 0;

	RawResidualJacobian *rJ = r.J;
//	rJ->resF << -18.8107, -11.7317, -13.735, -7.90389, -11.5988, -15.4545, -11.5062, -15.5353;
	rJ->resF << 1, 1, 1, 1, 1, 1, 1, 1;
	rJ->Jpdxi[0] << 0, 0, 0, 0, 0, 0;
	rJ->Jpdxi[1] << 0, 0, 0, 0, 0, 0;
	rJ->Jpdc[0] << -4.45125, 0.0764789, -49.6303, -0.18808, -1.18718, 0, 50, 0, 422.22, 0, 10.025, -4.24155, 457.706, 178.639;
	rJ->Jpdc[1] << 0.00860962, 20.0585, 0.095995, -49.3288, 0, -19.5256, 0, 50, 0, 421.159, 164.468, -525.884, 4.23089, -10.8342;
	rJ->Jpdd << -52.2923, -0.262607;
	rJ->JIdx[0] << 0.00456108, -0.240531, -0.792925, 0.267926, -2.26187, -1.19662, 0.812185, -3.76767;
	rJ->JIdx[1] << 6.86107, 17.8285, 12.9863, 26.9443, 18.4175, 11.3203, 17.1141, 8.46049;
	rJ->JabF[0] << 0, 0, 0, 0, 0, 0, 0, 0;
	rJ->JabF[1] << 0, 0, 0, 0, 0, 0, 0, 0;
	rJ->JIdx2 << 22.1613, -80.5158, -80.5158, 2091.4;
	rJ->JabJIdx << 0, 0, 0, 0;
	rJ->Jab2 << 0, 0, 0, 0;

	PointHessianBase ph = PointHessianBase();
	EFPoint p = EFPoint(&ph, 0);
	p.priorF = 0;
	p.deltaF = 0;
	p.bdSumF = 0;
	p.HdiF = 0;
	p.Hdd_accLF = 0;
	p.Hcd_accLF << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	p.bd_accLF = 0;
	p.Hdd_accAF = 58532.8;
	p.Hcd_accAF << 5095.81, 73350.2, 56817, -180386, 1350.68, -71486.1, -56886.2, 183057, -480370, 1.54192e+06, 590735, -1.92051e+06, -505254, -242908;
	p.bd_accAF = -5133.02;
	p.residualsAll.push_back(&r);

	acc->addPoint(&p, false);

	acc->stitchDoubleMT(0, HResult, bResult, false);

	MatXX Hab = MatXX::Zero(30, 0);
	VecX Hbb = VecX::Zero(0);
	addPointToHsc(p, Hab, Hbb);

	MatXX HExpected = Hab * Hbb.asDiagonal() * Hab.transpose();
	ASSERT_PRED2(MatEq, HResult, HExpected);

	VecX bExpected = Hab * Hbb.asDiagonal() * p.bdSumF;
	ASSERT_PRED2(MatEq, bResult, bExpected);
}

TEST_F(AccumulatedSCHessianTest, leftLeftAndLeftRightResidual) {
	setNFrames(2);

	EFResidual lrr = EFResidual(0, 0, 0, 0);
	lrr.isActive = true;
	lrr.hostIDX = 0;
	lrr.targetIDX = 0;
	lrr.JpJdF << -480370, 1.54192e+06, 590735, -1.92051e+06, -505254, -242908, -0, -0;
	lrr.JpJdAdH << 241705, -765600, -307813, 1.92063e+06, 580680, 63863.8, 0, 0;
	lrr.JpJdAdT << -240185, 770962, 295367, -1.92051e+06, -505254, -242908, 0, 0;

	RawResidualJacobian *lrrJ = lrr.J;
	//	rJ->resF << -18.8107, -11.7317, -13.735, -7.90389, -11.5988, -15.4545, -11.5062, -15.5353;
	lrrJ->resF << 1, 1, 1, 1, 1, 1, 1, 1;
	lrrJ->Jpdxi[0] << 0, 0, 0, 0, 0, 0;
	lrrJ->Jpdxi[1] << 0, 0, 0, 0, 0, 0;
	lrrJ->Jpdc[0] << -4.45125, 0.0764789, -49.6303, -0.18808, -1.18718, 0, 50, 0, 422.22, 0, 10.025, -4.24155, 457.706, 178.639;
	lrrJ->Jpdc[1] << 0.00860962, 20.0585, 0.095995, -49.3288, 0, -19.5256, 0, 50, 0, 421.159, 164.468, -525.884, 4.23089, -10.8342;
	lrrJ->Jpdd << -52.2923, -0.262607;
	lrrJ->JIdx[0] << 0.00456108, -0.240531, -0.792925, 0.267926, -2.26187, -1.19662, 0.812185, -3.76767;
	lrrJ->JIdx[1] << 6.86107, 17.8285, 12.9863, 26.9443, 18.4175, 11.3203, 17.1141, 8.46049;
	lrrJ->JabF[0] << 0, 0, 0, 0, 0, 0, 0, 0;
	lrrJ->JabF[1] << 0, 0, 0, 0, 0, 0, 0, 0;
	lrrJ->JIdx2 << 22.1613, -80.5158, -80.5158, 2091.4;
	lrrJ->JabJIdx << 0, 0, 0, 0;
	lrrJ->Jab2 << 0, 0, 0, 0;

	EFResidual llr = EFResidual(0, 0, 0, 0);
	llr.isActive = true;
	llr.hostIDX = 0;
	llr.targetIDX = 1;
	llr.JpJdF << -280783, -3.26764e+06, -1.32053e+06, 5.67216e+07, -2.54746e+06, -5.75699e+06, -338404, -3681.05;
	llr.JpJdAdH << 168296, 1.63255e+06, 656888, -5.78608e+07, 3.4568e+06, 6.23294e+06, -3.38398e+06, -3.68099e+06;
	llr.JpJdAdT << -140392, -1.63382e+06, -660266, 5.67216e+07, -2.54746e+06, -5.75699e+06, 3.38398e+06, 3.68105e+06;

	RawResidualJacobian *llrJ = llr.J;
	//rJ->resF << 4.68968, 9.57117, 8.2565, 9.99534, 8.69053, 8.9224, 9.2604, 9.45533;
	llrJ->resF << 1, 1, 1, 1, 1, 1, 1, 1;
	llrJ->Jpdxi[0] << 30.8081, 0, -2.55785, 15.6578, 461.731, 188.59;
	llrJ->Jpdxi[1] << 0, 30.7235, 12.6353, -534.658, -15.6148, 37.9684;
	llrJ->Jpdc[0] << -0.117362, -0.341594, -1.14896, 0.840933, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	llrJ->Jpdc[1] << -0.0728401, 0.233756, -0.87281, -1.19696, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	llrJ->Jpdd << -35.5237, -30.8936;
	llrJ->JIdx[0] << 0.296003, 2.21457, 0.0884682, 2.39257, 1.66139, 2.02502, 2.85492, 1.24406;
	llrJ->JIdx[1] << 8.91267, 17.7865, 21.4215, 20.8002, 23.7126, 23.0477, 22.3909, 16.0425;
	llrJ->JabF[0] << 47.2914, 37.7051, 44.5779, 48.1901, 62.5143, 60.2125, 90.855, 125.5;
	llrJ->JabF[1] << 0.98489, 0.705555, 0.760794, 0.601223, 0.667422, 0.613587, 0.720281, 0.800947;
	llrJ->JIdx2 << 27.2833, 263.639, 263.639, 3139.52;
	llrJ->JabJIdx << 858.046, 9967.21, 8.76395, 109.075;
	llrJ->Jab2 << 39506, 380.696, 380.696, 4.39035;

	PointHessianBase ph = PointHessianBase();
	EFPoint p = EFPoint(&ph, 0);
	p.priorF = 0;
	p.deltaF = 0;
	p.bdSumF = 0;
	p.HdiF = 0;
	p.Hdd_accLF = 0;
	p.Hcd_accLF << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	p.bd_accLF = 0;
	p.Hdd_accAF = 58532.8;
	p.Hcd_accAF << 5095.81, 73350.2, 56817, -180386, 1350.68, -71486.1, -56886.2, 183057, -480370, 1.54192e+06, 590735, -1.92051e+06, -505254, -242908;
	p.bd_accAF = -5133.02;
	p.residualsAll.push_back(&lrr);
	p.residualsAll.push_back(&llr);

	acc->addPoint(&p, false);

	acc->stitchDoubleMT(0, HResult, bResult, false);

	MatXX Hab = MatXX::Zero(30, 0);
	VecX Hbb = VecX::Zero(0);
	addPointToHsc(p, Hab, Hbb);

	//std::cout << "Hab = [\n" << Hab << "]\nHbb = [\n" << Hbb << "]\n";

	MatXX HExpected = Hab * Hbb.asDiagonal() * Hab.transpose();
	ASSERT_PRED2(MatEq, HResult, HExpected);

	VecX bExpected = Hab * Hbb.asDiagonal() * p.bdSumF;
	ASSERT_PRED2(MatEq, bResult, bExpected);
}
