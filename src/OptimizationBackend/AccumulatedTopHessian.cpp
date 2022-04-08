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

#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <iostream>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

template<int mode>
void AccumulatedTopHessianSSE::addPoint(EFPoint *p, EnergyFunctional const *const ef, int tid) // 0 = active, 1 = linearized, 2=marginalize
		{

	assert(mode==0 || mode==1 || mode==2);

	VecCf dc = ef->cDeltaF;
	float dd = p->deltaF;

	float bd_acc = 0;
	float Hdd_acc = 0;
	VecCf Hcd_acc = VecCf::Zero();

	for (EFResidual *r : p->residualsAll) {
		bool leftToRight = r->hostIDX == r->targetIDX;

		if (mode == 0) {
			if (r->isLinearized || !r->isActive())
				continue;
		}
		if (mode == 1) {
			if (!r->isLinearized || !r->isActive())
				continue;
		}
		if (mode == 2) {
			if (!r->isActive())
				continue;
			assert(r->isLinearized);
		}

		RawResidualJacobian *rJ = r->J;
		int htIDX = r->hostIDX + r->targetIDX * nframes[tid];
		Mat18f dp = ef->adHTdeltaF[htIDX];

		const dso::Vec6f &jpdxiX = leftToRight ? rJ->Jpdc[0].segment(CIPARS, 6) : rJ->Jpdxi[0];
		const dso::Vec6f &jpdxiY = leftToRight ? rJ->Jpdc[1].segment(CIPARS, 6) : rJ->Jpdxi[1];

		VecNRf resApprox;
		if (mode == 0)
			resApprox = rJ->resF;
		if (mode == 2)
			resApprox = r->res_toZeroF;
		if (mode == 1) {
			// compute Jp*delta
			__m128 Jp_delta_x = _mm_set1_ps(jpdxiX.dot(dp.head<6>()) + rJ->Jpdc[0].dot(dc) + rJ->Jpdd[0] * dd);
			__m128 Jp_delta_y = _mm_set1_ps(jpdxiY.dot(dp.head<6>()) + rJ->Jpdc[1].dot(dc) + rJ->Jpdd[1] * dd);
			__m128 delta_a = _mm_set1_ps((float) (dp[6]));
			__m128 delta_b = _mm_set1_ps((float) (dp[7]));

			for (int i = 0; i < patternNum; i += 4) {
				// PATTERN: rtz = resF - [JI*Jp Ja]*delta.
				__m128 rtz = _mm_load_ps(((float*) &r->res_toZeroF) + i);
				rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*) (rJ->JIdx)) + i), Jp_delta_x));
				rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*) (rJ->JIdx + 1)) + i), Jp_delta_y));
				rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*) (rJ->JabF)) + i), delta_a));
				rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*) (rJ->JabF + 1)) + i), delta_b));
				_mm_store_ps(((float*) &resApprox) + i, rtz);
			}
		}

		// need to compute JI^T * r, and Jab^T * r. (both are 2-vectors).
		Vec2f JI_r(0, 0);
		Vec2f Jab_r(0, 0);
		float rr = 0;
		for (int i = 0; i < patternNum; i++) {
			JI_r[0] += resApprox[i] * rJ->JIdx[0][i];
			JI_r[1] += resApprox[i] * rJ->JIdx[1][i];
			Jab_r[0] += resApprox[i] * rJ->JabF[0][i];
			Jab_r[1] += resApprox[i] * rJ->JabF[1][i];
			rr += resApprox[i] * resApprox[i];
		}

		acc[tid][htIDX].update(rJ->Jpdc[0].data(), jpdxiX.data(), rJ->Jpdc[1].data(), jpdxiY.data(), rJ->JIdx2(0, 0),
				rJ->JIdx2(0, 1), rJ->JIdx2(1, 1));

		acc[tid][htIDX].updateBotRight(rJ->Jab2(0, 0), rJ->Jab2(0, 1), Jab_r[0], rJ->Jab2(1, 1), Jab_r[1], rr);

		acc[tid][htIDX].updateTopRight(rJ->Jpdc[0].data(), jpdxiX.data(), rJ->Jpdc[1].data(), jpdxiY.data(), rJ->JabJIdx(0, 0),
				rJ->JabJIdx(0, 1), rJ->JabJIdx(1, 0), rJ->JabJIdx(1, 1), JI_r[0], JI_r[1]);

		Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd;
		bd_acc += JI_r[0] * rJ->Jpdd[0] + JI_r[1] * rJ->Jpdd[1];
		Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd);
		Hcd_acc += rJ->Jpdc[0] * Ji2_Jpdd[0] + rJ->Jpdc[1] * Ji2_Jpdd[1];

		nres[tid]++;
	}

	if (mode == 0) {
		p->Hdd_accAF = Hdd_acc;
		p->bd_accAF = bd_acc;
		p->Hcd_accAF = Hcd_acc;
	}
	if (mode == 1 || mode == 2) {
		p->Hdd_accLF = Hdd_acc;
		p->bd_accLF = bd_acc;
		p->Hcd_accLF = Hcd_acc;
	}
	if (mode == 2) {
		p->Hcd_accAF.setZero();
		p->Hdd_accAF = 0;
		p->bd_accAF = 0;
	}

}
template void AccumulatedTopHessianSSE::addPoint<0>(EFPoint *p, EnergyFunctional const *const ef, int tid);
template void AccumulatedTopHessianSSE::addPoint<1>(EFPoint *p, EnergyFunctional const *const ef, int tid);
template void AccumulatedTopHessianSSE::addPoint<2>(EFPoint *p, EnergyFunctional const *const ef, int tid);

void AccumulatedTopHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional const *const EF, bool usePrior, bool useDelta) {
	H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
	b = VecX::Zero(nframes[0] * 8 + CPARS);

	stitchDoubleInternal(&H, &b, EF, usePrior, 0, nframes[0] * nframes[0], 0, -1);

	copyUpperToLowerDiagonal(&H);
}

void AccumulatedTopHessianSSE::copyUpperToLowerDiagonal(MatXX *H) {
	// make diagonal by copying over parts.
	for (int h = 0; h < nframes[0]; h++) {
		int hIdx = CPARS + h * 8;
		H->block<CPARS, 8>(0, hIdx).noalias() = H->block<8, CPARS>(hIdx, 0).transpose();

		for (int t = h + 1; t < nframes[0]; t++) {
			int tIdx = CPARS + t * 8;
			H->block<8, 8>(hIdx, tIdx).noalias() += H->block<8, 8>(tIdx, hIdx).transpose();
			H->block<8, 8>(tIdx, hIdx).noalias() = H->block<8, 8>(hIdx, tIdx).transpose();
		}
	}
	H->block<CIPARS, 6>(0, CIPARS).noalias() = H->block<6, CIPARS>(CIPARS, 0).transpose();
}

void AccumulatedTopHessianSSE::stitchDoubleInternal(MatXX *H, VecX *b, EnergyFunctional const *const EF, bool usePrior, int min,
		int max, Vec10 *stats, int tid) {
	int toAggregate = NUM_THREADS;
	if (tid == -1) {
		toAggregate = 1;
		tid = 0;
	}	// special case: if we dont do multithreading, dont aggregate.
	if (min == max)
		return;

	for (int k = min; k < max; k++) {
		int h = k % nframes[0];
		int t = k / nframes[0];

		int hIdx = CPARS + h * 8;
		int tIdx = CPARS + t * 8;
		int aidx = h + nframes[0] * t;

		assert(aidx == k);

		MatPCPC accH = MatPCPC::Zero();
		int num = 0;

		for (int tid2 = 0; tid2 < toAggregate; tid2++) {
			acc[tid2][aidx].finish();
			if (acc[tid2][aidx].num > 0) {
				accH += acc[tid2][aidx].H.cast<double>();
				num += acc[tid2][aidx].num;
			}
		}

		if (num == 0)
			continue;

//		if (accH.block<6,6>(8,8).sum() > 0 && accH.block<6,6>(14,14).sum() > 0) {
//			std::cout << "H:T idx: " << h << ":" << t << "\t!!!\n";
//			std::cout << accH << "\n\n";
//			abort();
//		}

		if (h != t) {
			const Eigen::Ref<Mat88> &poseAbDiagBlk = accH.block<8, 8>(CIPARS, CIPARS);
			H[tid].block<8, 8>(hIdx, hIdx).noalias() += EF->adHost[aidx] * poseAbDiagBlk * EF->adHost[aidx].transpose();
			H[tid].block<8, 8>(tIdx, tIdx).noalias() += EF->adTarget[aidx] * poseAbDiagBlk * EF->adTarget[aidx].transpose();
			H[tid].block<8, 8>(hIdx, tIdx).noalias() += EF->adHost[aidx] * poseAbDiagBlk * EF->adTarget[aidx].transpose();

			// For left-left residuals only copy the intrisics, LR bit is zero for h != t
			const Eigen::Ref<Mat88> &poseAbIntrinsicsLowerBlk = accH.block<8, CIPARS>(CIPARS, 0);
			H[tid].block<8, CIPARS>(hIdx, 0).noalias() += EF->adHost[aidx] * poseAbIntrinsicsLowerBlk;
			H[tid].block<8, CIPARS>(tIdx, 0).noalias() += EF->adTarget[aidx] * poseAbIntrinsicsLowerBlk;

			// Intrisics only, again.. Top Left block
			H[tid].topLeftCorner<CIPARS, CIPARS>().noalias() += accH.block<CIPARS, CIPARS>(0, 0);

			// Accumulate residual to host and target pose & AB.
			const Eigen::Ref<Vec8> &poseAbResidual = accH.block<8, 1>(CIPARS, CIPARS + 8);
			b[tid].segment<8>(hIdx).noalias() += EF->adHost[aidx] * poseAbResidual;
			b[tid].segment<8>(tIdx).noalias() += EF->adTarget[aidx] * poseAbResidual;

			// Accumulate intrisics residual only, LR pose for LL res is zero.
			b[tid].head<CIPARS>().noalias() += accH.block<CIPARS, 1>(0, CIPARS + 8);
		} else {
			// h == t. Accumulate just the CPARS to the top left corner of H and first elements of b

			// Intrinsics
			H[tid].topLeftCorner<CIPARS, CIPARS>().noalias() += accH.block<CIPARS, CIPARS>(0, 0);
			b[tid].head<CIPARS>().noalias() += accH.block<CIPARS, 1>(0, CIPARS + 8);

			// LR pose with adjoint...
			// const Eigen::Ref<Mat66> &adjoint = EF->adHost[aidx].topLeftCorner(6, 6);
			//std::cout << adjoint << "\n\n";
			const Mat66 adjoint = Mat66::Identity().topLeftCorner(6, 6); // TODO why does using EF->adHost not work...
			H[tid].block<6, 6>(CIPARS, CIPARS).noalias() += adjoint * accH.block<6, 6>(CIPARS, CIPARS) * adjoint.transpose();
			H[tid].block<6, CIPARS>(CIPARS, 0).noalias() += adjoint * accH.block<6, CIPARS>(CIPARS, 0); // Is copied/flipped in copyUpperToLowerDiagonal
			b[tid].segment<6>(CIPARS).noalias() += adjoint * accH.block<6, 1>(CIPARS, CIPARS + 8);
		}
	}

	// only do this on one thread.
	if (min == 0 && usePrior) {
		H[tid].diagonal().head<CPARS>() += EF->cPrior;
		b[tid].head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
		for (int h = 0; h < nframes[tid]; h++) {
			H[tid].diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;
			b[tid].segment<8>(CPARS + h * 8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
		}
	}
}

}

