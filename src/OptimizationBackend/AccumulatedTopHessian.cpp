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
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <iostream>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

template<int mode>
void __attribute__((optimize(0))) AccumulatedTopHessianSSE::addPoint(EFPoint *p, Mat18f *adHTdeltaF, VecC *cDelta, int tid) { // 0 = active, 1 = linearized, 2=marginalize
	assert(mode == 0 || mode == 1 || mode == 2);

	float bd_acc = 0;
	float Hdd_acc = 0;
	VecCf Hcd_acc = VecCf::Zero();
	VecCf dc = cDelta->cast<float>();

	for (EFResidual *r : p->residualsAll) {
		if (mode == 0) {
			if (r->isLinearized || !r->isActive)
				continue;
		}
		if (mode == 1) {
			if (!r->isLinearized || !r->isActive)
				continue;
		}
		if (mode == 2) {
			if (!r->isActive)
				continue;
			assert(r->isLinearized);
		}

		RawResidualJacobian *rJ = r->J;
		int htIDX = r->hostIDX + r->targetIDX * nframes[tid];

		bool leftToRight = r->hostIDX == r->targetIDX;

		VecNRf resApprox;
		if (mode == 0)
			resApprox = rJ->resF;
		if (mode == 2)
			resApprox = r->res_toZeroF;
		if (mode == 1) {
			Mat18f dp = adHTdeltaF[htIDX];
			float dd = p->deltaF;

			float Jp_delta_x_1 = rJ->Jpdd[0] * dd;
			float Jp_delta_y_1 = rJ->Jpdd[1] * dd;

			// compute Jp*delta
			if (leftToRight) {
				Jp_delta_x_1 += rJ->Jpdc[0].dot(dc);
				Jp_delta_y_1 += rJ->Jpdc[1].dot(dc);
			} else {
				Jp_delta_x_1 += rJ->Jpdc[0].head<4>().dot(dc.head<4>()); // Left camera intrinsics only
				Jp_delta_x_1 += rJ->Jpdxi[0].dot(dp.head<6>());
				Jp_delta_y_1 += rJ->Jpdc[1].head<4>().dot(dc.head<4>());
				Jp_delta_y_1 += rJ->Jpdxi[1].dot(dp.head<6>());
			}

			__m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
			__m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
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

		const Vec6f &JpdxiX = leftToRight ? rJ->Jpdc[0].segment(CIPARS, 6) : rJ->Jpdxi[0];
		const Vec6f &JpdxiY = leftToRight ? rJ->Jpdc[1].segment(CIPARS, 6) : rJ->Jpdxi[1];

		// The left-right contribution is stored in the blocks on the diagonal..
		acc[tid][htIDX].update(rJ->Jpdc[0].data(), JpdxiX.data(), rJ->Jpdc[1].data(), JpdxiY.data(), rJ->JIdx2(0, 0),
				rJ->JIdx2(0, 1), rJ->JIdx2(1, 1));

		acc[tid][htIDX].updateBotRight(rJ->Jab2(0, 0), rJ->Jab2(0, 1), Jab_r[0], rJ->Jab2(1, 1), Jab_r[1], rr);

		acc[tid][htIDX].updateTopRight(rJ->Jpdc[0].data(), JpdxiX.data(), rJ->Jpdc[1].data(), JpdxiY.data(), rJ->JabJIdx(0, 0),
				rJ->JabJIdx(0, 1), rJ->JabJIdx(1, 0), rJ->JabJIdx(1, 1), JI_r[0], JI_r[1]);

		Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd;
		bd_acc += JI_r[0] * rJ->Jpdd[0] + JI_r[1] * rJ->Jpdd[1];
		Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd);   // dRes^2/dDepth^2
		Hcd_acc += rJ->Jpdc[0] * Ji2_Jpdd[0] + rJ->Jpdc[1] * Ji2_Jpdd[1];

		nres[tid]++;
	}

	if (mode == 0) {
		p->Hdd_accAF = Hdd_acc;
		p->Hcd_accAF = Hcd_acc;
		p->bd_accAF = bd_acc;
	}
	if (mode == 1 || mode == 2) {
		p->Hdd_accLF = Hdd_acc;
		p->Hcd_accLF = Hcd_acc;
		p->bd_accLF = bd_acc;
	}
	if (mode == 2) {
		p->Hdd_accAF = 0;
		p->Hcd_accAF.setZero();
		p->bd_accAF = 0;
	}

}
template void AccumulatedTopHessianSSE::addPoint<0>(EFPoint *p, Mat18f *adHTdeltaF, VecC *dc, int tid);
template void AccumulatedTopHessianSSE::addPoint<1>(EFPoint *p, Mat18f *adHTdeltaF, VecC *dc, int tid);
template void AccumulatedTopHessianSSE::addPoint<2>(EFPoint *p, Mat18f *adHTdeltaF, VecC *dc, int tid);

void AccumulatedTopHessianSSE::stitchDouble(MatXX &H, VecX &b, Mat88 *adHost, Mat88 *adTarget, bool useDelta) {
	H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
	b = VecX::Zero(nframes[0] * 8 + CPARS);

	stitchDoubleInternal(&H, &b, adHost, adTarget, 0, nframes[0] * nframes[0], 0, -1);

	copyUpperToLowerDiagonal(H);
}

void AccumulatedTopHessianSSE::copyUpperToLowerDiagonal(MatXX &H) {
	// make diagonal by copying over parts.
	for (int h = 0; h < nframes[0]; h++) {
		int hIdx = CPARS + h * 8;
		H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();

		for (int t = h + 1; t < nframes[0]; t++) {
			int tIdx = CPARS + t * 8;
			H.block<8, 8>(hIdx, tIdx).noalias() += H.block<8, 8>(tIdx, hIdx).transpose();
			H.block<8, 8>(tIdx, hIdx).noalias() = H.block<8, 8>(hIdx, tIdx).transpose();
		}
	}
	H.block<CIPARS, 6>(0, CIPARS).noalias() = H.block<6, CIPARS>(CIPARS, 0).transpose();
}

void AccumulatedTopHessianSSE::stitchDoubleInternal(MatXX *H, VecX *b, Mat88 *adHost, Mat88 *adTarget, int min, int max, Vec10 *stats, int tid) {
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

#ifndef ADD_LR_RESIDUALS
		if (h == t) { assert(num == 0); }
#endif

		if (num == 0)
			continue;

		// accH is symmetric.

		// Intrinsics.  Top left block of H and head of b.
		H[tid].topLeftCorner<CIPARS, CIPARS>().noalias() += accH.block<CIPARS, CIPARS>(0, 0);
		b[tid].head<CIPARS>().noalias() += accH.block<CIPARS, 1>(0, CIPARS + 8);

		if (h != t) {
			const Eigen::Ref<Mat88> &poseAbBlk = accH.block<8, 8>(CIPARS, CIPARS);
			H[tid].block<8, 8>(hIdx, hIdx).noalias() += adHost[aidx] * poseAbBlk * adHost[aidx].transpose();
			H[tid].block<8, 8>(tIdx, tIdx).noalias() += adTarget[aidx] * poseAbBlk * adTarget[aidx].transpose();
			H[tid].block<8, 8>(hIdx, tIdx).noalias() += adHost[aidx] * poseAbBlk * adTarget[aidx].transpose();

			// For left-left residuals only copy the intrisics, LR bit is zero for h != t
			const Eigen::Ref<Mat88> &poseAbIntrinsicsLowerBlk = accH.block<8, CIPARS>(CIPARS, 0);
			H[tid].block<8, CIPARS>(hIdx, 0).noalias() += adHost[aidx] * poseAbIntrinsicsLowerBlk;
			H[tid].block<8, CIPARS>(tIdx, 0).noalias() += adTarget[aidx] * poseAbIntrinsicsLowerBlk;

			// Accumulate residual to host and target pose & AB.
			const Eigen::Ref<Vec8> &poseAbResidual = accH.block<8, 1>(CIPARS, CIPARS + 8);
			b[tid].segment<8>(hIdx).noalias() += adHost[aidx] * poseAbResidual;
			b[tid].segment<8>(tIdx).noalias() += adTarget[aidx] * poseAbResidual;

		} else {
			// h == t. Accumulate just the LR pose to the top left corner of H and first elements of b
			// LR pose with adjoint...
//			const Mat66 adjoint = adHost[aidx].topLeftCorner(6, 6);
			const Mat66 adjoint = adTarget[aidx].topLeftCorner(6, 6); // Identity with scaling.
			H[tid].block<6, 6>(CIPARS, CIPARS).noalias() += adjoint * accH.block<6, 6>(CIPARS, CIPARS) * adjoint.transpose();
			H[tid].block<6, CIPARS>(CIPARS, 0).noalias() += adjoint * accH.block<6, CIPARS>(CIPARS, 0); // Is copied/flipped in copyUpperToLowerDiagonal
			b[tid].segment<6>(CIPARS).noalias() += adjoint * accH.block<6, 1>(CIPARS, CIPARS + 8);
		}
	}
}

void AccumulatedTopHessianSSE::addPrior(MatXX &H, VecX &b, VecC &cPrior, VecC &cDelta, std::vector<EFFrame*> &frames) {
	H.diagonal().head<CPARS>() += cPrior;
	b.head<CPARS>() += cPrior.cwiseProduct(cDelta);
	for (int h = 0; h < nframes[0]; h++) {
		H.diagonal().segment<8>(CPARS + h * 8) += frames[h]->prior;
		b.segment<8>(CPARS + h * 8) += frames[h]->prior.cwiseProduct(frames[h]->delta_prior);
	}
}

}

