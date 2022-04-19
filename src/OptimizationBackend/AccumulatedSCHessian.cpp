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

#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso {

void AccumulatedSCHessianSSE::addPoint(EFPoint *p, bool shiftPriorToZero, int tid) {
	int ngoodres = 0;
	for (EFResidual *r : p->residualsAll)
		if (r->isActive())
			ngoodres++;
	if (ngoodres == 0) {
		p->HdiF = 0;
		p->bdSumF = 0;
		p->data->idepth_hessian = 0;
		p->data->maxRelBaseline = 0;
		return;
	}

	float H = p->Hdd_accAF + p->Hdd_accLF + p->priorF;
	if (H < 1e-10)
		H = 1e-10;

	p->data->idepth_hessian = H;

	/* Depth part of H(H_bb) is a diagonal matrix. We need it's inverse for the Schur-Complement.
	 * Just save the inverted diagonal element:
	 *    1 / (dRes^2/dDepth^2)
	 * One element per point (not pixel)
	 */
	p->HdiF = 1.0 / H;
	p->bdSumF = p->bd_accAF + p->bd_accLF;
	if (shiftPriorToZero)
		p->bdSumF += p->priorF * p->deltaF;

	VecCf Hcd = p->Hcd_accAF + p->Hcd_accLF;
	accHcc[tid].update(Hcd, Hcd, p->HdiF);
	accbc[tid].update(Hcd, p->bdSumF * p->HdiF);

	assert(std::isfinite((float)(p->HdiF)));

	int nf = nframes[tid];
	for (EFResidual *r1 : p->residualsAll) {
		if (!r1->isActive())
			continue;

		int i = r1->hostIDX;
		int j = r1->targetIDX;
		int ii = i + nf * i;

		for (EFResidual *r2 : p->residualsAll) {
			int k = r2->targetIDX;

			if (!r2->isActive() || j > k)
				continue;

			/* Contribution factor of pair of depth residuals on pose-pose block Haa...
			 * Accumulation over pairs of residuals that have a common host targets
			 *
			 * Why 4 updates?
			 * We're computing the H_ab * H_bb' * H_ba bit of the Schur Compliment.
			 * H_bb is a diagonal matrix with one entry for each point, the point's inverse depth.
			 * A point's host frame poseAB info is held in a block on the diagonal of H_aa.
			 * H_ab and H_ba have column and row entries for the influences between each point in H_bb and each frame poseAB in H_aa.
			 * r1 and r2 go through all combinations of these H_ab & H_ba entries.
			 * In H_aa, the host frame's poseAB entry and the poseAB entry that correspond to the selected entries in H_ab and H_ba form a square.
			 * The four updates are to the corners of this square.
			 * i - Indexes the host frame's poseAB on the diagonal of H_aa
			 * j - Indexes the row of the column in H_ab
			 * k - Indexes the column of the row in H_ba
			 *
			 * The Adjoints...
			 *
			 * Symmetric...   Seems this computes for a pair of residuals the updates for r1,r2 and then r2,r1. which keeps the Pose-Pose block symmetric..
			 * Could be cheaper to compute only the upper half and make it symmetric by copying/transposing blocks...
			 *
			 */
			if (i == j && i == k) {
				// Both r1 and r2 are to the right image

			} else if (i == j) {
				// r1 is to right image

			} else if (i == k) {
				// r2 is to right image

			} else if (j == k) {
				int jk = j + nf * k;
				int ji = j + nf * i;
				int ik = i + nf * k;

				accD[tid][ii].update(r1->JpJdAdH, r2->JpJdAdH, p->HdiF);
				accD[tid][jk].update(r1->JpJdAdT, r2->JpJdAdT, p->HdiF);
				accD[tid][ji].update(r1->JpJdAdT, r2->JpJdAdH, p->HdiF);
				accD[tid][ik].update(r1->JpJdAdH, r2->JpJdAdT, p->HdiF);
			} else {
				accD[tid][ii].update(r1->JpJdAdH, r2->JpJdAdH, p->HdiF);
				accD[tid][ii].update(r2->JpJdAdH, r1->JpJdAdH, p->HdiF);

				if (j < k) {
					int jk = j + nf * k;
					accD[tid][jk].update(r1->JpJdAdT, r2->JpJdAdT, p->HdiF);
				} else {
					int kj = k + nf * j;
					accD[tid][kj].update(r2->JpJdAdT, r1->JpJdAdT, p->HdiF);
				}

				if (j < i) {
					int ji = j + nf * i;
					accD[tid][ji].update(r1->JpJdAdT, r2->JpJdAdH, p->HdiF);
				} else {
					int ij = i + nf * j;
					accD[tid][ij].update(r2->JpJdAdH, r1->JpJdAdT, p->HdiF);
				}

				if (i < k) {
					int ik = i + nf * k;
					accD[tid][ik].update(r1->JpJdAdH, r2->JpJdAdT, p->HdiF);
				} else {
					int ki = k + nf * i;
					accD[tid][ki].update(r2->JpJdAdT, r1->JpJdAdH, p->HdiF);
				}
			}

			//accD[tid][r1ht + r2->targetIDX * nf * nf].update(r1->JpJdF, r2->JpJdF, p->HdiF);
		}

		accE[tid][i].update(r1->JpJdAdH, Hcd, p->HdiF);
		accE[tid][j].update(r1->JpJdAdT, Hcd, p->HdiF);
		accEB[tid][i].update(r1->JpJdAdH, p->HdiF * p->bdSumF);
		accEB[tid][j].update(r1->JpJdAdT, p->HdiF * p->bdSumF);
		
		//accE[tid][r1ht].update(r1->JpJdF, Hcd, p->HdiF);
		//accEB[tid][r1ht].update(r1->JpJdF, p->HdiF * p->bdSumF);
	}
}

void AccumulatedSCHessianSSE::stitchDoubleInternal(MatXX *H, VecX *b, EnergyFunctional const *const EF, int min, int max,
		Vec10 *stats, int tid) {
	int toAggregate = NUM_THREADS;
	if (tid == -1) {
		toAggregate = 1;
		tid = 0;
	}	// special case: if we dont do multithreading, dont aggregate.
	if (min == max)
		return;

	int nf = nframes[0];

	for (int ij = min; ij < max; ij++) {
		int i = ij % nf;
		int j = ij / nf;

		int iIdx = CPARS + i * 8;
		int jIdx = CPARS + j * 8;

		if (i == 0) {
			for (int tid2 = 0; tid2 < toAggregate; tid2++) {
				H[tid].block<8, CPARS>(jIdx, 0) += accE[tid2][j].A.cast<double>();
				b[tid].segment<8>(jIdx) += accEB[tid2][j].A.cast<double>();
			}

//			H[tid].block<8, CPARS>(iIdx, 0) += EF->adHost[ij] * E;
//			H[tid].block<8, CPARS>(jIdx, 0) += EF->adTarget[ij] * E;
//			b[tid].segment<8>(iIdx) += EF->adHost[ij] * EB;
//			b[tid].segment<8>(jIdx) += EF->adTarget[ij] * EB;
		}

		if (j >= i) {
			for (int tid2 = 0; tid2 < toAggregate; tid2++) {
				H[tid].block<8, 8>(iIdx, jIdx) += accD[tid2][ij].A.cast<double>();
			}
		}

		// Nframes^2 Pose-pose blocks
//		for (int k = 0; k < nf; k++) {
//			int ijk = ij + k * nf * nf;
//			int ik = i + nf * k;
//
//			int kIdx = CPARS + k * 8;
//
//			Mat88 D = Mat88::Zero();
//
//			int num = 0;
//			for (int tid2 = 0; tid2 < toAggregate; tid2++) {
//				if (accD[tid2][ijk].num > 0) {
//					D += accD[tid2][ijk].A.cast<double>();
//					num += accD[tid2][ijk].num;
//				}
//			}
//
//			if (num > 0) {
//				H[tid].block<8, 8>(iIdx, iIdx) += EF->adHost[ij] * D * EF->adHost[ik].transpose();
//				H[tid].block<8, 8>(jIdx, kIdx) += EF->adTarget[ij] * D * EF->adTarget[ik].transpose();
//				H[tid].block<8, 8>(jIdx, iIdx) += EF->adTarget[ij] * D * EF->adHost[ik].transpose();
//				H[tid].block<8, 8>(iIdx, kIdx) += EF->adHost[ij] * D * EF->adTarget[ik].transpose();
//			}
//		}
	}

	// One camera params block
	if (min == 0) {
		for (int tid2 = 0; tid2 < toAggregate; tid2++) {
			H[tid].topLeftCorner<CPARS, CPARS>() += accHcc[tid2].A.cast<double>();
			b[tid].head<CPARS>() += accbc[tid2].A.cast<double>();
		}
	}
}

void AccumulatedSCHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional const *const EF) {
	H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
	b = VecX::Zero(nframes[0] * 8 + CPARS);

	stitchDoubleInternal(&H, &b, EF, 0, nframes[0] * nframes[0], 0, -1);
	copyUpperToLowerDiagonal(&H);
}

void AccumulatedSCHessianSSE::copyUpperToLowerDiagonal(MatXX *H) {
	// make diagonal by copying over parts.
	for (int h = 0; h < nframes[0]; h++) {
		int hIdx = CPARS + h * 8;
		H->block<CPARS, 8>(0, hIdx).noalias() = H->block<8, CPARS>(hIdx, 0).transpose();

		for (int t = h + 1; t < nframes[0]; t++) {
			int tIdx = CPARS + t * 8;
			H->block<8, 8>(tIdx, hIdx).noalias() = H->block<8, 8>(hIdx, tIdx).transpose();
		}
	}
}

}
