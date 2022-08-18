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
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso {

void AccumulatedSCHessianSSE::addPoint(EFPoint *p, bool shiftPriorToZero, int tid) {

//  To get data for unit tests..
//	if (rand() % 1000 == 0)
//		p->print();

	int ngoodres = 0;
	for (EFResidual *r : p->residualsAll)
		if (r->isActive)
			ngoodres++;
	if (ngoodres == 0) {
		p->HdiF = 0;
		p->bdSumF = 0;
		p->ph->idepth_hessian = 0;
		p->ph->maxRelBaseline = 0;
		return;
	}

	float H = p->Hdd_accAF + p->Hdd_accLF + p->priorF;
	if (H < 1e-10)
		H = 1e-10;

	p->ph->idepth_hessian = H;

	/* Depth part of H(H_bb) is a diagonal matrix. We need it's inverse for the Schur-Complement.
	 * Just save the inverted diagonal element:
	 *    1 / (dRes^2/dDepth^2)
	 * One element per point (not pixel)
	 */
	p->HdiF = 1.0 / H;
	assert(std::isfinite((float )(p->HdiF)));

	p->bdSumF = p->bd_accAF + p->bd_accLF;
	if (shiftPriorToZero)
		p->bdSumF += p->priorF * p->deltaF;

	VecCf Hcd = p->Hcd_accAF + p->Hcd_accLF;

	Vec8f HcdIntrinsics = Hcd.head<CIPARS>();
	accHcc[tid].update(HcdIntrinsics, HcdIntrinsics, p->HdiF);
	accbc[tid].update(HcdIntrinsics, p->bdSumF * p->HdiF);

	int dSize = nframes[tid] + 1;
	for (EFResidual *r1 : p->residualsAll) {
		if (!r1->isActive)
			continue;

		int i = r1->hostIDX + 1;
		int j = r1->targetIDX + 1;

		if ( j == i ) {
			for (EFResidual *r2 : p->residualsAll) {
				if (!r2->isActive)
					continue;

				int k = r2->targetIDX + 1;

				if ( k != i ) {
					accD[tid][i].update(r2->JpJdAdH, r1->JpJdAdT, p->HdiF);
					accD[tid][k].update(r2->JpJdAdT, r1->JpJdAdT, p->HdiF);
				}
			}

			accD[tid][0].update(r1->JpJdAdT, r1->JpJdAdT, p->HdiF);

			accE[tid][0].update(r1->JpJdAdT, HcdIntrinsics, p->HdiF);
			accEB[tid][0].update(r1->JpJdAdT, p->bdSumF * p->HdiF);
		} else {
			for (EFResidual *r2 : p->residualsAll) {
				if (!r2->isActive)
					continue;

				int k = r2->targetIDX + 1;

				if ( k != i ) {
					int ii = i + dSize * i;
					int jk = j + dSize * k;
					int ji = j + dSize * i;
					int ik = i + dSize * k;

					accD[tid][ii].update(r1->JpJdAdH, r2->JpJdAdH, p->HdiF);
					accD[tid][jk].update(r1->JpJdAdT, r2->JpJdAdT, p->HdiF);
					accD[tid][ji].update(r1->JpJdAdT, r2->JpJdAdH, p->HdiF);
					accD[tid][ik].update(r1->JpJdAdH, r2->JpJdAdT, p->HdiF);
				}
			}

			accE[tid][i].update(r1->JpJdAdH, HcdIntrinsics, p->HdiF);
			accEB[tid][i].update(r1->JpJdAdH, p->bdSumF * p->HdiF);
			accE[tid][j].update(r1->JpJdAdT, HcdIntrinsics, p->HdiF);
			accEB[tid][j].update(r1->JpJdAdT, p->bdSumF * p->HdiF);
		}
	}
}

void AccumulatedSCHessianSSE::stitchDoubleInternal(MatXX *H, VecX *b, int min, int max, Vec10 *stats, int tid) {
	int toAggregate = NUM_THREADS;
	if (tid == -1) {
		toAggregate = 1;
		tid = 0;
	}	// special case: if we dont do multithreading, dont aggregate.
	if (min == max)
		return;

	int dSize = nframes[0] + 1;

	for (int jk = min; jk < max; jk++) {
		int j = jk % dSize;
		int k = jk / dSize;

		const int jIdx = (j == 0) ? CIPARS : CPARS + (j - 1) * 8;
		const int kIdx = (k == 0) ? CIPARS : CPARS + (k - 1) * 8;

		const int js = (j == 0) ? 6 : 8;
		const int ks = (k == 0) ? 6 : 8;

		if (j == 0) {
			for (int tid2 = 0; tid2 < toAggregate; tid2++) {
				H[tid].block(kIdx, 0, ks, CIPARS) += accE[tid2][k].A.topLeftCorner(ks, CIPARS).cast<double>();
				b[tid].segment(kIdx, ks) += accEB[tid2][k].A.topLeftCorner(ks, 1).cast<double>();
			}
		}

		for (int tid2 = 0; tid2 < toAggregate; tid2++) {
			H[tid].block(jIdx, kIdx, js, ks) += accD[tid2][jk].A.topLeftCorner(js, ks).cast<double>();
		}
	}

// One camera params block
	if (tid == 0) {
		for (int tid2 = 0; tid2 < toAggregate; tid2++) {
			H[tid].topLeftCorner<CIPARS, CIPARS>() += accHcc[tid2].A.cast<double>();
			b[tid].head<CIPARS>() += accbc[tid2].A.cast<double>();
		}
	}
}

void AccumulatedSCHessianSSE::stitchDouble(MatXX &H, VecX &b) {
	const int dSizeSquared = (nframes[0] + 1) * (nframes[0] + 1);

	H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
	b = VecX::Zero(nframes[0] * 8 + CPARS);

	stitchDoubleInternal(&H, &b, 0, dSizeSquared, 0, -1);
	copyUpperToLowerDiagonal(&H);
}

void AccumulatedSCHessianSSE::copyUpperToLowerDiagonal(MatXX *H) {
// make diagonal by copying over parts.
	H->block<CIPARS, 6>(0, CIPARS).noalias() = H->block<6, CIPARS>(CIPARS, 0).transpose();
	for (int h = 0; h < nframes[0]; h++) {
		int hIdx = CPARS + h * 8;
		H->block<CIPARS, 8>(0, hIdx).noalias() = H->block<8, CIPARS>(hIdx, 0).transpose();
		H->block<6,8>(CIPARS, hIdx).noalias() = H->block<8, 6>(hIdx, CIPARS).transpose();
	}
}

}
