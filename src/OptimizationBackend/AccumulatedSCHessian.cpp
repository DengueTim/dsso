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
		int r1ht = r1->hostIDX + r1->targetIDX * nf;

		for (EFResidual *r2 : p->residualsAll) {
			if (!r2->isActive())
				continue;

			/* Contribution factor of pair of depth residuals on pose-pose block Haa...
			 * Accumulation over pairs of residuals that have a common host targets
			 */
			accD[tid][r1ht + r2->targetIDX * nf * nf].update(r1->JpJdF, r2->JpJdF, p->HdiF);
		}

		accE[tid][r1ht].update(r1->JpJdF, Hcd, p->HdiF);
		accEB[tid][r1ht].update(r1->JpJdF, p->HdiF * p->bdSumF);
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
	int nframes2 = nf * nf;

	for (int k = min; k < max; k++) {
		int i = k % nf;
		int j = k / nf;

		int iIdx = CPARS + i * 8;
		int jIdx = CPARS + j * 8;
		int ijIdx = i + nf * j;   // = k!?

		Mat8C Hpc = Mat8C::Zero();
		Vec8 bp = Vec8::Zero();

		for (int tid2 = 0; tid2 < toAggregate; tid2++) {
			Hpc += accE[tid2][ijIdx].A.cast<double>();
			bp += accEB[tid2][ijIdx].A.cast<double>();
		}

		H[tid].block<8, CPARS>(iIdx, 0) += EF->adHost[ijIdx] * Hpc;
		H[tid].block<8, CPARS>(jIdx, 0) += EF->adTarget[ijIdx] * Hpc;
		b[tid].segment<8>(iIdx) += EF->adHost[ijIdx] * bp;
		b[tid].segment<8>(jIdx) += EF->adTarget[ijIdx] * bp;

		for (int fi = 0; fi < nf; fi++) {
			int kIdx = CPARS + fi * 8;
			int ijkIdx = ijIdx + fi * nframes2;
			int ikIdx = i + nf * fi;

			Mat88 accDM = Mat88::Zero();

			for (int tid2 = 0; tid2 < toAggregate; tid2++) {
				if (accD[tid2][ijkIdx].num > 0)
					accDM += accD[tid2][ijkIdx].A.cast<double>();
			}

			H[tid].block<8, 8>(iIdx, iIdx) += EF->adHost[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
			H[tid].block<8, 8>(jIdx, kIdx) += EF->adTarget[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
			H[tid].block<8, 8>(jIdx, iIdx) += EF->adTarget[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
			H[tid].block<8, 8>(iIdx, kIdx) += EF->adHost[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
		}
	}

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
	}
}

}
