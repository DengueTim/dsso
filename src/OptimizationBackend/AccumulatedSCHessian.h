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

#pragma once

#include "util/NumType.h"
#include "util/IndexThreadReduce.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "vector"
#include <math.h>

namespace dso {

class EFPoint;

class AccumulatedSCHessianSSE {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	inline AccumulatedSCHessianSSE() {
		for (int i = 0; i < NUM_THREADS; i++) {
			accE[i] = 0;
			accEB[i] = 0;
			accD[i] = 0;
			nframes[i] = 0;
		}
	}

	inline ~AccumulatedSCHessianSSE() {
		for (int i = 0; i < NUM_THREADS; i++) {
			if (accE[i] != 0)
				delete[] accE[i];
			if (accEB[i] != 0)
				delete[] accEB[i];
			if (accD[i] != 0)
				delete[] accD[i];
		}
	}

	inline void setZero(int n, int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0) {
		const int nplus1 = n + 1; // The CPARS Left/Right pose is accumulated as part of D not Hcc.
		if (n != nframes[tid]) {
			if (accE[tid] != 0)
				delete[] accE[tid];
			if (accEB[tid] != 0)
				delete[] accEB[tid];
			if (accD[tid] != 0)
				delete[] accD[tid];
			accE[tid] = new AccumulatorXX<8, CIPARS> [nplus1];
			accEB[tid] = new AccumulatorX<8> [nplus1];
			accD[tid] = new AccumulatorXX<8, 8> [nplus1 * nplus1];
		}
		accbc[tid].initialize();
		accHcc[tid].initialize();

		for (int i = 0; i < nplus1; i++) {
			accE[tid][i].initialize();
			accEB[tid][i].initialize();
		}
		for (int i = 0; i < nplus1 * nplus1; i++) {
			accD[tid][i].initialize();
		}
		nframes[tid] = n;
	}
	void stitchDouble(MatXX &H_sc, VecX &b_sc);
	void addPoint(EFPoint *p, bool shiftPriorToZero, int tid = 0);

	void stitchDoubleMT(IndexThreadReduce<Vec10> *red, MatXX &H, VecX &b, bool MT) {
		// sum up, splitting by bock in square.
		const int dSizeSquared = (nframes[0] + 1) * (nframes[0] + 1);

		if (MT) {
			MatXX Hs[NUM_THREADS];
			VecX bs[NUM_THREADS];
			for (int i = 0; i < NUM_THREADS; i++) {
				assert(nframes[0] == nframes[i]);
				Hs[i] = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
				bs[i] = VecX::Zero(nframes[0] * 8 + CPARS);
			}

			red->reduce(boost::bind(&AccumulatedSCHessianSSE::stitchDoubleInternal, this, Hs, bs, _1, _2, _3, _4), 0, dSizeSquared,
					0);

			// sum up results
			H = Hs[0];
			b = bs[0];

			for (int i = 1; i < NUM_THREADS; i++) {
				H.noalias() += Hs[i];
				b.noalias() += bs[i];
			}
		} else {
			H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
			b = VecX::Zero(nframes[0] * 8 + CPARS);
			stitchDoubleInternal(&H, &b, 0, dSizeSquared, 0, -1);
		}

		copyUpperToLowerDiagonal(&H);
	}

	AccumulatorXX<8, CIPARS> *accE[NUM_THREADS];
	AccumulatorX<8> *accEB[NUM_THREADS];
	AccumulatorXX<8, 8> *accD[NUM_THREADS];
	AccumulatorXX<CIPARS, CIPARS> accHcc[NUM_THREADS];
	AccumulatorX<CIPARS> accbc[NUM_THREADS];
	int nframes[NUM_THREADS];

	void addPointsInternal(std::vector<EFPoint*> *points, bool shiftPriorToZero, int min = 0, int max = 1, Vec10 *stats = 0,
			int tid = 0) {
		for (int i = min; i < max; i++)
			addPoint((*points)[i], shiftPriorToZero, tid);
	}

private:
	void stitchDoubleInternal(MatXX *H, VecX *b, int min, int max, Vec10 *stats, int tid);
	void copyUpperToLowerDiagonal(MatXX *H);
};

}

