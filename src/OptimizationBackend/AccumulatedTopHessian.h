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
#include "OptimizationBackend/MatrixAccumulators.h"
#include "vector"
#include <math.h>
#include "util/IndexThreadReduce.h"

namespace dso {

class EFFrame;
class EFPoint;

class AccumulatedTopHessianSSE {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;
	inline AccumulatedTopHessianSSE() {
		for (int tid = 0; tid < NUM_THREADS; tid++) {
			nres[tid] = 0;
			acc[tid] = 0;
			nframes[tid] = 0;
		}

	}
	;
	inline ~AccumulatedTopHessianSSE() {
		for (int tid = 0; tid < NUM_THREADS; tid++) {
			if (acc[tid] != 0)
				delete[] acc[tid];
		}
	}
	;

	inline void setZero(int nFrames, int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0) {

		if (nFrames != nframes[tid]) {
			if (acc[tid] != 0)
				delete[] acc[tid];
#if USE_XI_MODEL
			acc[tid] = new Accumulator14[nFrames*nFrames];
#else
			acc[tid] = new AccumulatorApprox[nFrames * nFrames];
#endif
		}

		for (int i = 0; i < nFrames * nFrames; i++) {
			acc[tid][i].initialize();
		}

		nframes[tid] = nFrames;
		nres[tid] = 0;

	}
	void stitchDouble(MatXX &H, VecX &b, Mat88 *adHost, Mat88 *adTarget, bool useDelta);

	template<int mode> void addPoint(EFPoint *p, Mat18f *adHTdeltaF, VecC *cDelta, int tid = 0);

	void stitchDoubleMT(IndexThreadReduce<Vec10> *red, MatXX &H, VecX &b, Mat88 *adHost, Mat88 *adTarget, bool MT) {
		// sum up, splitting by bock in square.
		if (MT) {
			MatXX Hs[NUM_THREADS];
			VecX bs[NUM_THREADS];
			for (int i = 0; i < NUM_THREADS; i++) {
				assert(nframes[0] == nframes[i]);
				Hs[i] = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
				bs[i] = VecX::Zero(nframes[0] * 8 + CPARS);
			}

			red->reduce(
					boost::bind(&AccumulatedTopHessianSSE::stitchDoubleInternal, this, Hs, bs, adHost, adTarget, _1, _2, _3, _4), 0,
					nframes[0] * nframes[0], 0);

			// sum up results
			H = Hs[0];
			b = bs[0];

			for (int i = 1; i < NUM_THREADS; i++) {
				H.noalias() += Hs[i];
				b.noalias() += bs[i];
				nres[0] += nres[i];
			}
		} else {
			H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
			b = VecX::Zero(nframes[0] * 8 + CPARS);
			stitchDoubleInternal(&H, &b, adHost, adTarget, 0, nframes[0] * nframes[0], 0, -1);
		}

		copyUpperToLowerDiagonal(H);
	}

	void addPrior(MatXX &H, VecX &b, VecC &cPrior, VecC &cDelta, std::vector<EFFrame*> &frames);

	int nframes[NUM_THREADS];

	EIGEN_ALIGN16 AccumulatorApprox *acc[NUM_THREADS];

	int nres[NUM_THREADS];

	template<int mode> void addPointsInternal(std::vector<EFPoint*> *points, Mat18f *adHTdeltaF, VecC *cDelta, int min = 0,
			int max = 1, Vec10 *stats = 0, int tid = 0) {
		for (int i = min; i < max; i++)
			addPoint<mode>((*points)[i], adHTdeltaF, cDelta, tid);
	}

private:
	void stitchDoubleInternal(MatXX *H, VecX *b, Mat88 *adHost, Mat88 *adTarget, int min, int max, Vec10 *stats, int tid);
	void copyUpperToLowerDiagonal(MatXX &H);
};
}

