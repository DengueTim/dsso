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

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

template<int i, int j>
class AccumulatorXX {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;

	Eigen::Matrix<float, i, j> A;
	size_t num;

	inline void initialize() {
		A.setZero();
		num = 0;
	}

	inline void update(const Eigen::Matrix<float, i, 1> &L, const Eigen::Matrix<float, j, 1> &R, float w) {
		A += w * L * R.transpose();
		num++;
	}
};

class AccumulatorCC {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;

	Eigen::Matrix<float, CPARS, CPARS> A;
	size_t num;

	inline void initialize() {
		A.setZero();
		num = 0;
	}

	inline void update(const Eigen::Matrix<float, CPARS, 1> &L, const Eigen::Matrix<float, CPARS, 1> &R, float w) {
		A += w * L * R.transpose();
		num++;
	}

	inline void updateLrPose(const Eigen::Matrix<float, 8, 1> &L, const Eigen::Matrix<float, 8, 1> &R, float w) {
		// Camera params don't have photometric params A & B
		A.bottomRightCorner(6, 6) += w * L.topLeftCorner(6, 6) * R.topLeftCorner(6, 6).transpose();
		num++;
	}
};

class Accumulator11 {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;

	float A;
	size_t num;

	inline void initialize() {
		A = 0;
		memset(SSEData, 0, sizeof(float) * 4 * 1);
		num = 0;
	}

	inline void finish() {
		A = SSEData[0 + 0] + SSEData[0 + 1] + SSEData[0 + 2] + SSEData[0 + 3];
	}

	inline void updateSingle(const float val) {
		SSEData[0] += val;
		num++;
	}

	inline void updateSSE(const __m128 val) {
		_mm_store_ps(SSEData, _mm_add_ps(_mm_load_ps(SSEData), val));
		num += 4;
	}

private:
	EIGEN_ALIGN16 float SSEData[4 * 1];
};

template<int i>
class AccumulatorX {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;

	Eigen::Matrix<float, i, 1> A;
	size_t num;

	inline void initialize() {
		A.setZero();
		num = 0;
	}

	inline void update(const Eigen::Matrix<float, i, 1> &L, float w) {
		A += w * L;
		num++;
	}

	inline void updateNoWeight(const Eigen::Matrix<float, i, 1> &L) {
		A += L;
		num++;
	}
};

class Accumulator14 {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;

	Mat1414f H;
	size_t num;

	inline void initialize() {
		H.setZero();
		memset(SSEData, 0, sizeof(float) * 4 * 105);
		num = 0;
	}

	inline void finish() {
		H.setZero();

		int idx = 0;
		for (int r = 0; r < 14; r++)
			for (int c = r; c < 14; c++) {
				float d = SSEData[idx + 0] + SSEData[idx + 1] + SSEData[idx + 2] + SSEData[idx + 3];
				H(r, c) = H(c, r) = d;
				idx += 4;
			}
		assert(idx == 4 * 105);
	}

	inline void updateSSE(const __m128 J0, const __m128 J1, const __m128 J2, const __m128 J3, const __m128 J4, const __m128 J5,
			const __m128 J6, const __m128 J7, const __m128 J8, const __m128 J9, const __m128 J10, const __m128 J11,
			const __m128 J12, const __m128 J13) {
		float *pt = SSEData;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J0)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J1)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J2)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J3)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J8)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J9)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J10)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J11)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J12)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J13)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J1)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J2)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J3)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J8)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J9)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J10)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J11)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J12)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J13)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J2)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J3)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J8)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J9)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J10)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J11)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J12)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J13)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J3)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J8)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J9)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J10)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J11)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J12)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J13)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J8)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J9)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J10)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J11)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J12)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J13)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J8)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J9)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J10)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J11)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J12)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J13)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J8)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J9)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J10)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J11)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J12)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J13)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J8)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J9)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J10)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J11)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J12)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J13)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J8)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J9)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J10)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J11)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J12)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J13)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J9)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J10)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J11)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J12)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J13)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J10, J10)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J10, J11)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J10, J12)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J10, J13)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J11, J11)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J11, J12)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J11, J13)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J12, J12)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J12, J13)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J13, J13)));
		pt += 4;

		num += 4;
	}

	inline void updateSingle(const float J0, const float J1, const float J2, const float J3, const float J4, const float J5,
			const float J6, const float J7, const float J8, const float J9, const float J10, const float J11, const float J12,
			const float J13, int off = 0) {
		float *pt = SSEData + off;
		*pt += J0 * J0;
		pt += 4;
		*pt += J1 * J0;
		pt += 4;
		*pt += J2 * J0;
		pt += 4;
		*pt += J3 * J0;
		pt += 4;
		*pt += J4 * J0;
		pt += 4;
		*pt += J5 * J0;
		pt += 4;
		*pt += J6 * J0;
		pt += 4;
		*pt += J7 * J0;
		pt += 4;
		*pt += J8 * J0;
		pt += 4;
		*pt += J9 * J0;
		pt += 4;
		*pt += J10 * J0;
		pt += 4;
		*pt += J11 * J0;
		pt += 4;
		*pt += J12 * J0;
		pt += 4;
		*pt += J13 * J0;
		pt += 4;

		*pt += J1 * J1;
		pt += 4;
		*pt += J2 * J1;
		pt += 4;
		*pt += J3 * J1;
		pt += 4;
		*pt += J4 * J1;
		pt += 4;
		*pt += J5 * J1;
		pt += 4;
		*pt += J6 * J1;
		pt += 4;
		*pt += J7 * J1;
		pt += 4;
		*pt += J8 * J1;
		pt += 4;
		*pt += J9 * J1;
		pt += 4;
		*pt += J10 * J1;
		pt += 4;
		*pt += J11 * J1;
		pt += 4;
		*pt += J12 * J1;
		pt += 4;
		*pt += J13 * J1;
		pt += 4;

		*pt += J2 * J2;
		pt += 4;
		*pt += J3 * J2;
		pt += 4;
		*pt += J4 * J2;
		pt += 4;
		*pt += J5 * J2;
		pt += 4;
		*pt += J6 * J2;
		pt += 4;
		*pt += J7 * J2;
		pt += 4;
		*pt += J8 * J2;
		pt += 4;
		*pt += J9 * J2;
		pt += 4;
		*pt += J10 * J2;
		pt += 4;
		*pt += J11 * J2;
		pt += 4;
		*pt += J12 * J2;
		pt += 4;
		*pt += J13 * J2;
		pt += 4;

		*pt += J3 * J3;
		pt += 4;
		*pt += J4 * J3;
		pt += 4;
		*pt += J5 * J3;
		pt += 4;
		*pt += J6 * J3;
		pt += 4;
		*pt += J7 * J3;
		pt += 4;
		*pt += J8 * J3;
		pt += 4;
		*pt += J9 * J3;
		pt += 4;
		*pt += J10 * J3;
		pt += 4;
		*pt += J11 * J3;
		pt += 4;
		*pt += J12 * J3;
		pt += 4;
		*pt += J13 * J3;
		pt += 4;

		*pt += J4 * J4;
		pt += 4;
		*pt += J5 * J4;
		pt += 4;
		*pt += J6 * J4;
		pt += 4;
		*pt += J7 * J4;
		pt += 4;
		*pt += J8 * J4;
		pt += 4;
		*pt += J9 * J4;
		pt += 4;
		*pt += J10 * J4;
		pt += 4;
		*pt += J11 * J4;
		pt += 4;
		*pt += J12 * J4;
		pt += 4;
		*pt += J13 * J4;
		pt += 4;

		*pt += J5 * J5;
		pt += 4;
		*pt += J6 * J5;
		pt += 4;
		*pt += J7 * J5;
		pt += 4;
		*pt += J8 * J5;
		pt += 4;
		*pt += J9 * J5;
		pt += 4;
		*pt += J10 * J5;
		pt += 4;
		*pt += J11 * J5;
		pt += 4;
		*pt += J12 * J5;
		pt += 4;
		*pt += J13 * J5;
		pt += 4;

		*pt += J6 * J6;
		pt += 4;
		*pt += J7 * J6;
		pt += 4;
		*pt += J8 * J6;
		pt += 4;
		*pt += J9 * J6;
		pt += 4;
		*pt += J10 * J6;
		pt += 4;
		*pt += J11 * J6;
		pt += 4;
		*pt += J12 * J6;
		pt += 4;
		*pt += J13 * J6;
		pt += 4;

		*pt += J7 * J7;
		pt += 4;
		*pt += J8 * J7;
		pt += 4;
		*pt += J9 * J7;
		pt += 4;
		*pt += J10 * J7;
		pt += 4;
		*pt += J11 * J7;
		pt += 4;
		*pt += J12 * J7;
		pt += 4;
		*pt += J13 * J7;
		pt += 4;

		*pt += J8 * J8;
		pt += 4;
		*pt += J9 * J8;
		pt += 4;
		*pt += J10 * J8;
		pt += 4;
		*pt += J11 * J8;
		pt += 4;
		*pt += J12 * J8;
		pt += 4;
		*pt += J13 * J8;
		pt += 4;

		*pt += J9 * J9;
		pt += 4;
		*pt += J10 * J9;
		pt += 4;
		*pt += J11 * J9;
		pt += 4;
		*pt += J12 * J9;
		pt += 4;
		*pt += J13 * J9;
		pt += 4;

		*pt += J10 * J10;
		pt += 4;
		*pt += J11 * J10;
		pt += 4;
		*pt += J12 * J10;
		pt += 4;
		*pt += J13 * J10;
		pt += 4;

		*pt += J11 * J11;
		pt += 4;
		*pt += J12 * J11;
		pt += 4;
		*pt += J13 * J11;
		pt += 4;

		*pt += J12 * J12;
		pt += 4;
		*pt += J13 * J12;
		pt += 4;

		*pt += J13 * J13;
		pt += 4;

		num++;
	}

private:
	EIGEN_ALIGN16 float SSEData[4 * 105];
};

class AccumulatorApprox {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;

	// 17x17
	MatPCPCf H;
	size_t num;

	inline void initialize() {
		memset(Data, 0, sizeof(float) * 108);
		memset(TopRight_Data, 0, sizeof(float) * 44);
		memset(BotRight_Data, 0, sizeof(float) * 8);
		num = 0;
	}

	inline void finish() {
		H.setZero();

		int idx = 0;
		for (int r = 0; r < 14; r++)
			for (int c = r; c < 14; c++) {
				H(r, c) = H(c, r) = Data[idx];
				idx++;
			}

		idx = 0;
		for (int r = 0; r < 14; r++)
			for (int c = 14; c < 17; c++) {
				H(r, c) = H(c, r) = TopRight_Data[idx];
				idx++;
			}

		H(14, 14) = BotRight_Data[0];
		H(14, 15) = H(15, 14) = BotRight_Data[1];
		H(14, 16) = H(16, 14) = BotRight_Data[2];
		H(15, 15) = BotRight_Data[3];
		H(15, 16) = H(16, 15) = BotRight_Data[4];
		H(16, 16) = BotRight_Data[5];
	}

	/*
	 * Params are how image x & y change with Camera intrisics(xC,yC) and pose (xX yX). And
	 * how the point (sum of squared pixel)residual change with image x & y squared(a,b,c).
	 *
	 * Accumulates dRes/(dCameraP, dPose)?
	 * 
	 * computes the outer sum of 14x2 matrices, weighted with a 2x2 matrix:
	 * 	[x y] * [a b; b c] * [x y]^T
	 * (assuming x,y are column-vectors).
	 * numerically robust to large sums.
	 */
	inline void update(const float *const xC, const float *const xX, const float *const yC, const float *const yX, const float a,
			const float b, const float c) {

		Data[0] += a * xC[0] * xC[0] + c * yC[0] * yC[0] + b * (xC[0] * yC[0] + yC[0] * xC[0]);
		Data[1] += a * xC[1] * xC[0] + c * yC[1] * yC[0] + b * (xC[1] * yC[0] + yC[1] * xC[0]);
		Data[2] += a * xC[2] * xC[0] + c * yC[2] * yC[0] + b * (xC[2] * yC[0] + yC[2] * xC[0]);
		Data[3] += a * xC[3] * xC[0] + c * yC[3] * yC[0] + b * (xC[3] * yC[0] + yC[3] * xC[0]);
		Data[4] += a * xC[4] * xC[0] + c * yC[4] * yC[0] + b * (xC[4] * yC[0] + yC[4] * xC[0]);
		Data[5] += a * xC[5] * xC[0] + c * yC[5] * yC[0] + b * (xC[5] * yC[0] + yC[5] * xC[0]);
		Data[6] += a * xC[6] * xC[0] + c * yC[6] * yC[0] + b * (xC[6] * yC[0] + yC[6] * xC[0]);
		Data[7] += a * xC[7] * xC[0] + c * yC[7] * yC[0] + b * (xC[7] * yC[0] + yC[7] * xC[0]);
		Data[8] += a * xX[0] * xC[0] + c * yX[0] * yC[0] + b * (xX[0] * yC[0] + yX[0] * xC[0]);
		Data[9] += a * xX[1] * xC[0] + c * yX[1] * yC[0] + b * (xX[1] * yC[0] + yX[1] * xC[0]);
		Data[10] += a * xX[2] * xC[0] + c * yX[2] * yC[0] + b * (xX[2] * yC[0] + yX[2] * xC[0]);
		Data[11] += a * xX[3] * xC[0] + c * yX[3] * yC[0] + b * (xX[3] * yC[0] + yX[3] * xC[0]);
		Data[12] += a * xX[4] * xC[0] + c * yX[4] * yC[0] + b * (xX[4] * yC[0] + yX[4] * xC[0]);
		Data[13] += a * xX[5] * xC[0] + c * yX[5] * yC[0] + b * (xX[5] * yC[0] + yX[5] * xC[0]);

		Data[14] += a * xC[1] * xC[1] + c * yC[1] * yC[1] + b * (xC[1] * yC[1] + yC[1] * xC[1]);
		Data[15] += a * xC[2] * xC[1] + c * yC[2] * yC[1] + b * (xC[2] * yC[1] + yC[2] * xC[1]);
		Data[16] += a * xC[3] * xC[1] + c * yC[3] * yC[1] + b * (xC[3] * yC[1] + yC[3] * xC[1]);
		Data[17] += a * xC[4] * xC[1] + c * yC[4] * yC[1] + b * (xC[4] * yC[1] + yC[4] * xC[1]);
		Data[18] += a * xC[5] * xC[1] + c * yC[5] * yC[1] + b * (xC[5] * yC[1] + yC[5] * xC[1]);
		Data[19] += a * xC[6] * xC[1] + c * yC[6] * yC[1] + b * (xC[6] * yC[1] + yC[6] * xC[1]);
		Data[20] += a * xC[7] * xC[1] + c * yC[7] * yC[1] + b * (xC[7] * yC[1] + yC[7] * xC[1]);
		Data[21] += a * xX[0] * xC[1] + c * yX[0] * yC[1] + b * (xX[0] * yC[1] + yX[0] * xC[1]);
		Data[22] += a * xX[1] * xC[1] + c * yX[1] * yC[1] + b * (xX[1] * yC[1] + yX[1] * xC[1]);
		Data[23] += a * xX[2] * xC[1] + c * yX[2] * yC[1] + b * (xX[2] * yC[1] + yX[2] * xC[1]);
		Data[24] += a * xX[3] * xC[1] + c * yX[3] * yC[1] + b * (xX[3] * yC[1] + yX[3] * xC[1]);
		Data[25] += a * xX[4] * xC[1] + c * yX[4] * yC[1] + b * (xX[4] * yC[1] + yX[4] * xC[1]);
		Data[26] += a * xX[5] * xC[1] + c * yX[5] * yC[1] + b * (xX[5] * yC[1] + yX[5] * xC[1]);

		Data[27] += a * xC[2] * xC[2] + c * yC[2] * yC[2] + b * (xC[2] * yC[2] + yC[2] * xC[2]);
		Data[28] += a * xC[3] * xC[2] + c * yC[3] * yC[2] + b * (xC[3] * yC[2] + yC[3] * xC[2]);
		Data[29] += a * xC[4] * xC[2] + c * yC[4] * yC[2] + b * (xC[4] * yC[2] + yC[4] * xC[2]);
		Data[30] += a * xC[5] * xC[2] + c * yC[5] * yC[2] + b * (xC[5] * yC[2] + yC[5] * xC[2]);
		Data[31] += a * xC[6] * xC[2] + c * yC[6] * yC[2] + b * (xC[6] * yC[2] + yC[6] * xC[2]);
		Data[32] += a * xC[7] * xC[2] + c * yC[7] * yC[2] + b * (xC[7] * yC[2] + yC[7] * xC[2]);
		Data[33] += a * xX[0] * xC[2] + c * yX[0] * yC[2] + b * (xX[0] * yC[2] + yX[0] * xC[2]);
		Data[34] += a * xX[1] * xC[2] + c * yX[1] * yC[2] + b * (xX[1] * yC[2] + yX[1] * xC[2]);
		Data[35] += a * xX[2] * xC[2] + c * yX[2] * yC[2] + b * (xX[2] * yC[2] + yX[2] * xC[2]);
		Data[36] += a * xX[3] * xC[2] + c * yX[3] * yC[2] + b * (xX[3] * yC[2] + yX[3] * xC[2]);
		Data[37] += a * xX[4] * xC[2] + c * yX[4] * yC[2] + b * (xX[4] * yC[2] + yX[4] * xC[2]);
		Data[38] += a * xX[5] * xC[2] + c * yX[5] * yC[2] + b * (xX[5] * yC[2] + yX[5] * xC[2]);

		Data[39] += a * xC[3] * xC[3] + c * yC[3] * yC[3] + b * (xC[3] * yC[3] + yC[3] * xC[3]);
		Data[40] += a * xC[4] * xC[3] + c * yC[4] * yC[3] + b * (xC[4] * yC[3] + yC[4] * xC[3]);
		Data[41] += a * xC[5] * xC[3] + c * yC[5] * yC[3] + b * (xC[5] * yC[3] + yC[5] * xC[3]);
		Data[42] += a * xC[6] * xC[3] + c * yC[6] * yC[3] + b * (xC[6] * yC[3] + yC[6] * xC[3]);
		Data[43] += a * xC[7] * xC[3] + c * yC[7] * yC[3] + b * (xC[7] * yC[3] + yC[7] * xC[3]);
		Data[44] += a * xX[0] * xC[3] + c * yX[0] * yC[3] + b * (xX[0] * yC[3] + yX[0] * xC[3]);
		Data[45] += a * xX[1] * xC[3] + c * yX[1] * yC[3] + b * (xX[1] * yC[3] + yX[1] * xC[3]);
		Data[46] += a * xX[2] * xC[3] + c * yX[2] * yC[3] + b * (xX[2] * yC[3] + yX[2] * xC[3]);
		Data[47] += a * xX[3] * xC[3] + c * yX[3] * yC[3] + b * (xX[3] * yC[3] + yX[3] * xC[3]);
		Data[48] += a * xX[4] * xC[3] + c * yX[4] * yC[3] + b * (xX[4] * yC[3] + yX[4] * xC[3]);
		Data[49] += a * xX[5] * xC[3] + c * yX[5] * yC[3] + b * (xX[5] * yC[3] + yX[5] * xC[3]);

		Data[50] += a * xC[4] * xC[4] + c * yC[4] * yC[4] + b * (xC[4] * yC[4] + yC[4] * xC[4]);
		Data[51] += a * xC[5] * xC[4] + c * yC[5] * yC[4] + b * (xC[5] * yC[4] + yC[5] * xC[4]);
		Data[52] += a * xC[6] * xC[4] + c * yC[6] * yC[4] + b * (xC[6] * yC[4] + yC[6] * xC[4]);
		Data[53] += a * xC[7] * xC[4] + c * yC[7] * yC[4] + b * (xC[7] * yC[4] + yC[7] * xC[4]);
		Data[54] += a * xX[0] * xC[4] + c * yX[0] * yC[4] + b * (xX[0] * yC[4] + yX[0] * xC[4]);
		Data[55] += a * xX[1] * xC[4] + c * yX[1] * yC[4] + b * (xX[1] * yC[4] + yX[1] * xC[4]);
		Data[56] += a * xX[2] * xC[4] + c * yX[2] * yC[4] + b * (xX[2] * yC[4] + yX[2] * xC[4]);
		Data[57] += a * xX[3] * xC[4] + c * yX[3] * yC[4] + b * (xX[3] * yC[4] + yX[3] * xC[4]);
		Data[58] += a * xX[4] * xC[4] + c * yX[4] * yC[4] + b * (xX[4] * yC[4] + yX[4] * xC[4]);
		Data[59] += a * xX[5] * xC[4] + c * yX[5] * yC[4] + b * (xX[5] * yC[4] + yX[5] * xC[4]);

		Data[60] += a * xC[5] * xC[5] + c * yC[5] * yC[5] + b * (xC[5] * yC[5] + yC[5] * xC[5]);
		Data[61] += a * xC[6] * xC[5] + c * yC[6] * yC[5] + b * (xC[6] * yC[5] + yC[6] * xC[5]);
		Data[62] += a * xC[7] * xC[5] + c * yC[7] * yC[5] + b * (xC[7] * yC[5] + yC[7] * xC[5]);
		Data[63] += a * xX[0] * xC[5] + c * yX[0] * yC[5] + b * (xX[0] * yC[5] + yX[0] * xC[5]);
		Data[64] += a * xX[1] * xC[5] + c * yX[1] * yC[5] + b * (xX[1] * yC[5] + yX[1] * xC[5]);
		Data[65] += a * xX[2] * xC[5] + c * yX[2] * yC[5] + b * (xX[2] * yC[5] + yX[2] * xC[5]);
		Data[66] += a * xX[3] * xC[5] + c * yX[3] * yC[5] + b * (xX[3] * yC[5] + yX[3] * xC[5]);
		Data[67] += a * xX[4] * xC[5] + c * yX[4] * yC[5] + b * (xX[4] * yC[5] + yX[4] * xC[5]);
		Data[68] += a * xX[5] * xC[5] + c * yX[5] * yC[5] + b * (xX[5] * yC[5] + yX[5] * xC[5]);

		Data[69] += a * xC[6] * xC[6] + c * yC[6] * yC[6] + b * (xC[6] * yC[6] + yC[6] * xC[6]);
		Data[70] += a * xC[7] * xC[6] + c * yC[7] * yC[6] + b * (xC[7] * yC[6] + yC[7] * xC[6]);
		Data[71] += a * xX[0] * xC[6] + c * yX[0] * yC[6] + b * (xX[0] * yC[6] + yX[0] * xC[6]);
		Data[72] += a * xX[1] * xC[6] + c * yX[1] * yC[6] + b * (xX[1] * yC[6] + yX[1] * xC[6]);
		Data[73] += a * xX[2] * xC[6] + c * yX[2] * yC[6] + b * (xX[2] * yC[6] + yX[2] * xC[6]);
		Data[74] += a * xX[3] * xC[6] + c * yX[3] * yC[6] + b * (xX[3] * yC[6] + yX[3] * xC[6]);
		Data[75] += a * xX[4] * xC[6] + c * yX[4] * yC[6] + b * (xX[4] * yC[6] + yX[4] * xC[6]);
		Data[76] += a * xX[5] * xC[6] + c * yX[5] * yC[6] + b * (xX[5] * yC[6] + yX[5] * xC[6]);

		Data[77] += a * xC[7] * xC[7] + c * yC[7] * yC[7] + b * (xC[7] * yC[7] + yC[7] * xC[7]);
		Data[78] += a * xX[0] * xC[7] + c * yX[0] * yC[7] + b * (xX[0] * yC[7] + yX[0] * xC[7]);
		Data[79] += a * xX[1] * xC[7] + c * yX[1] * yC[7] + b * (xX[1] * yC[7] + yX[1] * xC[7]);
		Data[80] += a * xX[2] * xC[7] + c * yX[2] * yC[7] + b * (xX[2] * yC[7] + yX[2] * xC[7]);
		Data[81] += a * xX[3] * xC[7] + c * yX[3] * yC[7] + b * (xX[3] * yC[7] + yX[3] * xC[7]);
		Data[82] += a * xX[4] * xC[7] + c * yX[4] * yC[7] + b * (xX[4] * yC[7] + yX[4] * xC[7]);
		Data[83] += a * xX[5] * xC[7] + c * yX[5] * yC[7] + b * (xX[5] * yC[7] + yX[5] * xC[7]);

		Data[84] += a * xX[0] * xX[0] + c * yX[0] * yX[0] + b * (xX[0] * yX[0] + yX[0] * xX[0]);
		Data[85] += a * xX[1] * xX[0] + c * yX[1] * yX[0] + b * (xX[1] * yX[0] + yX[1] * xX[0]);
		Data[86] += a * xX[2] * xX[0] + c * yX[2] * yX[0] + b * (xX[2] * yX[0] + yX[2] * xX[0]);
		Data[87] += a * xX[3] * xX[0] + c * yX[3] * yX[0] + b * (xX[3] * yX[0] + yX[3] * xX[0]);
		Data[88] += a * xX[4] * xX[0] + c * yX[4] * yX[0] + b * (xX[4] * yX[0] + yX[4] * xX[0]);
		Data[89] += a * xX[5] * xX[0] + c * yX[5] * yX[0] + b * (xX[5] * yX[0] + yX[5] * xX[0]);

		Data[90] += a * xX[1] * xX[1] + c * yX[1] * yX[1] + b * (xX[1] * yX[1] + yX[1] * xX[1]);
		Data[91] += a * xX[2] * xX[1] + c * yX[2] * yX[1] + b * (xX[2] * yX[1] + yX[2] * xX[1]);
		Data[92] += a * xX[3] * xX[1] + c * yX[3] * yX[1] + b * (xX[3] * yX[1] + yX[3] * xX[1]);
		Data[93] += a * xX[4] * xX[1] + c * yX[4] * yX[1] + b * (xX[4] * yX[1] + yX[4] * xX[1]);
		Data[94] += a * xX[5] * xX[1] + c * yX[5] * yX[1] + b * (xX[5] * yX[1] + yX[5] * xX[1]);

		Data[95] += a * xX[2] * xX[2] + c * yX[2] * yX[2] + b * (xX[2] * yX[2] + yX[2] * xX[2]);
		Data[96] += a * xX[3] * xX[2] + c * yX[3] * yX[2] + b * (xX[3] * yX[2] + yX[3] * xX[2]);
		Data[97] += a * xX[4] * xX[2] + c * yX[4] * yX[2] + b * (xX[4] * yX[2] + yX[4] * xX[2]);
		Data[98] += a * xX[5] * xX[2] + c * yX[5] * yX[2] + b * (xX[5] * yX[2] + yX[5] * xX[2]);

		Data[99] += a * xX[3] * xX[3] + c * yX[3] * yX[3] + b * (xX[3] * yX[3] + yX[3] * xX[3]);
		Data[100] += a * xX[4] * xX[3] + c * yX[4] * yX[3] + b * (xX[4] * yX[3] + yX[4] * xX[3]);
		Data[101] += a * xX[5] * xX[3] + c * yX[5] * yX[3] + b * (xX[5] * yX[3] + yX[5] * xX[3]);

		Data[102] += a * xX[4] * xX[4] + c * yX[4] * yX[4] + b * (xX[4] * yX[4] + yX[4] * xX[4]);
		Data[103] += a * xX[5] * xX[4] + c * yX[5] * yX[4] + b * (xX[5] * yX[4] + yX[5] * xX[4]);

		Data[104] += a * xX[5] * xX[5] + c * yX[5] * yX[5] + b * (xX[5] * yX[5] + yX[5] * xX[5]);

		num++;
	}

	inline void updateTopRight(const float *const xC, const float *const xX, const float *const yC, const float *const yX,
			const float TR00, const float TR10, const float TR01, const float TR11, const float TR02, const float TR12) {
		TopRight_Data[0] += xC[0] * TR00 + yC[0] * TR10;
		TopRight_Data[1] += xC[0] * TR01 + yC[0] * TR11;
		TopRight_Data[2] += xC[0] * TR02 + yC[0] * TR12;

		TopRight_Data[3] += xC[1] * TR00 + yC[1] * TR10;
		TopRight_Data[4] += xC[1] * TR01 + yC[1] * TR11;
		TopRight_Data[5] += xC[1] * TR02 + yC[1] * TR12;

		TopRight_Data[6] += xC[2] * TR00 + yC[2] * TR10;
		TopRight_Data[7] += xC[2] * TR01 + yC[2] * TR11;
		TopRight_Data[8] += xC[2] * TR02 + yC[2] * TR12;

		TopRight_Data[9] += xC[3] * TR00 + yC[3] * TR10;
		TopRight_Data[10] += xC[3] * TR01 + yC[3] * TR11;
		TopRight_Data[11] += xC[3] * TR02 + yC[3] * TR12;

		TopRight_Data[12] += xC[4] * TR00 + yC[4] * TR10;
		TopRight_Data[13] += xC[4] * TR01 + yC[4] * TR11;
		TopRight_Data[14] += xC[4] * TR02 + yC[4] * TR12;

		TopRight_Data[15] += xC[5] * TR00 + yC[5] * TR10;
		TopRight_Data[16] += xC[5] * TR01 + yC[5] * TR11;
		TopRight_Data[17] += xC[5] * TR02 + yC[5] * TR12;

		TopRight_Data[18] += xC[6] * TR00 + yC[6] * TR10;
		TopRight_Data[19] += xC[6] * TR01 + yC[6] * TR11;
		TopRight_Data[20] += xC[6] * TR02 + yC[6] * TR12;

		TopRight_Data[21] += xC[7] * TR00 + yC[7] * TR10;
		TopRight_Data[22] += xC[7] * TR01 + yC[7] * TR11;
		TopRight_Data[23] += xC[7] * TR02 + yC[7] * TR12;

		TopRight_Data[24] += xX[0] * TR00 + yX[0] * TR10;
		TopRight_Data[25] += xX[0] * TR01 + yX[0] * TR11;
		TopRight_Data[26] += xX[0] * TR02 + yX[0] * TR12;

		TopRight_Data[27] += xX[1] * TR00 + yX[1] * TR10;
		TopRight_Data[28] += xX[1] * TR01 + yX[1] * TR11;
		TopRight_Data[29] += xX[1] * TR02 + yX[1] * TR12;

		TopRight_Data[30] += xX[2] * TR00 + yX[2] * TR10;
		TopRight_Data[31] += xX[2] * TR01 + yX[2] * TR11;
		TopRight_Data[32] += xX[2] * TR02 + yX[2] * TR12;

		TopRight_Data[33] += xX[3] * TR00 + yX[3] * TR10;
		TopRight_Data[34] += xX[3] * TR01 + yX[3] * TR11;
		TopRight_Data[35] += xX[3] * TR02 + yX[3] * TR12;

		TopRight_Data[36] += xX[4] * TR00 + yX[4] * TR10;
		TopRight_Data[37] += xX[4] * TR01 + yX[4] * TR11;
		TopRight_Data[38] += xX[4] * TR02 + yX[4] * TR12;

		TopRight_Data[39] += xX[5] * TR00 + yX[5] * TR10;
		TopRight_Data[40] += xX[5] * TR01 + yX[5] * TR11;
		TopRight_Data[41] += xX[5] * TR02 + yX[5] * TR12;

	}

	// ...why a 3x3 upper diagonal matrix of affine brightness AB?!
	inline void updateBotRight(const float a00, const float a01, const float a02, const float a11, const float a12,
			const float a22) {
		BotRight_Data[0] += a00;
		BotRight_Data[1] += a01;
		BotRight_Data[2] += a02;
		BotRight_Data[3] += a11;
		BotRight_Data[4] += a12;
		BotRight_Data[5] += a22;
	}

private:
	EIGEN_ALIGN16 float Data[108];  // 105 are used, but is divisible by 4 (4*4=16) for word alignment.
	EIGEN_ALIGN16 float TopRight_Data[44]; // 42 are used but divisible by 4..
	EIGEN_ALIGN16 float BotRight_Data[8]; // 6 are used...
};

class Accumulator9 {
public:EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;

	Mat99f H;
	size_t num;

	inline void initialize() {
		H.setZero();
		memset(SSEData, 0, sizeof(float) * 4 * 45);
		num = 0;
	}

	inline void finish() {
		H.setZero();

		int idx = 0;
		for (int r = 0; r < 9; r++)
			for (int c = r; c < 9; c++) {
				float d = SSEData[idx + 0] + SSEData[idx + 1] + SSEData[idx + 2] + SSEData[idx + 3];
				H(r, c) = H(c, r) = d;
				idx += 4;
			}
		assert(idx == 4 * 45);
	}

	inline void updateSSE(const __m128 J0, const __m128 J1, const __m128 J2, const __m128 J3, const __m128 J4, const __m128 J5,
			const __m128 J6, const __m128 J7, const __m128 J8) {
		float *pt = SSEData;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J0)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J1)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J2)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J3)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J8)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J1)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J2)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J3)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J8)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J2)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J3)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J8)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J3)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J8)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J8)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J8)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J8)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J8)));
		pt += 4;

		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J8)));
		pt += 4;

		num += 4;
	}

	inline void updateSSE_eighted(const __m128 J0, const __m128 J1, const __m128 J2, const __m128 J3, const __m128 J4,
			const __m128 J5, const __m128 J6, const __m128 J7, const __m128 J8, const __m128 w) {
		float *pt = SSEData;

		__m128 J0w = _mm_mul_ps(J0, w);
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J0)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J1)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J2)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J3)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J8)));
		pt += 4;

		__m128 J1w = _mm_mul_ps(J1, w);
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J1)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J2)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J3)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J8)));
		pt += 4;

		__m128 J2w = _mm_mul_ps(J2, w);
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J2)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J3)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J8)));
		pt += 4;

		__m128 J3w = _mm_mul_ps(J3, w);
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J3)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J8)));
		pt += 4;

		__m128 J4w = _mm_mul_ps(J4, w);
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J4)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J8)));
		pt += 4;

		__m128 J5w = _mm_mul_ps(J5, w);
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J5)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J8)));
		pt += 4;

		__m128 J6w = _mm_mul_ps(J6, w);
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6w, J6)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6w, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6w, J8)));
		pt += 4;

		__m128 J7w = _mm_mul_ps(J7, w);
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7w, J7)));
		pt += 4;
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7w, J8)));
		pt += 4;

		__m128 J8w = _mm_mul_ps(J8, w);
		_mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8w, J8)));
		pt += 4;

		num += 4;
	}

	inline void updateSingle(const float J0, const float J1, const float J2, const float J3, const float J4, const float J5,
			const float J6, const float J7, const float J8, int off = 0) {
		float *pt = SSEData + off;
		*pt += J0 * J0;
		pt += 4;
		*pt += J1 * J0;
		pt += 4;
		*pt += J2 * J0;
		pt += 4;
		*pt += J3 * J0;
		pt += 4;
		*pt += J4 * J0;
		pt += 4;
		*pt += J5 * J0;
		pt += 4;
		*pt += J6 * J0;
		pt += 4;
		*pt += J7 * J0;
		pt += 4;
		*pt += J8 * J0;
		pt += 4;

		*pt += J1 * J1;
		pt += 4;
		*pt += J2 * J1;
		pt += 4;
		*pt += J3 * J1;
		pt += 4;
		*pt += J4 * J1;
		pt += 4;
		*pt += J5 * J1;
		pt += 4;
		*pt += J6 * J1;
		pt += 4;
		*pt += J7 * J1;
		pt += 4;
		*pt += J8 * J1;
		pt += 4;

		*pt += J2 * J2;
		pt += 4;
		*pt += J3 * J2;
		pt += 4;
		*pt += J4 * J2;
		pt += 4;
		*pt += J5 * J2;
		pt += 4;
		*pt += J6 * J2;
		pt += 4;
		*pt += J7 * J2;
		pt += 4;
		*pt += J8 * J2;
		pt += 4;

		*pt += J3 * J3;
		pt += 4;
		*pt += J4 * J3;
		pt += 4;
		*pt += J5 * J3;
		pt += 4;
		*pt += J6 * J3;
		pt += 4;
		*pt += J7 * J3;
		pt += 4;
		*pt += J8 * J3;
		pt += 4;

		*pt += J4 * J4;
		pt += 4;
		*pt += J5 * J4;
		pt += 4;
		*pt += J6 * J4;
		pt += 4;
		*pt += J7 * J4;
		pt += 4;
		*pt += J8 * J4;
		pt += 4;

		*pt += J5 * J5;
		pt += 4;
		*pt += J6 * J5;
		pt += 4;
		*pt += J7 * J5;
		pt += 4;
		*pt += J8 * J5;
		pt += 4;

		*pt += J6 * J6;
		pt += 4;
		*pt += J7 * J6;
		pt += 4;
		*pt += J8 * J6;
		pt += 4;

		*pt += J7 * J7;
		pt += 4;
		*pt += J8 * J7;
		pt += 4;

		*pt += J8 * J8;
		pt += 4;

		num++;
	}

	inline void updateSingleWeighted(float J0, float J1, float J2, float J3, float J4, float J5, float J6, float J7, float J8,
			float w, int off = 0) {

		float *pt = SSEData + off;
		*pt += J0 * J0 * w;
		pt += 4;
		J0 *= w;
		*pt += J1 * J0;
		pt += 4;
		*pt += J2 * J0;
		pt += 4;
		*pt += J3 * J0;
		pt += 4;
		*pt += J4 * J0;
		pt += 4;
		*pt += J5 * J0;
		pt += 4;
		*pt += J6 * J0;
		pt += 4;
		*pt += J7 * J0;
		pt += 4;
		*pt += J8 * J0;
		pt += 4;

		*pt += J1 * J1 * w;
		pt += 4;
		J1 *= w;
		*pt += J2 * J1;
		pt += 4;
		*pt += J3 * J1;
		pt += 4;
		*pt += J4 * J1;
		pt += 4;
		*pt += J5 * J1;
		pt += 4;
		*pt += J6 * J1;
		pt += 4;
		*pt += J7 * J1;
		pt += 4;
		*pt += J8 * J1;
		pt += 4;

		*pt += J2 * J2 * w;
		pt += 4;
		J2 *= w;
		*pt += J3 * J2;
		pt += 4;
		*pt += J4 * J2;
		pt += 4;
		*pt += J5 * J2;
		pt += 4;
		*pt += J6 * J2;
		pt += 4;
		*pt += J7 * J2;
		pt += 4;
		*pt += J8 * J2;
		pt += 4;

		*pt += J3 * J3 * w;
		pt += 4;
		J3 *= w;
		*pt += J4 * J3;
		pt += 4;
		*pt += J5 * J3;
		pt += 4;
		*pt += J6 * J3;
		pt += 4;
		*pt += J7 * J3;
		pt += 4;
		*pt += J8 * J3;
		pt += 4;

		*pt += J4 * J4 * w;
		pt += 4;
		J4 *= w;
		*pt += J5 * J4;
		pt += 4;
		*pt += J6 * J4;
		pt += 4;
		*pt += J7 * J4;
		pt += 4;
		*pt += J8 * J4;
		pt += 4;

		*pt += J5 * J5 * w;
		pt += 4;
		J5 *= w;
		*pt += J6 * J5;
		pt += 4;
		*pt += J7 * J5;
		pt += 4;
		*pt += J8 * J5;
		pt += 4;

		*pt += J6 * J6 * w;
		pt += 4;
		J6 *= w;
		*pt += J7 * J6;
		pt += 4;
		*pt += J8 * J6;
		pt += 4;

		*pt += J7 * J7 * w;
		pt += 4;
		J7 *= w;
		*pt += J8 * J7;
		pt += 4;

		*pt += J8 * J8 * w;
		pt += 4;

		num++;
	}

private:
	EIGEN_ALIGN16 float SSEData[4 * 45];

};
}
