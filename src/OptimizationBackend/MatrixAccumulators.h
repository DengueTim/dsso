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
		assert(idx==4*105);
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

	Mat2323f H;
	size_t num;

	inline void initialize() {
		memset(Data, 0, sizeof(float) * 212);
		memset(TopRight_Data, 0, sizeof(float) * 60);
		memset(BotRight_Data, 0, sizeof(float) * 8);
		num = 0;
	}

	inline void finish() {
		H.setZero();

		int idx = 0;
		for (int r = 0; r < 20; r++)
			for (int c = r; c < 20; c++) {
				H(r, c) = H(c, r) = Data[idx];
				idx++;
			}

		idx = 0;
		for (int r = 0; r < 20; r++)
			for (int c = 0; c < 3; c++) {
				H(r, c + 20) = H(c + 20, r) = TopRight_Data[idx];
				idx++;
			}

		H(20, 20) = BotRight_Data[0];
		H(20, 21) = H(21, 20) = BotRight_Data[1];
		H(20, 22) = H(22, 20) = BotRight_Data[2];
		H(21, 21) = BotRight_Data[3];
		H(21, 22) = H(22, 21) = BotRight_Data[4];
		H(22, 22) = BotRight_Data[5];
	}

	/*
	 * Params are how image x & y change with Camera intrisics(xC,yC) and pose (xX yX). And
	 * how the point (sum of squared pixel)residual change with image x & y squared(a,b,c).
	 *
	 * Accumulates dRes/(dCameraP, dPose)?
	 * 
	 * computes the outer sum of 20x2 matrices, weighted with a 2x2 matrix:
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
		Data[8] += a * xC[8] * xC[0] + c * yC[8] * yC[0] + b * (xC[8] * yC[0] + yC[8] * xC[0]);
		Data[9] += a * xC[9] * xC[0] + c * yC[9] * yC[0] + b * (xC[9] * yC[0] + yC[9] * xC[0]);
		Data[10] += a * xC[10] * xC[0] + c * yC[10] * yC[0] + b * (xC[10] * yC[0] + yC[10] * xC[0]);
		Data[11] += a * xC[11] * xC[0] + c * yC[11] * yC[0] + b * (xC[11] * yC[0] + yC[11] * xC[0]);
		Data[12] += a * xC[12] * xC[0] + c * yC[12] * yC[0] + b * (xC[12] * yC[0] + yC[12] * xC[0]);
		Data[13] += a * xC[13] * xC[0] + c * yC[13] * yC[0] + b * (xC[13] * yC[0] + yC[13] * xC[0]);
		Data[14] += a * xX[0] * xC[0] + c * yX[0] * yC[0] + b * (xX[0] * yC[0] + yX[0] * xC[0]);
		Data[15] += a * xX[1] * xC[0] + c * yX[1] * yC[0] + b * (xX[1] * yC[0] + yX[1] * xC[0]);
		Data[16] += a * xX[2] * xC[0] + c * yX[2] * yC[0] + b * (xX[2] * yC[0] + yX[2] * xC[0]);
		Data[17] += a * xX[3] * xC[0] + c * yX[3] * yC[0] + b * (xX[3] * yC[0] + yX[3] * xC[0]);
		Data[18] += a * xX[4] * xC[0] + c * yX[4] * yC[0] + b * (xX[4] * yC[0] + yX[4] * xC[0]);
		Data[19] += a * xX[5] * xC[0] + c * yX[5] * yC[0] + b * (xX[5] * yC[0] + yX[5] * xC[0]);

		Data[20] += a * xC[1] * xC[1] + c * yC[1] * yC[1] + b * (xC[1] * yC[1] + yC[1] * xC[1]);
		Data[21] += a * xC[2] * xC[1] + c * yC[2] * yC[1] + b * (xC[2] * yC[1] + yC[2] * xC[1]);
		Data[22] += a * xC[3] * xC[1] + c * yC[3] * yC[1] + b * (xC[3] * yC[1] + yC[3] * xC[1]);
		Data[23] += a * xC[4] * xC[1] + c * yC[4] * yC[1] + b * (xC[4] * yC[1] + yC[4] * xC[1]);
		Data[24] += a * xC[5] * xC[1] + c * yC[5] * yC[1] + b * (xC[5] * yC[1] + yC[5] * xC[1]);
		Data[25] += a * xC[6] * xC[1] + c * yC[6] * yC[1] + b * (xC[6] * yC[1] + yC[6] * xC[1]);
		Data[26] += a * xC[7] * xC[1] + c * yC[7] * yC[1] + b * (xC[7] * yC[1] + yC[7] * xC[1]);
		Data[27] += a * xC[8] * xC[1] + c * yC[8] * yC[1] + b * (xC[8] * yC[1] + yC[8] * xC[1]);
		Data[28] += a * xC[9] * xC[1] + c * yC[9] * yC[1] + b * (xC[9] * yC[1] + yC[9] * xC[1]);
		Data[29] += a * xC[10] * xC[1] + c * yC[10] * yC[1] + b * (xC[10] * yC[1] + yC[10] * xC[1]);
		Data[30] += a * xC[11] * xC[1] + c * yC[11] * yC[1] + b * (xC[11] * yC[1] + yC[11] * xC[1]);
		Data[31] += a * xC[12] * xC[1] + c * yC[12] * yC[1] + b * (xC[12] * yC[1] + yC[12] * xC[1]);
		Data[32] += a * xC[13] * xC[1] + c * yC[13] * yC[1] + b * (xC[13] * yC[1] + yC[13] * xC[1]);
		Data[33] += a * xX[0] * xC[1] + c * yX[0] * yC[1] + b * (xX[0] * yC[1] + yX[0] * xC[1]);
		Data[34] += a * xX[1] * xC[1] + c * yX[1] * yC[1] + b * (xX[1] * yC[1] + yX[1] * xC[1]);
		Data[35] += a * xX[2] * xC[1] + c * yX[2] * yC[1] + b * (xX[2] * yC[1] + yX[2] * xC[1]);
		Data[36] += a * xX[3] * xC[1] + c * yX[3] * yC[1] + b * (xX[3] * yC[1] + yX[3] * xC[1]);
		Data[37] += a * xX[4] * xC[1] + c * yX[4] * yC[1] + b * (xX[4] * yC[1] + yX[4] * xC[1]);
		Data[38] += a * xX[5] * xC[1] + c * yX[5] * yC[1] + b * (xX[5] * yC[1] + yX[5] * xC[1]);

		Data[39] += a * xC[2] * xC[2] + c * yC[2] * yC[2] + b * (xC[2] * yC[2] + yC[2] * xC[2]);
		Data[40] += a * xC[3] * xC[2] + c * yC[3] * yC[2] + b * (xC[3] * yC[2] + yC[3] * xC[2]);
		Data[41] += a * xC[4] * xC[2] + c * yC[4] * yC[2] + b * (xC[4] * yC[2] + yC[4] * xC[2]);
		Data[42] += a * xC[5] * xC[2] + c * yC[5] * yC[2] + b * (xC[5] * yC[2] + yC[5] * xC[2]);
		Data[43] += a * xC[6] * xC[2] + c * yC[6] * yC[2] + b * (xC[6] * yC[2] + yC[6] * xC[2]);
		Data[44] += a * xC[7] * xC[2] + c * yC[7] * yC[2] + b * (xC[7] * yC[2] + yC[7] * xC[2]);
		Data[45] += a * xC[8] * xC[2] + c * yC[8] * yC[2] + b * (xC[8] * yC[2] + yC[8] * xC[2]);
		Data[46] += a * xC[9] * xC[2] + c * yC[9] * yC[2] + b * (xC[9] * yC[2] + yC[9] * xC[2]);
		Data[47] += a * xC[10] * xC[2] + c * yC[10] * yC[2] + b * (xC[10] * yC[2] + yC[10] * xC[2]);
		Data[48] += a * xC[11] * xC[2] + c * yC[11] * yC[2] + b * (xC[11] * yC[2] + yC[11] * xC[2]);
		Data[49] += a * xC[12] * xC[2] + c * yC[12] * yC[2] + b * (xC[12] * yC[2] + yC[12] * xC[2]);
		Data[50] += a * xC[13] * xC[2] + c * yC[13] * yC[2] + b * (xC[13] * yC[2] + yC[13] * xC[2]);
		Data[51] += a * xX[0] * xC[2] + c * yX[0] * yC[2] + b * (xX[0] * yC[2] + yX[0] * xC[2]);
		Data[52] += a * xX[1] * xC[2] + c * yX[1] * yC[2] + b * (xX[1] * yC[2] + yX[1] * xC[2]);
		Data[53] += a * xX[2] * xC[2] + c * yX[2] * yC[2] + b * (xX[2] * yC[2] + yX[2] * xC[2]);
		Data[54] += a * xX[3] * xC[2] + c * yX[3] * yC[2] + b * (xX[3] * yC[2] + yX[3] * xC[2]);
		Data[55] += a * xX[4] * xC[2] + c * yX[4] * yC[2] + b * (xX[4] * yC[2] + yX[4] * xC[2]);
		Data[56] += a * xX[5] * xC[2] + c * yX[5] * yC[2] + b * (xX[5] * yC[2] + yX[5] * xC[2]);

		Data[57] += a * xC[3] * xC[3] + c * yC[3] * yC[3] + b * (xC[3] * yC[3] + yC[3] * xC[3]);
		Data[58] += a * xC[4] * xC[3] + c * yC[4] * yC[3] + b * (xC[4] * yC[3] + yC[4] * xC[3]);
		Data[59] += a * xC[5] * xC[3] + c * yC[5] * yC[3] + b * (xC[5] * yC[3] + yC[5] * xC[3]);
		Data[60] += a * xC[6] * xC[3] + c * yC[6] * yC[3] + b * (xC[6] * yC[3] + yC[6] * xC[3]);
		Data[61] += a * xC[7] * xC[3] + c * yC[7] * yC[3] + b * (xC[7] * yC[3] + yC[7] * xC[3]);
		Data[62] += a * xC[8] * xC[3] + c * yC[8] * yC[3] + b * (xC[8] * yC[3] + yC[8] * xC[3]);
		Data[63] += a * xC[9] * xC[3] + c * yC[9] * yC[3] + b * (xC[9] * yC[3] + yC[9] * xC[3]);
		Data[64] += a * xC[10] * xC[3] + c * yC[10] * yC[3] + b * (xC[10] * yC[3] + yC[10] * xC[3]);
		Data[65] += a * xC[11] * xC[3] + c * yC[11] * yC[3] + b * (xC[11] * yC[3] + yC[11] * xC[3]);
		Data[66] += a * xC[12] * xC[3] + c * yC[12] * yC[3] + b * (xC[12] * yC[3] + yC[12] * xC[3]);
		Data[67] += a * xC[13] * xC[3] + c * yC[13] * yC[3] + b * (xC[13] * yC[3] + yC[13] * xC[3]);
		Data[68] += a * xX[0] * xC[3] + c * yX[0] * yC[3] + b * (xX[0] * yC[3] + yX[0] * xC[3]);
		Data[69] += a * xX[1] * xC[3] + c * yX[1] * yC[3] + b * (xX[1] * yC[3] + yX[1] * xC[3]);
		Data[70] += a * xX[2] * xC[3] + c * yX[2] * yC[3] + b * (xX[2] * yC[3] + yX[2] * xC[3]);
		Data[71] += a * xX[3] * xC[3] + c * yX[3] * yC[3] + b * (xX[3] * yC[3] + yX[3] * xC[3]);
		Data[72] += a * xX[4] * xC[3] + c * yX[4] * yC[3] + b * (xX[4] * yC[3] + yX[4] * xC[3]);
		Data[73] += a * xX[5] * xC[3] + c * yX[5] * yC[3] + b * (xX[5] * yC[3] + yX[5] * xC[3]);

		Data[74] += a * xC[4] * xC[4] + c * yC[4] * yC[4] + b * (xC[4] * yC[4] + yC[4] * xC[4]);
		Data[75] += a * xC[5] * xC[4] + c * yC[5] * yC[4] + b * (xC[5] * yC[4] + yC[5] * xC[4]);
		Data[76] += a * xC[6] * xC[4] + c * yC[6] * yC[4] + b * (xC[6] * yC[4] + yC[6] * xC[4]);
		Data[77] += a * xC[7] * xC[4] + c * yC[7] * yC[4] + b * (xC[7] * yC[4] + yC[7] * xC[4]);
		Data[78] += a * xC[8] * xC[4] + c * yC[8] * yC[4] + b * (xC[8] * yC[4] + yC[8] * xC[4]);
		Data[79] += a * xC[9] * xC[4] + c * yC[9] * yC[4] + b * (xC[9] * yC[4] + yC[9] * xC[4]);
		Data[80] += a * xC[10] * xC[4] + c * yC[10] * yC[4] + b * (xC[10] * yC[4] + yC[10] * xC[4]);
		Data[81] += a * xC[11] * xC[4] + c * yC[11] * yC[4] + b * (xC[11] * yC[4] + yC[11] * xC[4]);
		Data[82] += a * xC[12] * xC[4] + c * yC[12] * yC[4] + b * (xC[12] * yC[4] + yC[12] * xC[4]);
		Data[83] += a * xC[13] * xC[4] + c * yC[13] * yC[4] + b * (xC[13] * yC[4] + yC[13] * xC[4]);
		Data[84] += a * xX[0] * xC[4] + c * yX[0] * yC[4] + b * (xX[0] * yC[4] + yX[0] * xC[4]);
		Data[85] += a * xX[1] * xC[4] + c * yX[1] * yC[4] + b * (xX[1] * yC[4] + yX[1] * xC[4]);
		Data[86] += a * xX[2] * xC[4] + c * yX[2] * yC[4] + b * (xX[2] * yC[4] + yX[2] * xC[4]);
		Data[87] += a * xX[3] * xC[4] + c * yX[3] * yC[4] + b * (xX[3] * yC[4] + yX[3] * xC[4]);
		Data[88] += a * xX[4] * xC[4] + c * yX[4] * yC[4] + b * (xX[4] * yC[4] + yX[4] * xC[4]);
		Data[89] += a * xX[5] * xC[4] + c * yX[5] * yC[4] + b * (xX[5] * yC[4] + yX[5] * xC[4]);

		Data[90] += a * xC[5] * xC[5] + c * yC[5] * yC[5] + b * (xC[5] * yC[5] + yC[5] * xC[5]);
		Data[91] += a * xC[6] * xC[5] + c * yC[6] * yC[5] + b * (xC[6] * yC[5] + yC[6] * xC[5]);
		Data[92] += a * xC[7] * xC[5] + c * yC[7] * yC[5] + b * (xC[7] * yC[5] + yC[7] * xC[5]);
		Data[93] += a * xC[8] * xC[5] + c * yC[8] * yC[5] + b * (xC[8] * yC[5] + yC[8] * xC[5]);
		Data[94] += a * xC[9] * xC[5] + c * yC[9] * yC[5] + b * (xC[9] * yC[5] + yC[9] * xC[5]);
		Data[95] += a * xC[10] * xC[5] + c * yC[10] * yC[5] + b * (xC[10] * yC[5] + yC[10] * xC[5]);
		Data[96] += a * xC[11] * xC[5] + c * yC[11] * yC[5] + b * (xC[11] * yC[5] + yC[11] * xC[5]);
		Data[97] += a * xC[12] * xC[5] + c * yC[12] * yC[5] + b * (xC[12] * yC[5] + yC[12] * xC[5]);
		Data[98] += a * xC[13] * xC[5] + c * yC[13] * yC[5] + b * (xC[13] * yC[5] + yC[13] * xC[5]);
		Data[99] += a * xX[0] * xC[5] + c * yX[0] * yC[5] + b * (xX[0] * yC[5] + yX[0] * xC[5]);
		Data[100] += a * xX[1] * xC[5] + c * yX[1] * yC[5] + b * (xX[1] * yC[5] + yX[1] * xC[5]);
		Data[101] += a * xX[2] * xC[5] + c * yX[2] * yC[5] + b * (xX[2] * yC[5] + yX[2] * xC[5]);
		Data[102] += a * xX[3] * xC[5] + c * yX[3] * yC[5] + b * (xX[3] * yC[5] + yX[3] * xC[5]);
		Data[103] += a * xX[4] * xC[5] + c * yX[4] * yC[5] + b * (xX[4] * yC[5] + yX[4] * xC[5]);
		Data[104] += a * xX[5] * xC[5] + c * yX[5] * yC[5] + b * (xX[5] * yC[5] + yX[5] * xC[5]);

		Data[105] += a * xC[6] * xC[6] + c * yC[6] * yC[6] + b * (xC[6] * yC[6] + yC[6] * xC[6]);
		Data[106] += a * xC[7] * xC[6] + c * yC[7] * yC[6] + b * (xC[7] * yC[6] + yC[7] * xC[6]);
		Data[107] += a * xC[8] * xC[6] + c * yC[8] * yC[6] + b * (xC[8] * yC[6] + yC[8] * xC[6]);
		Data[108] += a * xC[9] * xC[6] + c * yC[9] * yC[6] + b * (xC[9] * yC[6] + yC[9] * xC[6]);
		Data[109] += a * xC[10] * xC[6] + c * yC[10] * yC[6] + b * (xC[10] * yC[6] + yC[10] * xC[6]);
		Data[110] += a * xC[11] * xC[6] + c * yC[11] * yC[6] + b * (xC[11] * yC[6] + yC[11] * xC[6]);
		Data[111] += a * xC[12] * xC[6] + c * yC[12] * yC[6] + b * (xC[12] * yC[6] + yC[12] * xC[6]);
		Data[112] += a * xC[13] * xC[6] + c * yC[13] * yC[6] + b * (xC[13] * yC[6] + yC[13] * xC[6]);
		Data[113] += a * xX[0] * xC[6] + c * yX[0] * yC[6] + b * (xX[0] * yC[6] + yX[0] * xC[6]);
		Data[114] += a * xX[1] * xC[6] + c * yX[1] * yC[6] + b * (xX[1] * yC[6] + yX[1] * xC[6]);
		Data[115] += a * xX[2] * xC[6] + c * yX[2] * yC[6] + b * (xX[2] * yC[6] + yX[2] * xC[6]);
		Data[116] += a * xX[3] * xC[6] + c * yX[3] * yC[6] + b * (xX[3] * yC[6] + yX[3] * xC[6]);
		Data[117] += a * xX[4] * xC[6] + c * yX[4] * yC[6] + b * (xX[4] * yC[6] + yX[4] * xC[6]);
		Data[118] += a * xX[5] * xC[6] + c * yX[5] * yC[6] + b * (xX[5] * yC[6] + yX[5] * xC[6]);

		Data[119] += a * xC[7] * xC[7] + c * yC[7] * yC[7] + b * (xC[7] * yC[7] + yC[7] * xC[7]);
		Data[120] += a * xC[8] * xC[7] + c * yC[8] * yC[7] + b * (xC[8] * yC[7] + yC[8] * xC[7]);
		Data[121] += a * xC[9] * xC[7] + c * yC[9] * yC[7] + b * (xC[9] * yC[7] + yC[9] * xC[7]);
		Data[122] += a * xC[10] * xC[7] + c * yC[10] * yC[7] + b * (xC[10] * yC[7] + yC[10] * xC[7]);
		Data[123] += a * xC[11] * xC[7] + c * yC[11] * yC[7] + b * (xC[11] * yC[7] + yC[11] * xC[7]);
		Data[124] += a * xC[12] * xC[7] + c * yC[12] * yC[7] + b * (xC[12] * yC[7] + yC[12] * xC[7]);
		Data[125] += a * xC[13] * xC[7] + c * yC[13] * yC[7] + b * (xC[13] * yC[7] + yC[13] * xC[7]);
		Data[126] += a * xX[0] * xC[7] + c * yX[0] * yC[7] + b * (xX[0] * yC[7] + yX[0] * xC[7]);
		Data[127] += a * xX[1] * xC[7] + c * yX[1] * yC[7] + b * (xX[1] * yC[7] + yX[1] * xC[7]);
		Data[128] += a * xX[2] * xC[7] + c * yX[2] * yC[7] + b * (xX[2] * yC[7] + yX[2] * xC[7]);
		Data[129] += a * xX[3] * xC[7] + c * yX[3] * yC[7] + b * (xX[3] * yC[7] + yX[3] * xC[7]);
		Data[130] += a * xX[4] * xC[7] + c * yX[4] * yC[7] + b * (xX[4] * yC[7] + yX[4] * xC[7]);
		Data[131] += a * xX[5] * xC[7] + c * yX[5] * yC[7] + b * (xX[5] * yC[7] + yX[5] * xC[7]);

		Data[132] += a * xC[8] * xC[8] + c * yC[8] * yC[8] + b * (xC[8] * yC[8] + yC[8] * xC[8]);
		Data[133] += a * xC[9] * xC[8] + c * yC[9] * yC[8] + b * (xC[9] * yC[8] + yC[9] * xC[8]);
		Data[134] += a * xC[10] * xC[8] + c * yC[10] * yC[8] + b * (xC[10] * yC[8] + yC[10] * xC[8]);
		Data[135] += a * xC[11] * xC[8] + c * yC[11] * yC[8] + b * (xC[11] * yC[8] + yC[11] * xC[8]);
		Data[136] += a * xC[12] * xC[8] + c * yC[12] * yC[8] + b * (xC[12] * yC[8] + yC[12] * xC[8]);
		Data[137] += a * xC[13] * xC[8] + c * yC[13] * yC[8] + b * (xC[13] * yC[8] + yC[13] * xC[8]);
		Data[138] += a * xX[0] * xC[8] + c * yX[0] * yC[8] + b * (xX[0] * yC[8] + yX[0] * xC[8]);
		Data[139] += a * xX[1] * xC[8] + c * yX[1] * yC[8] + b * (xX[1] * yC[8] + yX[1] * xC[8]);
		Data[140] += a * xX[2] * xC[8] + c * yX[2] * yC[8] + b * (xX[2] * yC[8] + yX[2] * xC[8]);
		Data[141] += a * xX[3] * xC[8] + c * yX[3] * yC[8] + b * (xX[3] * yC[8] + yX[3] * xC[8]);
		Data[142] += a * xX[4] * xC[8] + c * yX[4] * yC[8] + b * (xX[4] * yC[8] + yX[4] * xC[8]);
		Data[143] += a * xX[5] * xC[8] + c * yX[5] * yC[8] + b * (xX[5] * yC[8] + yX[5] * xC[8]);

		Data[144] += a * xC[9] * xC[9] + c * yC[9] * yC[9] + b * (xC[9] * yC[9] + yC[9] * xC[9]);
		Data[145] += a * xC[10] * xC[9] + c * yC[10] * yC[9] + b * (xC[10] * yC[9] + yC[10] * xC[9]);
		Data[146] += a * xC[11] * xC[9] + c * yC[11] * yC[9] + b * (xC[11] * yC[9] + yC[11] * xC[9]);
		Data[147] += a * xC[12] * xC[9] + c * yC[12] * yC[9] + b * (xC[12] * yC[9] + yC[12] * xC[9]);
		Data[148] += a * xC[13] * xC[9] + c * yC[13] * yC[9] + b * (xC[13] * yC[9] + yC[13] * xC[9]);
		Data[149] += a * xX[0] * xC[9] + c * yX[0] * yC[9] + b * (xX[0] * yC[9] + yX[0] * xC[9]);
		Data[150] += a * xX[1] * xC[9] + c * yX[1] * yC[9] + b * (xX[1] * yC[9] + yX[1] * xC[9]);
		Data[151] += a * xX[2] * xC[9] + c * yX[2] * yC[9] + b * (xX[2] * yC[9] + yX[2] * xC[9]);
		Data[152] += a * xX[3] * xC[9] + c * yX[3] * yC[9] + b * (xX[3] * yC[9] + yX[3] * xC[9]);
		Data[153] += a * xX[4] * xC[9] + c * yX[4] * yC[9] + b * (xX[4] * yC[9] + yX[4] * xC[9]);
		Data[154] += a * xX[5] * xC[9] + c * yX[5] * yC[9] + b * (xX[5] * yC[9] + yX[5] * xC[9]);

		Data[155] += a * xC[10] * xC[10] + c * yC[10] * yC[10] + b * (xC[10] * yC[10] + yC[10] * xC[10]);
		Data[156] += a * xC[11] * xC[10] + c * yC[11] * yC[10] + b * (xC[11] * yC[10] + yC[11] * xC[10]);
		Data[157] += a * xC[12] * xC[10] + c * yC[12] * yC[10] + b * (xC[12] * yC[10] + yC[12] * xC[10]);
		Data[158] += a * xC[13] * xC[10] + c * yC[13] * yC[10] + b * (xC[13] * yC[10] + yC[13] * xC[10]);
		Data[159] += a * xX[0] * xC[10] + c * yX[0] * yC[10] + b * (xX[0] * yC[10] + yX[0] * xC[10]);
		Data[160] += a * xX[1] * xC[10] + c * yX[1] * yC[10] + b * (xX[1] * yC[10] + yX[1] * xC[10]);
		Data[161] += a * xX[2] * xC[10] + c * yX[2] * yC[10] + b * (xX[2] * yC[10] + yX[2] * xC[10]);
		Data[162] += a * xX[3] * xC[10] + c * yX[3] * yC[10] + b * (xX[3] * yC[10] + yX[3] * xC[10]);
		Data[163] += a * xX[4] * xC[10] + c * yX[4] * yC[10] + b * (xX[4] * yC[10] + yX[4] * xC[10]);
		Data[164] += a * xX[5] * xC[10] + c * yX[5] * yC[10] + b * (xX[5] * yC[10] + yX[5] * xC[10]);

		Data[165] += a * xC[11] * xC[11] + c * yC[11] * yC[11] + b * (xC[11] * yC[11] + yC[11] * xC[11]);
		Data[166] += a * xC[12] * xC[11] + c * yC[12] * yC[11] + b * (xC[12] * yC[11] + yC[12] * xC[11]);
		Data[167] += a * xC[13] * xC[11] + c * yC[13] * yC[11] + b * (xC[13] * yC[11] + yC[13] * xC[11]);
		Data[168] += a * xX[0] * xC[11] + c * yX[0] * yC[11] + b * (xX[0] * yC[11] + yX[0] * xC[11]);
		Data[169] += a * xX[1] * xC[11] + c * yX[1] * yC[11] + b * (xX[1] * yC[11] + yX[1] * xC[11]);
		Data[170] += a * xX[2] * xC[11] + c * yX[2] * yC[11] + b * (xX[2] * yC[11] + yX[2] * xC[11]);
		Data[171] += a * xX[3] * xC[11] + c * yX[3] * yC[11] + b * (xX[3] * yC[11] + yX[3] * xC[11]);
		Data[172] += a * xX[4] * xC[11] + c * yX[4] * yC[11] + b * (xX[4] * yC[11] + yX[4] * xC[11]);
		Data[173] += a * xX[5] * xC[11] + c * yX[5] * yC[11] + b * (xX[5] * yC[11] + yX[5] * xC[11]);

		Data[174] += a * xC[12] * xC[12] + c * yC[12] * yC[12] + b * (xC[12] * yC[12] + yC[12] * xC[12]);
		Data[175] += a * xC[13] * xC[12] + c * yC[13] * yC[12] + b * (xC[13] * yC[12] + yC[13] * xC[12]);
		Data[176] += a * xX[0] * xC[12] + c * yX[0] * yC[12] + b * (xX[0] * yC[12] + yX[0] * xC[12]);
		Data[177] += a * xX[1] * xC[12] + c * yX[1] * yC[12] + b * (xX[1] * yC[12] + yX[1] * xC[12]);
		Data[178] += a * xX[2] * xC[12] + c * yX[2] * yC[12] + b * (xX[2] * yC[12] + yX[2] * xC[12]);
		Data[179] += a * xX[3] * xC[12] + c * yX[3] * yC[12] + b * (xX[3] * yC[12] + yX[3] * xC[12]);
		Data[180] += a * xX[4] * xC[12] + c * yX[4] * yC[12] + b * (xX[4] * yC[12] + yX[4] * xC[12]);
		Data[181] += a * xX[5] * xC[12] + c * yX[5] * yC[12] + b * (xX[5] * yC[12] + yX[5] * xC[12]);

		Data[182] += a * xC[13] * xC[13] + c * yC[13] * yC[13] + b * (xC[13] * yC[13] + yC[13] * xC[13]);
		Data[183] += a * xX[0] * xC[13] + c * yX[0] * yC[13] + b * (xX[0] * yC[13] + yX[0] * xC[13]);
		Data[184] += a * xX[1] * xC[13] + c * yX[1] * yC[13] + b * (xX[1] * yC[13] + yX[1] * xC[13]);
		Data[185] += a * xX[2] * xC[13] + c * yX[2] * yC[13] + b * (xX[2] * yC[13] + yX[2] * xC[13]);
		Data[186] += a * xX[3] * xC[13] + c * yX[3] * yC[13] + b * (xX[3] * yC[13] + yX[3] * xC[13]);
		Data[187] += a * xX[4] * xC[13] + c * yX[4] * yC[13] + b * (xX[4] * yC[13] + yX[4] * xC[13]);
		Data[188] += a * xX[5] * xC[13] + c * yX[5] * yC[13] + b * (xX[5] * yC[13] + yX[5] * xC[13]);

		Data[189] += a * xX[0] * xX[0] + c * yX[0] * yX[0] + b * (xX[0] * yX[0] + yX[0] * xX[0]);
		Data[190] += a * xX[1] * xX[0] + c * yX[1] * yX[0] + b * (xX[1] * yX[0] + yX[1] * xX[0]);
		Data[191] += a * xX[2] * xX[0] + c * yX[2] * yX[0] + b * (xX[2] * yX[0] + yX[2] * xX[0]);
		Data[192] += a * xX[3] * xX[0] + c * yX[3] * yX[0] + b * (xX[3] * yX[0] + yX[3] * xX[0]);
		Data[193] += a * xX[4] * xX[0] + c * yX[4] * yX[0] + b * (xX[4] * yX[0] + yX[4] * xX[0]);
		Data[194] += a * xX[5] * xX[0] + c * yX[5] * yX[0] + b * (xX[5] * yX[0] + yX[5] * xX[0]);

		Data[195] += a * xX[1] * xX[1] + c * yX[1] * yX[1] + b * (xX[1] * yX[1] + yX[1] * xX[1]);
		Data[196] += a * xX[2] * xX[1] + c * yX[2] * yX[1] + b * (xX[2] * yX[1] + yX[2] * xX[1]);
		Data[197] += a * xX[3] * xX[1] + c * yX[3] * yX[1] + b * (xX[3] * yX[1] + yX[3] * xX[1]);
		Data[198] += a * xX[4] * xX[1] + c * yX[4] * yX[1] + b * (xX[4] * yX[1] + yX[4] * xX[1]);
		Data[199] += a * xX[5] * xX[1] + c * yX[5] * yX[1] + b * (xX[5] * yX[1] + yX[5] * xX[1]);

		Data[200] += a * xX[2] * xX[2] + c * yX[2] * yX[2] + b * (xX[2] * yX[2] + yX[2] * xX[2]);
		Data[201] += a * xX[3] * xX[2] + c * yX[3] * yX[2] + b * (xX[3] * yX[2] + yX[3] * xX[2]);
		Data[202] += a * xX[4] * xX[2] + c * yX[4] * yX[2] + b * (xX[4] * yX[2] + yX[4] * xX[2]);
		Data[203] += a * xX[5] * xX[2] + c * yX[5] * yX[2] + b * (xX[5] * yX[2] + yX[5] * xX[2]);

		Data[204] += a * xX[3] * xX[3] + c * yX[3] * yX[3] + b * (xX[3] * yX[3] + yX[3] * xX[3]);
		Data[205] += a * xX[4] * xX[3] + c * yX[4] * yX[3] + b * (xX[4] * yX[3] + yX[4] * xX[3]);
		Data[206] += a * xX[5] * xX[3] + c * yX[5] * yX[3] + b * (xX[5] * yX[3] + yX[5] * xX[3]);

		Data[207] += a * xX[4] * xX[4] + c * yX[4] * yX[4] + b * (xX[4] * yX[4] + yX[4] * xX[4]);
		Data[208] += a * xX[5] * xX[4] + c * yX[5] * yX[4] + b * (xX[5] * yX[4] + yX[5] * xX[4]);

		Data[209] += a * xX[5] * xX[5] + c * yX[5] * yX[5] + b * (xX[5] * yX[5] + yX[5] * xX[5]);

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

		TopRight_Data[24] += xC[8] * TR00 + yC[8] * TR10;
		TopRight_Data[25] += xC[8] * TR01 + yC[8] * TR11;
		TopRight_Data[26] += xC[8] * TR02 + yC[8] * TR12;

		TopRight_Data[27] += xC[9] * TR00 + yC[9] * TR10;
		TopRight_Data[28] += xC[9] * TR01 + yC[9] * TR11;
		TopRight_Data[29] += xC[9] * TR02 + yC[9] * TR12;

		TopRight_Data[30] += xC[10] * TR00 + yC[10] * TR10;
		TopRight_Data[31] += xC[10] * TR01 + yC[10] * TR11;
		TopRight_Data[32] += xC[10] * TR02 + yC[10] * TR12;

		TopRight_Data[33] += xC[11] * TR00 + yC[11] * TR10;
		TopRight_Data[34] += xC[11] * TR01 + yC[11] * TR11;
		TopRight_Data[35] += xC[11] * TR02 + yC[11] * TR12;

		TopRight_Data[36] += xC[12] * TR00 + yC[12] * TR10;
		TopRight_Data[37] += xC[12] * TR01 + yC[12] * TR11;
		TopRight_Data[38] += xC[12] * TR02 + yC[12] * TR12;

		TopRight_Data[39] += xC[13] * TR00 + yC[13] * TR10;
		TopRight_Data[40] += xC[13] * TR01 + yC[13] * TR11;
		TopRight_Data[41] += xC[13] * TR02 + yC[13] * TR12;

		TopRight_Data[42] += xX[0] * TR00 + yX[0] * TR10;
		TopRight_Data[43] += xX[0] * TR01 + yX[0] * TR11;
		TopRight_Data[44] += xX[0] * TR02 + yX[0] * TR12;

		TopRight_Data[45] += xX[1] * TR00 + yX[1] * TR10;
		TopRight_Data[46] += xX[1] * TR01 + yX[1] * TR11;
		TopRight_Data[47] += xX[1] * TR02 + yX[1] * TR12;

		TopRight_Data[48] += xX[2] * TR00 + yX[2] * TR10;
		TopRight_Data[49] += xX[2] * TR01 + yX[2] * TR11;
		TopRight_Data[50] += xX[2] * TR02 + yX[2] * TR12;

		TopRight_Data[51] += xX[3] * TR00 + yX[3] * TR10;
		TopRight_Data[52] += xX[3] * TR01 + yX[3] * TR11;
		TopRight_Data[53] += xX[3] * TR02 + yX[3] * TR12;

		TopRight_Data[54] += xX[4] * TR00 + yX[4] * TR10;
		TopRight_Data[55] += xX[4] * TR01 + yX[4] * TR11;
		TopRight_Data[56] += xX[4] * TR02 + yX[4] * TR12;

		TopRight_Data[57] += xX[5] * TR00 + yX[5] * TR10;
		TopRight_Data[58] += xX[5] * TR01 + yX[5] * TR11;
		TopRight_Data[59] += xX[5] * TR02 + yX[5] * TR12;

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
	EIGEN_ALIGN16 float Data[212];  // 210 are used, but is divisible by 4 (4*4=16) for word alignment.
	EIGEN_ALIGN16 float TopRight_Data[60];
	EIGEN_ALIGN16 float BotRight_Data[8];
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
		assert(idx==4*45);
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
