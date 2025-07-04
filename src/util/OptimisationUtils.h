//
// Created by Tim P on 09/05/2025.
//

#ifndef DSO_SRC_UTIL_OPTIMISATIONUTILS_H
#define DSO_SRC_UTIL_OPTIMISATIONUTILS_H

#include "NumType.h"

namespace dso {

template<int SegLength> inline void marginalizeSchur(MatXX &H,
											 VecX &b,
											 const int segStart,
											 const Eigen::Matrix<double, SegLength, 1> &prior,
											 const Eigen::Matrix<double, SegLength, 1> &delta,
											 const float scaleConditioner = 10) {
	const long unsigned int odim = b.size();
	assert(H.cols() == odim);
	assert(H.rows() == odim);
	assert(segStart >= 0 && segStart < odim);
	assert(SegLength >= 0 && (segStart + SegLength) < odim);
	const long unsigned int ndim = odim - SegLength;// new dimension

	if ((segStart + SegLength) < odim) {
		// Move/Swap segment to end.
		const int ntail = odim - (segStart + SegLength);
		assert((segStart + SegLength + ntail) == odim);

		VecX bTmp = b.segment<SegLength>(segStart);
		VecX tailTMP = b.tail(ntail);
		b.segment(segStart, ntail) = tailTMP;
		b.tail<SegLength>() = bTmp;

		MatXX HtmpCol = H.block(0, segStart, odim, SegLength);
		MatXX rightColsTmp = H.rightCols(ntail);
		H.block(0, segStart, odim, ntail) = rightColsTmp;
		H.rightCols<SegLength>() = HtmpCol;

		MatXX HtmpRow = H.block(segStart, 0, SegLength, odim);
		MatXX botRowsTmp = H.bottomRows(ntail);
		H.block(segStart, 0, ntail, odim) = botRowsTmp;
		H.bottomRows<SegLength>() = HtmpRow;
	}

	// marginalize.
	H.bottomRightCorner<SegLength, SegLength>().diagonal() += prior;
	b.tail<SegLength>() += prior.cwiseProduct(delta);

	VecX SVec = (H.diagonal().cwiseAbs() + VecX::Constant(H.cols(), scaleConditioner)).cwiseSqrt();
	VecX SVecI = SVec.cwiseInverse();

	// scale!
	MatXX HScaled = SVecI.asDiagonal() * H * SVecI.asDiagonal();
	VecX bScaled = SVecI.asDiagonal() * b;

	// invert bottom part!
	MatXX hpi = HScaled.bottomRightCorner<SegLength, SegLength>();
	hpi = 0.5f * (hpi + hpi);
	hpi = hpi.inverse();
	hpi = 0.5f * (hpi + hpi);
	assert(!hpi.hasNaN());

	// schur-complement!
	MatXX bli = HScaled.bottomLeftCorner(SegLength, ndim).transpose() * hpi;
	HScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HScaled.bottomLeftCorner(SegLength, ndim);
	bScaled.head(ndim).noalias() -= bli * bScaled.tail<SegLength>();

	//unscale!
	HScaled = SVec.asDiagonal() * HScaled * SVec.asDiagonal();
	bScaled = SVec.asDiagonal() * bScaled;

	// set.
	H = 0.5 * (HScaled.topLeftCorner(ndim, ndim) + HScaled.topLeftCorner(ndim, ndim).transpose());
	b = bScaled.head(ndim);

	assert(!H.hasNaN());
	assert(!b.hasNaN());
	assert(ndim == H.cols());
	assert(ndim == b.rows());
	assert(ndim == b.size());
}

inline const Mat33 so3RightJacobian(const Vec3 &omega) {
	double theta = omega.norm();
	if (theta < 1e-6) {  // Small angle approximation
		return Mat33::Identity();
	}

	Mat33 omegaHat = SO3::hat(omega);
	Mat33 omegaHatSquared = omegaHat * omegaHat;

	double thetaSquared = theta * theta;
	double thetaCubed = thetaSquared * theta;
	double cosTheta = std::cos(theta);
	double sinTheta = std::sin(theta);

	return Mat33::Identity()
		- (1 - cosTheta) / thetaSquared * omegaHat
		+ (theta - sinTheta) / thetaCubed * omegaHatSquared;
}

inline const Mat33 so3RightJacobianInverse(const Vec3 &omega) {
	double theta = omega.norm();

	if (theta < 1e-6) {  // Small angle approximation
		return Mat33::Identity();
	}

	Mat33 omegaHat = SO3::hat(omega);
	Mat33 omegaHatSquared = omegaHat * omegaHat;

	double thetaSquared = theta * theta;
	double cosTheta = std::cos(theta);
	double sinTheta = std::sin(theta);

	return Mat33::Identity()
		+ 0.5 * omegaHat
		+ (1 / thetaSquared - (1 + cosTheta) / (2 * theta * sinTheta)) * omegaHatSquared;
}

}

#endif //DSO_SRC_UTIL_OPTIMISATIONUTILS_H
