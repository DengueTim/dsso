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
#define MAX_ACTIVE_FRAMES 100

#include "util/globalCalib.h"
#include "vector"

#include <iostream>
#include <fstream>
#include "util/NumType.h"
#include "FullSystem/Residuals.h"
#include "util/ImageAndExposure.h"

namespace dso {

inline Vec2 affFromTo(const Vec2 &from, const Vec2 &to)	// contains affine parameters as XtoWorld.
		{
	return Vec2(from[0] / to[0], (from[1] - to[1]) / to[0]);
}

struct FrameHessian;
struct PointHessian;

class ImmaturePoint;
class FrameShell;

class EFFrame;
class EFPoint;

#define SCALE_IDEPTH 1.0f		// scales internal value to idepth.
#define SCALE_XI_ROT 1.0f
#define SCALE_XI_TRANS 0.5f
#define SCALE_F 50.0f
#define SCALE_C 50.0f
#define SCALE_W 1.0f
#define SCALE_A 10.0f
#define SCALE_B 1000.0f

#define SCALE_IDEPTH_INVERSE (1.0f / SCALE_IDEPTH)
#define SCALE_XI_ROT_INVERSE (1.0f / SCALE_XI_ROT)
#define SCALE_XI_TRANS_INVERSE (1.0f / SCALE_XI_TRANS)
#define SCALE_F_INVERSE (1.0f / SCALE_F)
#define SCALE_C_INVERSE (1.0f / SCALE_C)
#define SCALE_W_INVERSE (1.0f / SCALE_W)
#define SCALE_A_INVERSE (1.0f / SCALE_A)
#define SCALE_B_INVERSE (1.0f / SCALE_B)

struct FrameFramePrecalc {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;
	// static values
	static int instanceCounter;
	FrameHessian *host;	// defines row
	FrameHessian *target;	// defines column

	// precalc values
	Mat33f PRE_RTll;
	Mat33f PRE_KRKiTll;
	Mat33f PRE_RKiTll;
	Mat33f PRE_RTll_0;

	Vec2f PRE_aff_mode;
	float PRE_b0_mode;

	Vec3f PRE_tTll;
	Vec3f PRE_KtTll;
	Vec3f PRE_tTll_0;

	float distanceLL;

	inline ~FrameFramePrecalc() {
	}
	inline FrameFramePrecalc() {
		host = target = 0;
	}
	void set(FrameHessian *host, FrameHessian *target, CalibHessian *HCalib);
};

struct FrameHessian {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;
	EFFrame *efFrame;

	// constant info & pre-calculated values
	//DepthImageWrap* frame;
	FrameShell *shell;

	Eigen::Vector3f *dI;	// trace, fine tracking. Used for direction select (not for gradient histograms etc.)   dIp[0]
	Eigen::Vector3f *dIr;	// dI for Right image..
	Eigen::Vector3f *dIp[MAX_PYR_LEVELS];	// coarse tracking / coarse initializer. NAN in [0] only.  Elements: Pixel value, dx, dy
	Eigen::Vector3f *dIrp[MAX_PYR_LEVELS];  // optional for right image.
	float *absSquaredGrad[MAX_PYR_LEVELS];  // only used for pixel select (histograms etc.). no NAN.

	int keyFrameID;						// incremental ID for keyframes only!
	static int instanceCounter;
	int fhIdx;

	// Photometric Calibration Stuff
	float frameEnergyTH;	// set dynamically depending on tracking residual
	float ab_exposure;

	bool flaggedForMarginalization;

	std::vector<PointHessian*> pointHessians;				// contains all ACTIVE points.
	std::vector<PointHessian*> pointHessiansMarginalized;// contains all MARGINALIZED points (= fully marginalized, usually because point went OOB.)
	std::vector<PointHessian*> pointHessiansOut;		// contains all OUTLIER points (= discarded.).
	std::vector<ImmaturePoint*> immaturePoints;		// contains all OUTLIER points (= discarded.).

	Mat66 nullspaces_pose;
	Mat42 nullspaces_affine;
	Vec6 nullspaces_scale;

	// variable info.
	SE3 worldToCam_evalPT;
	Vec10 state_zero;
	Vec10 state_scaled;
	Vec10 state;	// [0-5: worldToCam-leftEps. 6-7: a,b]
	Vec10 step;
	Vec10 step_backup;
	Vec10 state_backup;

	EIGEN_STRONG_INLINE const SE3& get_worldToCam_evalPT() const {
		return worldToCam_evalPT;
	}
	EIGEN_STRONG_INLINE const Vec10& get_state_zero() const {
		return state_zero;
	}
	EIGEN_STRONG_INLINE const Vec10& get_state() const {
		return state;
	}
	EIGEN_STRONG_INLINE const Vec10& get_state_scaled() const {
		return state_scaled;
	}
	EIGEN_STRONG_INLINE const Vec10 get_state_minus_stateZero() const {
		return get_state() - state_zero;
	}

	// precalc values
	SE3 PRE_worldToCam;
	SE3 PRE_camToWorld;
	// Pre-calculate relative translations between this(host) frame and all other active(target) frames.
	std::vector<FrameFramePrecalc, Eigen::aligned_allocator<FrameFramePrecalc>> targetPrecalc;
	MinimalImageB3 *debugImage;

	inline Vec6 w2c_leftEps() const {
		return get_state_scaled().head<6>();
	}
	inline AffLight aff_g2l() const {
		return AffLight(get_state_scaled()[6], get_state_scaled()[7]);
	}
	inline AffLight aff_g2l_0() const {
		return AffLight(get_state_zero()[6] * SCALE_A, get_state_zero()[7] * SCALE_B);
	}

	void setStateZero();
	inline void setState(const Vec10 &state) {

		this->state = state;
		state_scaled.segment<3>(0) = SCALE_XI_TRANS * state.segment<3>(0);
		state_scaled.segment<3>(3) = SCALE_XI_ROT * state.segment<3>(3);
		state_scaled[6] = SCALE_A * state[6];
		state_scaled[7] = SCALE_B * state[7];
		state_scaled[8] = SCALE_A * state[8];
		state_scaled[9] = SCALE_B * state[9];

		PRE_worldToCam = SE3::exp(w2c_leftEps()) * worldToCam_evalPT;
		PRE_camToWorld = PRE_worldToCam.inverse();
		//setCurrentNullspace();
	}
	;
	inline void setStateScaled(const Vec10 &state_scaled) {

		this->state_scaled = state_scaled;
		state.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaled.segment<3>(0);
		state.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaled.segment<3>(3);
		state[6] = SCALE_A_INVERSE * state_scaled[6];
		state[7] = SCALE_B_INVERSE * state_scaled[7];
		state[8] = SCALE_A_INVERSE * state_scaled[8];
		state[9] = SCALE_B_INVERSE * state_scaled[9];

		PRE_worldToCam = SE3::exp(w2c_leftEps()) * worldToCam_evalPT;
		PRE_camToWorld = PRE_worldToCam.inverse();
		//setCurrentNullspace();
	}
	;
	inline void setEvalPT(const SE3 &worldToCam_evalPT, const Vec10 &state) {
		this->worldToCam_evalPT = worldToCam_evalPT;
		setState(state);
		setStateZero();
	}
	;

	inline void setEvalPT_scaled(const SE3 &worldToCam_evalPT, const AffLight &aff_g2l) {
		Vec10 initial_state = Vec10::Zero();
		initial_state[6] = aff_g2l.a;
		initial_state[7] = aff_g2l.b;
		this->worldToCam_evalPT = worldToCam_evalPT;
		setStateScaled(initial_state);
		setStateZero();
	}
	;

	void release();

	inline ~FrameHessian() {
		assert(efFrame == 0);
		release();
		instanceCounter--;
		for (int i = 0; i < pyrLevelsUsed; i++) {
			delete[] dIp[i];
			if (dIrp[i])
				delete[] dIrp[i];
			delete[] absSquaredGrad[i];

		}

		if (debugImage != 0)
			delete debugImage;
	}
	;
	inline FrameHessian() {
		for (int i = 0; i < pyrLevelsUsed; i++)
			dIrp[i] = 0;
		instanceCounter++;
		flaggedForMarginalization = false;
		keyFrameID = -1;
		efFrame = 0;
		frameEnergyTH = 8 * 8 * patternNum;

		debugImage = 0;
	}
	;

	template<bool makeRightPyramid>
	void makeImages(float *color, float *colorR, CalibHessian *HCalib);

	inline Vec10 getPrior() {
		Vec10 p = Vec10::Zero();
		if (keyFrameID == 0) {
			if (!(setting_solverMode & SOLVER_REMOVE_POSEPRIOR )) {
				p.head<3>() = Vec3::Constant(setting_initialTransPrior);
				p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);
			}

			p[6] = setting_initialAffAPrior;
			p[7] = setting_initialAffBPrior;
		} else {
			if (setting_affineOptModeA < 0)
				p[6] = setting_initialAffAPrior;
			else
				p[6] = setting_affineOptModeA;

			if (setting_affineOptModeB < 0)
				p[7] = setting_initialAffBPrior;
			else
				p[7] = setting_affineOptModeB;
		}
		p[8] = setting_initialAffAPrior;
		p[9] = setting_initialAffBPrior;
		return p;
	}

	inline Vec10 getPriorZero() {
		return Vec10::Zero();
	}

};

struct CalibHessian {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;

	VecC value_zero; // Start values from config files.
	VecC value_zero_scaled;
	VecC value_scaled; // The scaled value is actually the real value.
	VecCf value_scaledf;
	VecCf value_scaledi;
	VecC value; // This value is the one that is optimised on.
	VecC step;
	VecC step_backup;
	VecC value_backup;
	VecC value_minus_value_zero;

	inline CalibHessian(const Mat33 &leftK, const Mat33 &rightK, const SE3 leftToRight) {

		VecC initial_value = VecC::Zero();

		initial_value[0] = leftK(0, 0); // fx
		initial_value[1] = leftK(1, 1); // fy
		initial_value[2] = leftK(0, 2); // cx
		initial_value[3] = leftK(1, 2); // cy

		initial_value[4] = rightK(0, 0);
		initial_value[5] = rightK(1, 1);
		initial_value[6] = rightK(0, 2);
		initial_value[7] = rightK(1, 2);

		initial_value.segment<6>(8) = leftToRight.log();

		setValueScaled(initial_value);
		// Take initial value as eval PT (zero).
		value_zero = value;
		value_zero_scaled = value_scaled;
		value_minus_value_zero.setZero();

		for (int i = 0; i < 256; i++)
			Binv[i] = B[i] = i;		// set gamma function to identity
	}

	// normal mode: use the optimized parameters everywhere!
	inline float& fxl() {
		return value_scaledf[0];
	}
	inline float& fyl() {
		return value_scaledf[1];
	}
	inline float& cxl() {
		return value_scaledf[2];
	}
	inline float& cyl() {
		return value_scaledf[3];
	}
	inline float& fxli() {
		return value_scaledi[0];
	}
	inline float& fyli() {
		return value_scaledi[1];
	}
	inline float& cxli() {
		return value_scaledi[2];
	}
	inline float& cyli() {
		return value_scaledi[3];
	}

	inline float& fxlR() {
		return value_scaledf[4];
	}
	inline float& fylR() {
		return value_scaledf[5];
	}
	inline float& cxlR() {
		return value_scaledf[6];
	}
	inline float& cylR() {
		return value_scaledf[7];
	}
	inline float& fxliR() {
		return value_scaledi[4];
	}
	inline float& fyliR() {
		return value_scaledi[5];
	}
	inline float& cxliR() {
		return value_scaledi[6];
	}
	inline float& cyliR() {
		return value_scaledi[7];
	}

	// Called every GN step..
	inline void setValue(const VecC &value) {
		// [0-3: Kl, 4-7: Kr, 8-13: l2r]
		this->value = value;
		value_scaled[0] = SCALE_F * value[0];
		value_scaled[1] = SCALE_F * value[1];
		value_scaled[2] = SCALE_C * value[2];
		value_scaled[3] = SCALE_C * value[3];

		value_scaled[4] = SCALE_F * value[4];
		value_scaled[5] = SCALE_F * value[5];
		value_scaled[6] = SCALE_C * value[6];
		value_scaled[7] = SCALE_C * value[7];

		value_scaled[8] = SCALE_XI_TRANS * value[8];
		value_scaled[9] = SCALE_XI_TRANS * value[9];
		value_scaled[10] = SCALE_XI_TRANS * value[10];
		value_scaled[11] = SCALE_XI_ROT * value[11];
		value_scaled[12] = SCALE_XI_ROT * value[12];
		value_scaled[13] = SCALE_XI_ROT * value[13];

		this->value_scaledf = this->value_scaled.cast<float>();
		this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
		this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
		this->value_scaledi[2] = -this->value_scaledf[2] / this->value_scaledf[0];
		this->value_scaledi[3] = -this->value_scaledf[3] / this->value_scaledf[1];

		this->value_scaledi[4] = 1.0f / this->value_scaledf[4];
		this->value_scaledi[5] = 1.0f / this->value_scaledf[5];
		this->value_scaledi[6] = -this->value_scaledf[6] / this->value_scaledf[4];
		this->value_scaledi[7] = -this->value_scaledf[7] / this->value_scaledf[5];

		//TODO Left/Right transform inverse.

		this->value_minus_value_zero = this->value - this->value_zero;
	}

	float Binv[256];
	float B[256];

	EIGEN_STRONG_INLINE float getBGradOnly(float color) {
		int c = color + 0.5f;
		if (c < 5)
			c = 5;
		if (c > 250)
			c = 250;
		return B[c + 1] - B[c];
	}

	EIGEN_STRONG_INLINE float getBInvGradOnly(float color) {
		int c = color + 0.5f;
		if (c < 5)
			c = 5;
		if (c > 250)
			c = 250;
		return Binv[c + 1] - Binv[c];
	}

	SE3 getLeftToRightZero() {
		return SE3::exp(value_zero_scaled.segment<6>(8));
	}

	SE3 getLeftToRight() {
		return SE3::exp(value_scaled.segment<6>(8));
	}

	void updateLeftToRightZero() {
		std::cout << "LR Delta:\t" << (value_zero_scaled.segment<6>(8) - value_scaled.segment<6>(8)).transpose() << "\n";
		std::cout << "LR Abs:\t" << getLeftToRight().log().transpose() << "\n";
		value_zero.segment<6>(8) = value.segment<6>(8);
		value_zero_scaled[8] = SCALE_XI_TRANS * value_zero[8];
		value_zero_scaled[9] = SCALE_XI_TRANS * value_zero[9];
		value_zero_scaled[10] = SCALE_XI_TRANS * value_zero[10];
		value_zero_scaled[11] = SCALE_XI_ROT * value_zero[11];
		value_zero_scaled[12] = SCALE_XI_ROT * value_zero[12];
		value_zero_scaled[13] = SCALE_XI_ROT * value_zero[13];

		this->value_minus_value_zero.setZero();
	}

private:
	// Called only at startup with config values.
	inline void setValueScaled(const VecC &value_scaled) {
		this->value_scaled = value_scaled;
		this->value_scaledf = this->value_scaled.cast<float>();
		value[0] = SCALE_F_INVERSE * value_scaled[0];
		value[1] = SCALE_F_INVERSE * value_scaled[1];
		value[2] = SCALE_C_INVERSE * value_scaled[2];
		value[3] = SCALE_C_INVERSE * value_scaled[3];

		value[4] = SCALE_F_INVERSE * value_scaled[4];
		value[5] = SCALE_F_INVERSE * value_scaled[5];
		value[6] = SCALE_C_INVERSE * value_scaled[6];
		value[7] = SCALE_C_INVERSE * value_scaled[7];

		value[8] = SCALE_XI_TRANS_INVERSE * value_scaled[8];
		value[9] = SCALE_XI_TRANS_INVERSE * value_scaled[9];
		value[10] = SCALE_XI_TRANS_INVERSE * value_scaled[10];
		value[11] = SCALE_XI_ROT_INVERSE * value_scaled[11];
		value[12] = SCALE_XI_ROT_INVERSE * value_scaled[12];
		value[13] = SCALE_XI_ROT_INVERSE * value_scaled[13];

		this->value_minus_value_zero = this->value - this->value_zero;
		this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
		this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
		this->value_scaledi[2] = -this->value_scaledf[2] / this->value_scaledf[0];
		this->value_scaledi[3] = -this->value_scaledf[3] / this->value_scaledf[1];

		this->value_scaledi[4] = 1.0f / this->value_scaledf[4];
		this->value_scaledi[5] = 1.0f / this->value_scaledf[5];
		this->value_scaledi[6] = -this->value_scaledf[6] / this->value_scaledf[4];
		this->value_scaledi[7] = -this->value_scaledf[7] / this->value_scaledf[5];
	}
};

/* Hessian component associated with one point.
 * A point belongs to a "host" frame.
 * A point has a residual(PointFrameResidual) for each "target" frame it appears in.
 * Each PointFrameResidual has a RawResidualJacobian which holds:
 * 		The pixel residual's for each pixel in the point appearance
 * 		Various computed derivative components used to compute the Jacobian & Hessian partial derivatives.
 * The inverse depth is optimised using the residual error from the point's projection into the target frame.
 */

struct PointHessianBase {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;
	// Class with bits used by optimiser.. for easier testing..
	EFPoint *efPoint;

	bool hasDepthPrior;

	float idepth_zero;
	float idepth;
	float step;

	float idepth_hessian;
	float maxRelBaseline;

	PointHessianBase() {
		efPoint = 0;

		hasDepthPrior = false;

		idepth_zero = 0;
		idepth = 0;
		step = 0;

		idepth_hessian = 0;
		maxRelBaseline = 0;
	}
};

struct PointHessian: public PointHessianBase {
	static int instanceCounter;

	// static values
	float color[MAX_RES_PER_POINT];			// colors in host frame
	float weights[MAX_RES_PER_POINT];		// host-weights for respective residuals.

	float u, v;
	int idx;
	float energyTH;
	FrameHessian *host;

	char my_type;

	float idepth_scaled;
	float idepth_zero_scaled;
	float step_backup;
	float idepth_backup;

	float nullspaces_scale;
	int numGoodResiduals;

	enum PtStatus {
		ACTIVE = 0, INACTIVE, OUTLIER, OOB, MARGINALIZED
	};
	PtStatus status;

	inline void setPointStatus(PtStatus s) {
		status = s;
	}

	inline void setIdepth(float idepth) {
		this->idepth = idepth;
		this->idepth_scaled = SCALE_IDEPTH * idepth;
	}
	inline void setIdepthScaled(float idepth_scaled) {
		this->idepth = SCALE_IDEPTH_INVERSE * idepth_scaled;
		this->idepth_scaled = idepth_scaled;
	}
	inline void setIdepthZero(float idepth) {
		idepth_zero = idepth;
		idepth_zero_scaled = SCALE_IDEPTH * idepth;
		nullspaces_scale = -(idepth * 1.001 - idepth / 1.001) * 500;
	}

	std::vector<PointFrameResidual*> residuals;			// only contains good residuals (not OOB and not OUTLIER). Arbitrary order.
	std::pair<PointFrameResidual*, ResState> lastResiduals[2]; // contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).

	void release();
	PointHessian(const ImmaturePoint *const rawPoint);
	inline ~PointHessian() {
		assert(efPoint == 0);
		release();
		instanceCounter--;
	}

	inline bool isOOB(const std::vector<FrameHessian*> &toMarg) const {

		int visInToMarg = 0;
		for (PointFrameResidual *r : residuals) {
			if (r->state_state != ResState::IN)
				continue;
			for (FrameHessian *k : toMarg)
				if (r->target == k)
					visInToMarg++;
		}
		if ((int) residuals.size() >= setting_minGoodActiveResForMarg && numGoodResiduals > setting_minGoodResForMarg + 10
				&& (int) residuals.size() - visInToMarg < setting_minGoodActiveResForMarg)
			return true;

		if (lastResiduals[0].second == ResState::OOB)
			return true;
		if (residuals.size() < 2)
			return false;
		if (lastResiduals[0].second == ResState::OUTLIER && lastResiduals[1].second == ResState::OUTLIER)
			return true;
		return false;
	}

	inline bool isInlierNew() {
		return (int) residuals.size() >= setting_minGoodActiveResForMarg && numGoodResiduals >= setting_minGoodResForMarg;
	}

};

}

