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

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

bool EFAdjointsValid = false;
bool EFIndicesValid = false;
bool EFDeltaValid = false;

void EnergyFunctional::setAdjointsF(CalibHessian *Hcalib) {

	if (adHost != 0)
		delete[] adHost;
	if (adTarget != 0)
		delete[] adTarget;
	adHost = new Mat88[nFrames * nFrames];
	adTarget = new Mat88[nFrames * nFrames];

	for (int h = 0; h < nFrames; h++)
		for (int t = 0; t < nFrames; t++) {
			FrameHessian *host = frames[h]->fh;
			FrameHessian *target = frames[t]->fh;

            assert(!target->get_worldToCam_evalPT().matrix3x4().hasNaN());
            assert(!host->get_worldToCam_evalPT().matrix3x4().hasNaN());
            assert(!Hcalib->getLeftToRight().matrix3x4().hasNaN());

			SE3 hostToTarget =
					(h != t) ? target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse() : Hcalib->getLeftToRight();
			//SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

			Mat88 AH = Mat88::Identity();
			Mat88 AT = Mat88::Identity();

			AH.topLeftCorner<6, 6>() = -hostToTarget.Adj().transpose();
			AT.topLeftCorner<6, 6>() = Mat66::Identity();

			Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(),
					target->aff_g2l_0()).cast<float>();
			AT(6, 6) = -affLL[0];
			AH(6, 6) = affLL[0];
			AT(7, 7) = -1;
			AH(7, 7) = affLL[0];  // surely should be B?

			AH.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
			AH.block<3, 8>(3, 0) *= SCALE_XI_ROT;
			AH.block<1, 8>(6, 0) *= SCALE_A;
			AH.block<1, 8>(7, 0) *= SCALE_B;
			AT.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
			AT.block<3, 8>(3, 0) *= SCALE_XI_ROT;
			AT.block<1, 8>(6, 0) *= SCALE_A;
			AT.block<1, 8>(7, 0) *= SCALE_B;

			adHost[h + t * nFrames] = AH;
			adTarget[h + t * nFrames] = AT;

            assert(!AH.hasNaN());
            assert(!AT.hasNaN());
		}
	cPrior.head(8) = Vec8::Constant(setting_initialCalibHessian);
	cPrior.segment<3>(8) = Vec3::Constant(setting_initialTransPrior);
	cPrior.tail(3) = Vec3::Constant(setting_initialRotPrior);

	if (adHostF != 0)
		delete[] adHostF;
	if (adTargetF != 0)
		delete[] adTargetF;
	adHostF = new Mat88f[nFrames * nFrames];
	adTargetF = new Mat88f[nFrames * nFrames];

	for (int h = 0; h < nFrames; h++) {
		for (int t = 0; t < nFrames; t++) {
			adHostF[h + t * nFrames] = adHost[h + t * nFrames].cast<float>();
			adTargetF[h + t * nFrames] = adTarget[h + t * nFrames].cast<float>();
		}
	}

	EFAdjointsValid = true;
}

EnergyFunctional::EnergyFunctional() {
	adHost = 0;
	adTarget = 0;

	red = 0;

	adHostF = 0;
	adTargetF = 0;
	adHTdeltaF = 0;

	nFrames = nResiduals = nPoints = 0;

	HM = MatXX::Zero(CPARS, CPARS);
	bM = VecX::Zero(CPARS);

	accTop_L = new AccumulatedTopHessianSSE();
	accTop_A = new AccumulatedTopHessianSSE();
	accSC_bot = new AccumulatedSCHessianSSE();

	resInA = resInL = resInM = 0;
}
EnergyFunctional::~EnergyFunctional() {
	for (EFFrame *f : frames) {
		for (EFPoint *p : f->points) {
			for (EFResidual *r : p->residualsAll) {
				r->data->efResidual = 0;
				delete r;
			}
			p->ph->efPoint = 0;
			delete p;
		}
		f->fh->efFrame = 0;
		delete f;
	}

	if (adHost != 0)
		delete[] adHost;
	if (adTarget != 0)
		delete[] adTarget;

	if (adHostF != 0)
		delete[] adHostF;
	if (adTargetF != 0)
		delete[] adTargetF;
	if (adHTdeltaF != 0)
		delete[] adHTdeltaF;

	delete accTop_L;
	delete accTop_A;
	delete accSC_bot;
}

void EnergyFunctional::setDeltaF(CalibHessian *HCalib) {
	if (adHTdeltaF != 0)
		delete[] adHTdeltaF;
	adHTdeltaF = new Mat18f[nFrames * nFrames];
	for (int h = 0; h < nFrames; h++)
		for (int t = 0; t < nFrames; t++) {
			int idx = h + t * nFrames;
			adHTdeltaF[idx] = frames[h]->fh->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx]
					+ frames[t]->fh->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
		}

	cDelta = HCalib->value_minus_value_zero;
	for (EFFrame *f : frames) {
		f->delta = f->fh->get_state_minus_stateZero().head<8>();
		f->delta_prior = (f->fh->get_state() - f->fh->getPriorZero()).head<8>();

		for (EFPoint *p : f->points)
			p->deltaF = p->ph->idepth - p->ph->idepth_zero;
	}

	EFDeltaValid = true;
}

// accumulates & shifts L.
void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT) {
	if (MT) {
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accTop_A, nFrames, _1, _2, _3, _4), 0, 0, 0);
		red->reduce(
				boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>, accTop_A, &allPoints, adHTdeltaF, &cDelta, _1, _2, _3,
						_4), 0, allPoints.size(), 50);
		accTop_A->stitchDoubleMT(red, H, b, adHost, adTarget, true);
		resInA = accTop_A->nres[0];
	} else {
		accTop_A->setZero(nFrames);
		for (EFFrame *f : frames)
			for (EFPoint *p : f->points)
				accTop_A->addPoint<0>(p, adHTdeltaF, &cDelta);  // <0> - Active point.
		accTop_A->stitchDoubleMT(red, H, b, adHost, adTarget, false);
		resInA = accTop_A->nres[0];
	}
}

// accumulates & shifts L.
void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT) {
	if (MT) {
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accTop_L, nFrames, _1, _2, _3, _4), 0, 0, 0);
		red->reduce(
				boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>, accTop_L, &allPoints, adHTdeltaF, &cDelta, _1, _2, _3,
						_4), 0, allPoints.size(), 50);
		accTop_L->stitchDoubleMT(red, H, b, adHost, adTarget, true);
		accTop_L->addPrior(H, b, cPrior, cDelta, frames);
		resInL = accTop_L->nres[0];
	} else {
		accTop_L->setZero(nFrames);
		for (EFFrame *f : frames)
			for (EFPoint *p : f->points)
				accTop_L->addPoint<1>(p, adHTdeltaF, &cDelta); // <1> - Linearised point.
		accTop_L->stitchDoubleMT(red, H, b, adHost, adTarget, false);
		accTop_L->addPrior(H, b, cPrior, cDelta, frames);
		resInL = accTop_L->nres[0];
	}
}

void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT) {
	if (MT) {
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSC_bot, nFrames, _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal, accSC_bot, &allPoints, true, _1, _2, _3, _4), 0,
				allPoints.size(), 50);
		accSC_bot->stitchDoubleMT(red, H, b, true);
	} else {
		accSC_bot->setZero(nFrames);
		for (EFFrame *f : frames)
			for (EFPoint *p : f->points)
				accSC_bot->addPoint(p, true);
		accSC_bot->stitchDoubleMT(red, H, b, false);
	}
}

void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian *HCalib, bool MT) {
	assert(x.size() == CPARS+nFrames*8);

	VecXf xF = x.cast<float>();
	HCalib->step = -x.head<CPARS>();

	//std::cout << "HCalib->step: " << HCalib->step << "\n";

	Mat18f *xAd = new Mat18f[nFrames * nFrames];
	VecCf cstep = xF.head<CPARS>();
	for (EFFrame *h : frames) {
		h->fh->step.head<8>() = -x.segment<8>(CPARS + 8 * h->idxInFrames);
		h->fh->step.tail<2>().setZero();

		for (EFFrame *t : frames) {
			xAd[nFrames * h->idxInFrames + t->idxInFrames] = xF.segment<8>(CPARS + 8 * h->idxInFrames).transpose()
					* adHostF[h->idxInFrames + nFrames * t->idxInFrames]
					+ xF.segment<8>(CPARS + 8 * t->idxInFrames).transpose() * adTargetF[h->idxInFrames + nFrames * t->idxInFrames];
        }
	}

	if (MT)
		red->reduce(boost::bind(&EnergyFunctional::resubstituteFPt, this, cstep, xAd, _1, _2, _3, _4), 0, allPoints.size(), 50);
	else
		resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0, 0);

	delete[] xAd;
}

void EnergyFunctional::resubstituteFPt(const VecCf &cstep, Mat18f *xAd, int min, int max, Vec10 *stats, int tid) {
	for (int k = min; k < max; k++) {
		EFPoint *p = allPoints[k];

		int ngoodres = 0;
		for (EFResidual *r : p->residualsAll)
			if (r->isActive)
				ngoodres++;
		if (ngoodres == 0) {
			p->ph->step = 0;
			continue;
		}
		float b = p->bdSumF;
		b -= cstep.dot(p->Hcd_accAF + p->Hcd_accLF);

		for (EFResidual *r : p->residualsAll) {
			if (!r->isActive)
				continue;
			int xAdIdx = r->hostIDX * nFrames + r->targetIDX;
			b -= xAd[xAdIdx] * r->JpJdF;
            if (!std::isfinite(b)) {
                assert(xAdIdx >= 0 && xAdIdx < nFrames * nFrames);
                assert(!xAd[xAdIdx].hasNaN());
                assert(!r->JpJdF.hasNaN());
                assert(!std::isfinite(b));
            }
		}

		p->ph->step = -b * p->HdiF;
		assert(std::isfinite(p->ph->step));
	}
}

double EnergyFunctional::calcMEnergyF() {

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	VecX delta = getStitchedDeltaF();
	return delta.dot(2 * bM + HM * delta);
}

/**
 * Accumulated energy of linearised points.  Basicly the sum of a function of a point's Residual(from 0), Jababian and Delta.
 *    PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
 */
void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10 *stats, int tid) {

	Accumulator11 E;
	E.initialize();
	VecCf dc = cDelta.cast<float>();

	for (int i = min; i < max; i++) {
		EFPoint *p = allPoints[i];
		float dd = p->deltaF;

		for (EFResidual *r : p->residualsAll) {
			if (!r->isLinearized || !r->isActive)
				continue;

			Mat18f dp = adHTdeltaF[r->hostIDX + nFrames * r->targetIDX];
			RawResidualJacobian *rJ = r->J;

			bool leftToRight = r->hostIDX == r->targetIDX;

			float Jp_delta_x_1 = rJ->Jpdd[0] * dd;
			float Jp_delta_y_1 = rJ->Jpdd[1] * dd;

			// compute Jp*delta
			if (leftToRight) {
				Jp_delta_x_1 += rJ->Jpdc[0].dot(dc);
				Jp_delta_y_1 += rJ->Jpdc[1].dot(dc);
			} else {
				Jp_delta_x_1 += rJ->Jpdc[0].head<4>().dot(dc.head<4>()); // Left camera intrinsics only.
				Jp_delta_x_1 += rJ->Jpdxi[0].dot(dp.head<6>());
				Jp_delta_y_1 += rJ->Jpdc[1].head<4>().dot(dc.head<4>());
				Jp_delta_y_1 += rJ->Jpdxi[1].dot(dp.head<6>());
			}

			__m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
			__m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
			__m128 delta_a = _mm_set1_ps((float) (dp[6]));
			__m128 delta_b = _mm_set1_ps((float) (dp[7]));

			for (int i = 0; i + 3 < patternNum; i += 4) {
				// PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
				__m128 Jdelta = _mm_mul_ps(_mm_load_ps(((float*) (rJ->JIdx)) + i), Jp_delta_x);
				Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*) (rJ->JIdx + 1)) + i), Jp_delta_y));
				Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*) (rJ->JabF)) + i), delta_a));
				Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*) (rJ->JabF + 1)) + i), delta_b));

				__m128 r0 = _mm_load_ps(((float*) &r->res_toZeroF) + i);
				r0 = _mm_add_ps(r0, r0);
				r0 = _mm_add_ps(r0, Jdelta);
				Jdelta = _mm_mul_ps(Jdelta, r0);
				E.updateSSE(Jdelta);
			}
			for (int i = ((patternNum >> 2) << 2); i < patternNum; i++) {
				float Jdelta = rJ->JIdx[0][i] * Jp_delta_x_1 + rJ->JIdx[1][i] * Jp_delta_y_1 + rJ->JabF[0][i] * dp[6]
						+ rJ->JabF[1][i] * dp[7];
				E.updateSingle((float) (Jdelta * (Jdelta + 2 * r->res_toZeroF[i])));
			}
		}
		E.updateSingle(p->deltaF * p->deltaF * p->priorF);
	}
	E.finish();
	(*stats)[0] += E.A;
}

double EnergyFunctional::calcLEnergyF_MT() {
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	double E = 0;
	for (EFFrame *f : frames)
		E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

	E += cDelta.cwiseProduct(cPrior).dot(cDelta);

	red->reduce(boost::bind(&EnergyFunctional::calcLEnergyPt, this, _1, _2, _3, _4), 0, allPoints.size(), 50);

	return E + red->stats[0];
}

void EnergyFunctional::insertResidual(PointFrameResidual *r) {
	EFResidual *efr = new EFResidual(r, r->point->efPoint, r->point->host->efFrame, r->target->efFrame);
	efr->idxInAll = r->point->efPoint->residualsAll.size();
	r->point->efPoint->residualsAll.push_back(efr);

	connectivityMap[(((uint64_t) efr->host->keyFrameID) << 32) + ((uint64_t) efr->target->keyFrameID)][0]++;

	nResiduals++;
	r->efResidual = efr;
}

void EnergyFunctional::insertFrame(FrameHessian *fh, CalibHessian *Hcalib) {
	EFFrame *eff = new EFFrame(this, fh);
	eff->idxInFrames = frames.size();
	frames.push_back(eff);

	nFrames++;
	fh->efFrame = eff;

	assert(HM.cols() == 8*nFrames+CPARS-8);
	bM.conservativeResize(8 * nFrames + CPARS);
	HM.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);
	bM.tail<8>().setZero();
	HM.rightCols<8>().setZero();
	HM.bottomRows<8>().setZero();

	EFIndicesValid = false;
	EFAdjointsValid = false;
	EFDeltaValid = false;

	setAdjointsF(Hcalib);
	makeIDX();

	for (EFFrame *fh2 : frames) {
		connectivityMap[(((uint64_t) eff->keyFrameID) << 32) + ((uint64_t) fh2->keyFrameID)] = Eigen::Vector2i(0, 0);
		if (fh2 != eff)
			connectivityMap[(((uint64_t) fh2->keyFrameID) << 32) + ((uint64_t) eff->keyFrameID)] = Eigen::Vector2i(0, 0);
	}
}

void EnergyFunctional::insertPoint(PointHessian *ph) {
	EFPoint *efp = new EFPoint(ph, ph->host->efFrame);
	efp->idxInPoints = ph->host->efFrame->points.size();
	ph->host->efFrame->points.push_back(efp);

	nPoints++;
	ph->efPoint = efp;

	EFIndicesValid = false;
}

void EnergyFunctional::dropResidual(EFResidual *r) {
	EFPoint *p = r->point;
	assert(r == p->residualsAll[r->idxInAll]);

	p->residualsAll[r->idxInAll] = p->residualsAll.back();
	p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
	p->residualsAll.pop_back();

	if (r->isActive)
		r->host->fh->shell->statistics_goodResOnThis++;
	else
		r->host->fh->shell->statistics_outlierResOnThis++;

	connectivityMap[(((uint64_t) r->host->keyFrameID) << 32) + ((uint64_t) r->target->keyFrameID)][0]--;
	nResiduals--;
	r->data->efResidual = 0;
	delete r;
}
void EnergyFunctional::marginalizeFrame(EFFrame *eff) {

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	assert((int )eff->points.size() == 0);
	int ndim = nFrames * 8 + CPARS - 8;  // new dimension
	int odim = nFrames * 8 + CPARS;  // old dimension

//	VecX eigenvaluesPre = HM.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//

	if ((int) eff->idxInFrames != (int) frames.size() - 1) {
		int io = eff->idxInFrames * 8 + CPARS;	// index of frame to move to end
		int ntail = 8 * (nFrames - eff->idxInFrames - 1);
		assert((io+8+ntail) == nFrames*8+CPARS);

		Vec8 bTmp = bM.segment<8>(io);
		VecX tailTMP = bM.tail(ntail);
		bM.segment(io, ntail) = tailTMP;
		bM.tail<8>() = bTmp;

		MatXX HtmpCol = HM.block(0, io, odim, 8);
		MatXX rightColsTmp = HM.rightCols(ntail);
		HM.block(0, io, odim, ntail) = rightColsTmp;
		HM.rightCols(8) = HtmpCol;

		MatXX HtmpRow = HM.block(io, 0, 8, odim);
		MatXX botRowsTmp = HM.bottomRows(ntail);
		HM.block(io, 0, ntail, odim) = botRowsTmp;
		HM.bottomRows(8) = HtmpRow;
	}

//	// marginalize. First add prior here, instead of to active.
	HM.bottomRightCorner<8, 8>().diagonal() += eff->prior;
	bM.tail<8>() += eff->prior.cwiseProduct(eff->delta_prior);

//	std::cout << std::setprecision(16) << "HMPre:\n" << HM << "\n\n";

	VecX SVec = (HM.diagonal().cwiseAbs() + VecX::Constant(HM.cols(), 10)).cwiseSqrt();
	VecX SVecI = SVec.cwiseInverse();

//	std::cout << std::setprecision(16) << "SVec: " << SVec.transpose() << "\n\n";
//	std::cout << std::setprecision(16) << "SVecI: " << SVecI.transpose() << "\n\n";

	// scale!
	MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
	VecX bMScaled = SVecI.asDiagonal() * bM;

	// invert bottom part!
	Mat88 hpi = HMScaled.bottomRightCorner<8, 8>();
	hpi = 0.5f * (hpi + hpi);
	hpi = hpi.inverse();
	hpi = 0.5f * (hpi + hpi);

	// schur-complement!
	MatXX bli = HMScaled.bottomLeftCorner(8, ndim).transpose() * hpi;
	HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8, ndim);
	bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<8>();

	//unscale!
	HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
	bMScaled = SVec.asDiagonal() * bMScaled;

	// set.
	HM = 0.5 * (HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
	bM = bMScaled.head(ndim);

	// remove from vector, without changing the order!
	for (unsigned int i = eff->idxInFrames; i + 1 < frames.size(); i++) {
		frames[i] = frames[i + 1];
		frames[i]->idxInFrames = i;
	}
	frames.pop_back();
	nFrames--;
	eff->fh->efFrame = 0;

	assert((int)frames.size()*8+CPARS == (int)HM.rows());
	assert((int)frames.size()*8+CPARS == (int)HM.cols());
	assert((int)frames.size()*8+CPARS == (int)bM.size());
	assert((int )frames.size() == (int )nFrames);

//	VecX eigenvaluesPost = HM.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());

//	std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
//	std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

	EFIndicesValid = false;
	EFAdjointsValid = false;
	EFDeltaValid = false;

	makeIDX();
	delete eff;
}

void EnergyFunctional::marginalizePointsF() {
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	allPointsToMarg.clear();
	for (EFFrame *f : frames) {
		for (int i = 0; i < (int) f->points.size(); i++) {
			EFPoint *p = f->points[i];
			if (p->stateFlag == EFPointStatus::PS_MARGINALIZE) {
				p->priorF *= setting_idepthFixPriorMargFac;
				for (EFResidual *r : p->residualsAll)
					if (r->isActive)
						connectivityMap[(((uint64_t) r->host->keyFrameID) << 32) + ((uint64_t) r->target->keyFrameID)][1]++;
				allPointsToMarg.push_back(p);
			}
		}
	}

	accSC_bot->setZero(nFrames);
	accTop_A->setZero(nFrames);
	for (EFPoint *p : allPointsToMarg) {
		accTop_A->addPoint<2>(p, adHTdeltaF, &cDelta); // <2> - Marginalised points.
		accSC_bot->addPoint(p, false);
		removePoint(p);
	}
	MatXX M, Msc;
	VecX Mb, Mbsc;
	accTop_A->stitchDouble(M, Mb, adHost, adTarget, false);
	accSC_bot->stitchDouble(Msc, Mbsc);

	resInM += accTop_A->nres[0];

	MatXX H = M - Msc;
	VecX b = Mb - Mbsc;

	if (setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG) {
		// have a look if prior is there.
		bool haveFirstFrame = false;
		for (EFFrame *f : frames)
			if (f->keyFrameID == 0)
				haveFirstFrame = true;

		if (!haveFirstFrame)
			orthogonalize(&b, &H);

	}

	HM += setting_margWeightFac * H;
	bM += setting_margWeightFac * b;

	if (setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
		orthogonalize(&bM, &HM);

	EFIndicesValid = false;
	makeIDX();
}

void EnergyFunctional::dropPointsF() {

	for (EFFrame *f : frames) {
		for (int i = 0; i < (int) f->points.size(); i++) {
			EFPoint *p = f->points[i];
			if (p->stateFlag == EFPointStatus::PS_DROP) {
				removePoint(p);
				i--;
			}
		}
	}

	EFIndicesValid = false;
	makeIDX();
}

void EnergyFunctional::removePoint(EFPoint *p) {
	for (EFResidual *r : p->residualsAll)
		dropResidual(r);

	EFFrame *h = p->host;
	h->points[p->idxInPoints] = h->points.back();
	h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
	h->points.pop_back();

	nPoints--;
	p->ph->efPoint = 0;

	EFIndicesValid = false;

	delete p;
}

// remove the influence of null space
void EnergyFunctional::orthogonalize(VecX *b, MatXX *H) {
//	VecX eigenvaluesPre = H.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";

	// decide to which nullspaces to orthogonalize.
	std::vector<VecX> ns;
	ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
#ifndef ADD_LR_RESIDUALS
	ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
#endif
//	if(setting_affineOptModeA <= 0)
//		ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
//	if(setting_affineOptModeB <= 0)
//		ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());

	// make Nullspaces matrix
	MatXX N(ns[0].rows(), ns.size());
	for (unsigned int i = 0; i < ns.size(); i++)
		N.col(i) = ns[i].normalized();

	// compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
	Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

	VecX SNN = svdNN.singularValues();
	double minSv = 1e10, maxSv = 0;
	for (int i = 0; i < SNN.size(); i++) {
		if (SNN[i] < minSv)
			minSv = SNN[i];
		if (SNN[i] > maxSv)
			maxSv = SNN[i];
	}
	for (int i = 0; i < SNN.size(); i++) {
		if (SNN[i] > setting_solverModeDelta * maxSv)
			SNN[i] = 1.0 / SNN[i];
		else
			SNN[i] = 0;
	}

	MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); 	// [dim] x 9.
	MatXX NNpiT = N * Npi.transpose(); 	// [dim] x [dim].
			// N * N Pseudo Inverse Transposed. Made Symetric!?
	MatXX NNpiTS = 0.5 * (NNpiT + NNpiT.transpose());	// = N * (N' * N)^-1 * N'.

	// remove the influence of null space
	if (b != 0)
		*b -= NNpiTS * *b;
	if (H != 0)
		*H -= NNpiTS * *H * NNpiTS;

//	std::cout << std::setprecision(16) << "Orth SV: " << SNN.reverse().transpose() << "\n";

//	VecX eigenvaluesPost = H.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());
//	std::cout << "EigPost:: " << eigenvaluesPost.transpose() << "\n";

}

void O0 EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian *HCalib) {
	if (setting_solverMode & SOLVER_USE_GN)
		lambda = 0;
	if (setting_solverMode & SOLVER_FIX_LAMBDA)
		lambda = 1e-5;

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	MatXX HL_top, HA_top, H_sc;
	VecX bL_top, bA_top, bM_top, b_sc;

	accumulateAF_MT(HA_top, bA_top, multiThreading);

	accumulateLF_MT(HL_top, bL_top, multiThreading);

	accumulateSCF_MT(H_sc, b_sc, multiThreading);

	bM_top = (bM + HM * getStitchedDeltaF());

	MatXX HFinal_top;
	VecX bFinal_top;

	if (setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM) {
		// have a look if prior is there.
		bool haveFirstFrame = false;
		for (EFFrame *f : frames)
			if (f->keyFrameID == 0)
				haveFirstFrame = true;

		MatXX HT_act = HL_top + HA_top - H_sc;
		VecX bT_act = bL_top + bA_top - b_sc;

		if (!haveFirstFrame)
			orthogonalize(&bT_act, &HT_act);

		HFinal_top = HT_act + HM;
		bFinal_top = bT_act + bM_top;

		lastHS = HFinal_top;
		lastbS = bFinal_top;

		for (int i = 0; i < 8 * nFrames + CPARS; i++)
			HFinal_top(i, i) *= (1 + lambda);

	} else {
		HFinal_top = HL_top + HM + HA_top;
		bFinal_top = bL_top + bM_top + bA_top - b_sc;

		lastHS = HFinal_top - H_sc;
		lastbS = bFinal_top;

		for (int i = 0; i < 8 * nFrames + CPARS; i++)
			HFinal_top(i, i) *= (1 + lambda);
		HFinal_top -= H_sc * (1.0f / (1 + lambda));
	}

	VecX x;
	if (setting_solverMode & SOLVER_SVD) {
		VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		VecX bFinalScaled = SVecI.asDiagonal() * bFinal_top;
		Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

		VecX S = svd.singularValues();
		double minSv = 1e10, maxSv = 0;
		for (int i = 0; i < S.size(); i++) {
			if (S[i] < minSv)
				minSv = S[i];
			if (S[i] > maxSv)
				maxSv = S[i];
		}

		VecX Ub = svd.matrixU().transpose() * bFinalScaled;
		int setZero = 0;
		for (int i = 0; i < Ub.size(); i++) {
			if (S[i] < setting_solverModeDelta * maxSv) {
				Ub[i] = 0;
				setZero++;
			}

			if ((setting_solverMode & SOLVER_SVD_CUT7 ) && (i >= Ub.size() - 7)) {
				Ub[i] = 0;
				setZero++;
			}

			else
				Ub[i] /= S[i];
		}
		x = SVecI.asDiagonal() * svd.matrixV() * Ub;

	} else {
		VecX SVecI = (HFinal_top.diagonal() + VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
        if (x.hasNaN()) {
            assert(!HFinal_top.hasNaN());
            if (SVecI.hasNaN()) {
                VecX v = HFinal_top.diagonal() + VecX::Constant(HFinal_top.cols(), 10);
                std::cout << "HFinal_top.diagonal:\t" << v.transpose().head(22) << "\n";
                std::cout << "HFinal_top.d.sq.inv:\t" << v.cwiseSqrt().cwiseInverse().transpose().head(22) << "\n";
                std::cout << "             bFinal:\t" << bFinal_top.transpose().head(22) << "\n";
            }
            assert(!SVecI.hasNaN());
            assert(!HFinalScaled.hasNaN());
            assert(!bFinal_top.hasNaN());
            assert(!x.hasNaN());
        }
	}

	if ((setting_solverMode & SOLVER_ORTHOGONALIZE_X )
			|| (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER ))) {
		VecX xOld = x;
		orthogonalize(&x, 0);
	}

	lastX = x;

	//resubstituteF(x, HCalib);
	resubstituteF_MT(x, HCalib, multiThreading);
}

void EnergyFunctional::makeIDX() {
	for (unsigned int idx = 0; idx < frames.size(); idx++)
		frames[idx]->idxInFrames = idx;

	allPoints.clear();

	for (EFFrame *f : frames)
		for (EFPoint *p : f->points) {
			allPoints.push_back(p);
			for (EFResidual *r : p->residualsAll) {
				r->hostIDX = r->host->idxInFrames;
				r->targetIDX = r->target->idxInFrames;
			}
		}

	EFIndicesValid = true;
}

VecX EnergyFunctional::getStitchedDeltaF() const {
	VecX d = VecX(CPARS + nFrames * 8);
	d.head<CPARS>() = cDelta;
	for (int h = 0; h < nFrames; h++)
		d.segment<8>(CPARS + 8 * h) = frames[h]->delta;
	return d;
}

}
