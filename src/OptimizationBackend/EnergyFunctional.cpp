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


#include <Eigen/Core>
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/IMU.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "util/OptimisationUtils.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

bool EFAdjointsValid = false;
bool EFIndicesValid = false;
bool EFDeltaValid = false;

void EnergyFunctional::setAdjointsF() {
	if (adHost != 0)
		delete[] adHost;
	if (adTarget != 0)
		delete[] adTarget;
	adHost = new Mat88[nFrames * nFrames];
	adTarget = new Mat88[nFrames * nFrames];

	for (int h = 0; h < nFrames; h++)
		for (int t = 0; t < nFrames; t++) {
			FrameHessian *host = frames[h]->data;
			FrameHessian *target = frames[t]->data;

			SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

			Mat88 AH = Mat88::Identity();
			Mat88 AT = Mat88::Identity();

			AH.topLeftCorner<6, 6>() = -hostToTarget.Adj().transpose();
			AT.topLeftCorner<6, 6>() = Mat66::Identity();

			Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(), target->aff_g2l_0())
				.cast<float>();
			AT(6, 6) = -affLL[0];
			AH(6, 6) = affLL[0];
			AT(7, 7) = -1;
			AH(7, 7) = affLL[0];

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
		}

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

EnergyFunctional::EnergyFunctional(const double imuWeight, CalibHessian &HCalib, MetricWorldHessian &HWorld, ImuBiasHessian &HBias) :
		imuWeightSquared(imuWeight * imuWeight), HCalib(HCalib), HWorld(HWorld), HBias(HBias) {
	adHost=0;
	adTarget=0;

	red=0;

	adHostF=0;
	adTargetF=0;
	adHTdeltaF=0;

	nFrames = nResiduals = nPoints = 0;

	accSSE_top_L = new AccumulatedTopHessianSSE();
	accSSE_top_A = new AccumulatedTopHessianSSE();
	accSSE_bot = new AccumulatedSCHessianSSE();

	cPrior = VecC::Constant(setting_initialCalibHessian);
	wPrior << setting_initialScaleHessian,setting_initialDirABHessian,setting_initialDirABHessian,setting_initialDirABHessian;
	bPrior << setting_initialBiasAccHessian,setting_initialBiasAccHessian,setting_initialBiasAccHessian,
		setting_initialBiasGyroHessian,setting_initialBiasGyroHessian,setting_initialBiasGyroHessian;

	resInA = resInL = resInM = 0;

	// Marginalised DSO factors only.
	HMDso = MatXX::Zero(CPARS, CPARS);
	bMDso = VecX::Zero(CPARS);

	HMImuCurr = std::make_unique<MatXX>(ICPARS, ICPARS);
	HMImuCurr->setZero();
	bMImuCurr = std::make_unique<VecX>(ICPARS);
	bMImuCurr->setZero();
	HMImuHalf = std::make_unique<MatXX>(ICPARS, ICPARS);
	HMImuHalf->setZero();
	bMImuHalf = std::make_unique<VecX>(ICPARS);
	bMImuHalf->setZero();

	lastUpper = false;
	sMiddle = sLast = HWorld.TMetricDso.scale();
	di = dmin;

	currentLambda=0;
}

EnergyFunctional::~EnergyFunctional()
{
	for(EFFrame* f : frames)
	{
		for(EFPoint* p : f->points)
		{
			for(EFResidual* r : p->residualsAll)
			{
				r->data->efResidual=0;
				delete r;
			}
			p->data->efPoint=0;
			delete p;
		}
		f->data->efFrame=0;
		delete f;
	}

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;


	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;
	if(adHTdeltaF != 0) delete[] adHTdeltaF;



	delete accSSE_top_L;
	delete accSSE_top_A;
	delete accSSE_bot;
}




void EnergyFunctional::setDeltaF()
{
	if(adHTdeltaF != 0) delete[] adHTdeltaF;
	adHTdeltaF = new Mat18f[nFrames*nFrames];
	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			int idx = h+t*nFrames;
			// Remove velocity..
			VecIF hZero = frames[h]->data->get_state_minus_stateZero();
			VecIF tZero = frames[t]->data->get_state_minus_stateZero();
			hZero.segment<2>(6) = hZero.tail<2>();
			tZero.segment<2>(6) = tZero.tail<2>();
			adHTdeltaF[idx] = hZero.head<FPARS>().cast<float>().transpose() * adHostF[idx]
					        +tZero.head<FPARS>().cast<float>().transpose() * adTargetF[idx];
		}

	cDelta = HCalib.value_minus_value_zero;
	wDelta = HWorld.get_value_minus_valueZero();
	bDelta = HBias.get_value_minus_valueZero();

	cDeltaF = cDelta.cast<float>();

	for(EFFrame* f : frames)
	{
		f->delta = f->data->get_state_minus_stateZero();
		f->delta_prior = (f->data->get_state() - f->data->getPriorZero());

		for(EFPoint* p : f->points)
			p->deltaF = p->data->idepth-p->data->idepth_zero;
	}

	EFDeltaValid = true;
}

// accumulates & shifts Active.
void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
				accSSE_top_A, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_top_A->stitchDoubleMT(red,H,b,this,true);
		resInA = accSSE_top_A->nres[0];
	}
	else
	{
		accSSE_top_A->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_top_A->addPoint<0>(p,this);
		accSSE_top_A->stitchDoubleMT(red,H,b,this,false);
		resInA = accSSE_top_A->nres[0];
	}
}

// accumulates & shifts L.
void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
				accSSE_top_L, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_top_L->stitchDoubleMT(red,H,b,this,true);
		resInL = accSSE_top_L->nres[0];
	}
	else
	{
		accSSE_top_L->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_top_L->addPoint<1>(p,this);
		accSSE_top_L->stitchDoubleMT(red,H,b,this,false);
		resInL = accSSE_top_L->nres[0];
	}
}


void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal,
				accSSE_bot, &allPoints, true,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_bot->stitchDoubleMT(red,H,b,this,true);
	}
	else
	{
		accSSE_bot->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_bot->addPoint(p, true);
		accSSE_bot->stitchDoubleMT(red, H, b,this,false);
	}
}

[[clang::optnone]]
void EnergyFunctional::resubstituteF_MT(VecX x, bool MT)
{
	assert(x.size() == ICPARS+nFrames*IFPARS);
	assert(!x.hasNaN());

	VecXf xF = x.cast<float>();
	HCalib.step = - x.head<CPARS>();
//	HWorld.step(0) = - x(CPARS);
//	HWorld.step(1) = - x(CPARS+1);
//	HWorld.step(2) = - x(CPARS+2);
//	HWorld.step(3) = - x(CPARS+3);
	HWorld.step = - x.segment<IWPARS>(CPARS);
	HBias.step = - x.segment<IBPARS>(CPARS + IWPARS);
	std::cout << "HWorld step:" << HWorld.step.transpose().format(MatlabFmt);
	std::cout << "HBias step:" << HBias.step.transpose().format(MatlabFmt);

	Mat18f* xAd = new Mat18f[nFrames*nFrames];
	VecCf cstep = xF.head<CPARS>();

	for(EFFrame* h : frames)
	{
		h->data->step = - x.segment<IFPARS>(ICPARS+IFPARS*h->idx);
		std::cout << "Frame " << h->frameID << " step:" << h->data->step.transpose().format(MatlabFmt);

		for(EFFrame* t : frames) {
			Vec8f xh;
			xh.head<6>() = xF.segment<6>(ICPARS+IFPARS*h->idx);
			xh.tail<2>() = xF.segment<2>(ICPARS+IFPARS*h->idx+9);
			Vec8f xt;
			xt.head<6>() = xF.segment<6>(ICPARS+IFPARS*t->idx);
			xt.tail<2>() = xF.segment<2>(ICPARS+IFPARS*t->idx+9);
			xAd[nFrames * h->idx + t->idx] =
				xt.transpose() * adHostF[h->idx + nFrames * t->idx] + xh.transpose() * adTargetF[h->idx + nFrames * t->idx];
		}
	}

	if(MT)
		red->reduce(boost::bind(&EnergyFunctional::resubstituteFPt,
						this, cstep, xAd,  _1, _2, _3, _4), 0, allPoints.size(), 50);
	else
		resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0,0);

	delete[] xAd;
}

void EnergyFunctional::resubstituteFPt(
        const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
	{
		EFPoint* p = allPoints[k];

		int ngoodres = 0;
		for(EFResidual* r : p->residualsAll) if(r->isActive()) ngoodres++;
		if(ngoodres==0)
		{
			p->data->step = 0;
			continue;
		}
		float b = p->bdSumF;
		b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);

		for(EFResidual* r : p->residualsAll)
		{
			if(!r->isActive()) continue;
			b -= xAd[r->hostIDX*nFrames + r->targetIDX] * r->JpJdF;
		}

		p->data->step = - b*p->HdiF;
		assert(std::isfinite(p->data->step));
	}
}


double EnergyFunctional::calcMEnergyF() {
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	VecX delta(bMDso.size());
	getStitchedDeltaF(delta);
	return delta.dot(2*bMDso + HMDso*delta);
}


void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10* stats, int tid)
{
	Accumulator11 E;
	E.initialize();
	VecCf dc = cDelta.cast<float>();

	for(int i=min;i<max;i++)
	{
		EFPoint* p = allPoints[i];
		float dd = p->deltaF;

		for(EFResidual* r : p->residualsAll)
		{
			if(!r->isLinearized || !r->isActive()) continue;

			Mat18f dp = adHTdeltaF[r->hostIDX+nFrames*r->targetIDX];
			RawResidualJacobian* rJ = r->J;



			// compute Jp*delta
			float Jp_delta_x_1 =  rJ->Jpdxi[0].dot(dp.head<6>())
						   +rJ->Jpdc[0].dot(dc)
						   +rJ->Jpdd[0]*dd;

			float Jp_delta_y_1 =  rJ->Jpdxi[1].dot(dp.head<6>())
						   +rJ->Jpdc[1].dot(dc)
						   +rJ->Jpdd[1]*dd;

			__m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
			__m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
			__m128 delta_a = _mm_set1_ps((float)(dp[6]));
			__m128 delta_b = _mm_set1_ps((float)(dp[7]));

			for(int i=0;i+3<patternNum;i+=4)
			{
				// PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
				__m128 Jdelta =            _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx))+i),Jp_delta_x);
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx+1))+i),Jp_delta_y));
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF))+i),delta_a));
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF+1))+i),delta_b));

				__m128 r0 = _mm_load_ps(((float*)&r->res_toZeroF)+i);
				r0 = _mm_add_ps(r0,r0);
				r0 = _mm_add_ps(r0,Jdelta);
				Jdelta = _mm_mul_ps(Jdelta,r0);
				E.updateSSENoShift(Jdelta);
			}
			for(int i=((patternNum>>2)<<2); i < patternNum; i++)
			{
				float Jdelta = rJ->JIdx[0][i]*Jp_delta_x_1 + rJ->JIdx[1][i]*Jp_delta_y_1 +
								rJ->JabF[0][i]*dp[6] + rJ->JabF[1][i]*dp[7];
				E.updateSingleNoShift((float)(Jdelta * (Jdelta + 2*r->res_toZeroF[i])));
			}
		}
		E.updateSingle(p->deltaF*p->deltaF*p->priorF);
	}
	E.finish();
	(*stats)[0] += E.A;
}




double EnergyFunctional::calcLEnergyF_MT()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	double E = 0;
	for(EFFrame* f : frames) {
		E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

		assert((f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior)) == (
			f->delta_prior.head<9>().cwiseProduct(f->prior.head<9>()).dot(f->delta_prior.head<9>()) +
			f->delta_prior.tail<2>().cwiseProduct(f->prior.tail<2>()).dot(f->delta_prior.tail<2>())));
	}

	E += cDelta.cwiseProduct(cPrior).dot(cDelta);

	red->reduce(boost::bind(&EnergyFunctional::calcLEnergyPt,
			this, _1, _2, _3, _4), 0, allPoints.size(), 50);


	return E+red->stats[0];
}



EFResidual* EnergyFunctional::insertResidual(PointFrameResidual* r)
{
	EFResidual* efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
	efr->idxInAll = r->point->efPoint->residualsAll.size();
	r->point->efPoint->residualsAll.push_back(efr);

    connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;

	nResiduals++;
	r->efResidual = efr;
	return efr;
}

template<int N>inline void growHb(MatXX &H, VecX &b) {
	const int newSize = b.size() + N;
	H.conservativeResize(newSize, newSize);
	H.rightCols<N>().setZero();
	H.bottomRows<N>().setZero();
	b.conservativeResize(newSize);
	b.tail<N>().setZero();
}

EFFrame* EnergyFunctional::insertFrame(FrameHessian* fh)
{
	EFFrame* eff = new EFFrame(fh);
	eff->idx = frames.size();
	frames.push_back(eff);
	fh->efFrame = eff;

	std::cout << "sizes: " << HMDso.cols() << ", " << HMImuCurr->cols() << ", " << (ICPARS+IFPARS*nFrames) << "\n";
	assert(HMDso.cols() == CPARS+FPARS*nFrames);
	assert(HMImuCurr->cols() == ICPARS+IFPARS*nFrames);
	assert(HMImuHalf->cols() == ICPARS+IFPARS*nFrames);

	growHb<FPARS>(HMDso,bMDso);
	growHb<IFPARS>(*HMImuCurr, *bMImuCurr);
	growHb<IFPARS>(*HMImuHalf, *bMImuHalf);

	nFrames++;

	assert(HMDso.cols() == CPARS+FPARS*nFrames);
	assert(HMImuCurr->cols() == ICPARS+IFPARS*nFrames);
	assert(HMImuHalf->cols() == ICPARS+IFPARS*nFrames);

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	setAdjointsF();
	makeIDX();


	for(EFFrame* fh2 : frames)
	{
        connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0,0);
		if(fh2 != eff)
            connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0,0);
	}

	return eff;
}
EFPoint* EnergyFunctional::insertPoint(PointHessian* ph)
{
	EFPoint* efp = new EFPoint(ph, ph->host->efFrame);
	efp->idxInPoints = ph->host->efFrame->points.size();
	ph->host->efFrame->points.push_back(efp);

	nPoints++;
	ph->efPoint = efp;

	EFIndicesValid = false;

	return efp;
}


void EnergyFunctional::dropResidual(EFResidual* r)
{
	EFPoint* p = r->point;
	assert(r == p->residualsAll[r->idxInAll]);

	p->residualsAll[r->idxInAll] = p->residualsAll.back();
	p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
	p->residualsAll.pop_back();


	if(r->isActive())
		r->host->data->shell->statistics_goodResOnThis++;
	else
		r->host->data->shell->statistics_outlierResOnThis++;


    connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
	nResiduals--;
	r->data->efResidual=0;
	delete r;
}

[[clang::optnone]]
void EnergyFunctional::marginalizeFrame(EFFrame* fh)
{
	std::cout << "marginalizeFrame:" << fh->idx << "\n";
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	assert((int)fh->points.size()==0);

	marginalizeSchur<FPARS>(HMDso, bMDso, CPARS + FPARS * fh->idx, fh->dsoPrior(), fh->dsoDeltaPrior());

	const int nDimFull = ICPARS + IFPARS * (frames.size() - 1);
	const float sCurrent = HWorld.TMetricDso.scale();
	const bool upper = sCurrent > sMiddle;

	if (upper != lastUpper) {
		std::cout << "marginalizeFrame():Scale changing sides.\n";
		HMImuHalf->resize(nDimFull,nDimFull);
		HMImuHalf->setZero();
		bMImuHalf->resize(nDimFull);
		bMImuHalf->setZero();
	}

	if (sCurrent > sMiddle * di || sCurrent < sMiddle / di) {
		std::cout << "marginalizeFrame():Scale changing middle.\n";
		assert(upper == lastUpper); // Moving the scale middle shouldn't happen at the same time as changing sides.
		HMImuCurr.swap(HMImuHalf);
		bMImuCurr.swap(bMImuHalf);
		HMImuHalf->resize(nDimFull,nDimFull);
		HMImuHalf->setZero();
		bMImuHalf->resize(nDimFull);
		bMImuHalf->setZero();

		sMiddle *= (sCurrent > sMiddle * di) ? di : 1/di;
	}

	double ss = sCurrent >= sLast ? sCurrent / sLast : sLast /sCurrent;
	di = dmin;
	while(di <= ss) {
		di *= dmin;
	}

	std::cout << "di:" << di << " sCurrent:" << sCurrent << " sCurrent/sLast:" << ss << " upper(last):" << upper << "(" << lastUpper << ")\n";

	lastUpper = upper;
	sLast = sCurrent;

	if (bMImuCurr->size() != nDimFull) {
		marginalizeSchur<IFPARS>(*HMImuCurr, *bMImuCurr, ICPARS + IFPARS * fh->idx, VecIF::Ones(), VecIF::Ones());
	}
	if (bMImuHalf->size() != nDimFull) {
		marginalizeSchur<IFPARS>(*HMImuHalf, *bMImuHalf, ICPARS + IFPARS * fh->idx, VecIF::Ones(), VecIF::Ones());
	}

	// remove from vector, without changing the order!
	for(unsigned int i=fh->idx; i+1<frames.size();i++)
	{
		frames[i] = frames[i+1];
		frames[i]->idx = i;
	}
	frames.pop_back();
	nFrames--;
	fh->data->efFrame=0;

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	makeIDX();

	assert(nFrames == frames.size());

	const int ndim = frames.size()*FPARS+CPARS;
	assert(ndim == HMDso.rows());
	assert(ndim == HMDso.cols());
	assert(ndim == bMDso.size());
	assert(nDimFull == HMImuCurr->rows());
	assert(nDimFull == HMImuCurr->cols());
	assert(nDimFull == bMImuCurr->size());
	assert(nDimFull == HMImuHalf->rows());
	assert(nDimFull == HMImuHalf->cols());
	assert(nDimFull == bMImuHalf->size());
	assert(frames.size() == nFrames);
}

void addDsoToFullB(VecX &dsoB, VecX &fullB) {
	const int nFrames = (dsoB.size() - CPARS) / FPARS;
	fullB.head<CPARS>() = dsoB.head<CPARS>();
	for (int f = 0 ; f < nFrames ; f++) {
		fullB.segment<6>(ICPARS + f * IFPARS) = dsoB.segment<6>(CPARS + f * FPARS);
		fullB.segment<2>(ICPARS + f * IFPARS + 9) = dsoB.segment<2>(CPARS + f * FPARS + 6);
	}
}

void addDsoToFullH(MatXX &dsoH, MatXX &fullH) {
	const int nFrames = (dsoH.cols() - CPARS) / FPARS;
	fullH.topLeftCorner<CPARS,CPARS>() = dsoH.topLeftCorner<CPARS,CPARS>();

	for (int i = 0 ; i < nFrames ; i++) {
		int di = CPARS + i * FPARS;
		int fi = ICPARS + i * IFPARS;

		fullH.block<CPARS,6>(0,fi) = dsoH.block<CPARS,6>(0,di);
		fullH.block<CPARS,2>(0,fi+9) = dsoH.block<CPARS,2>(0,di+6);
		fullH.block<6,CPARS>(fi,0) = dsoH.block<6,CPARS>(di,0);
		fullH.block<2,CPARS>(fi+9,0) = dsoH.block<2,CPARS>(di+6,0);

		for (int j = 0 ; j < nFrames ; j++) {
			int dj = CPARS + j * FPARS;
			int fj = ICPARS + j * IFPARS;

			fullH.block<6,6>(fi,fj) = dsoH.block<6,6>(di,dj);
			fullH.block<2,2>(fi+9,fj+9) = dsoH.block<2,2>(di+6,dj+6);
			fullH.block<6,2>(fi,fj+9) = dsoH.block<6,2>(di,dj+6);
			fullH.block<2,6>(fi+9,fj) = dsoH.block<2,6>(di+6,dj);
		}
	}
}

void EnergyFunctional::marginalizePointsF()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);


	allPointsToMarg.clear();
	for(EFFrame* f : frames)
	{
		for(int i=0;i<(int)f->points.size();i++)
		{
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_MARGINALIZE)
			{
				p->priorF *= setting_idepthFixPriorMargFac;
				for(EFResidual* r : p->residualsAll)
					if(r->isActive())
                        connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;
				allPointsToMarg.push_back(p);
			}
		}
	}

	accSSE_bot->setZero(nFrames);
	accSSE_top_A->setZero(nFrames);
	for(EFPoint* p : allPointsToMarg)
	{
		accSSE_top_A->addPoint<2>(p,this);
		accSSE_bot->addPoint(p,false);
		removePoint(p);
	}
	MatXX M, Msc;
	VecX Mb, Mbsc;
	accSSE_top_A->stitchDouble(M,Mb,this,false);
	accSSE_bot->stitchDouble(Msc,Mbsc,this);

	resInM+= accSSE_top_A->nres[0];

	MatXX HDso =  M-Msc;
    VecX bDso =  Mb-Mbsc;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG) {
		// have a look if prior is there.
		bool haveFirstFrame = false;
		for (EFFrame *f : frames)
			if (f->frameID == 0)
				haveFirstFrame = true;

		if (!haveFirstFrame)
			orthogonalize(&bDso, &HDso);
	}

	HMDso += setting_margWeightFac*HDso;
	bMDso += setting_margWeightFac*bDso;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
		orthogonalize(&bDso, &HDso);

	EFIndicesValid = false;
	makeIDX();
}

void EnergyFunctional::dropPointsF()
{


	for(EFFrame* f : frames)
	{
		for(int i=0;i<(int)f->points.size();i++)
		{
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_DROP)
			{
				removePoint(p);
				i--;
			}
		}
	}

	EFIndicesValid = false;
	makeIDX();
}


void EnergyFunctional::removePoint(EFPoint* p)
{
	for(EFResidual* r : p->residualsAll)
		dropResidual(r);

	EFFrame* h = p->host;
	h->points[p->idxInPoints] = h->points.back();
	h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
	h->points.pop_back();

	nPoints--;
	p->data->efPoint = 0;

	EFIndicesValid = false;

	delete p;
}


/**
 * Orthogonalize given b and H to computed null-spaces.
 * Stops the optimization drifting in the unobservable "directions" of scale, and absolute position/orientation.
 * @param b
 * @param H
 */
 [[clang::optnone]]
void EnergyFunctional::orthogonalize(VecX* b, MatXX* H)
{
//	VecX eigenvaluesPre = H.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
	const bool fullSize = (b->rows() == ICPARS + IFPARS * nFrames);
	assert(fullSize);

	const int C = fullSize ? ICPARS : CPARS;
	const int F = fullSize ? IFPARS : FPARS;
	const int rowsFull = ICPARS + IFPARS * nFrames;
	const int rowsDso = CPARS + CPARS * nFrames;
	const int cols = lastNullspaces_pose.size() + lastNullspaces_scale.size();
	MatXX NFull(rowsFull, cols);
	//MatXX NDso(rowsDso, cols);

	int Nc = 0;
	for (const VecX &v : lastNullspaces_pose) {
		NFull.col(Nc++) = v.normalized();
	}
	for (const VecX &v : lastNullspaces_scale) {
		NFull.col(Nc++) = v.normalized();
	}

//		VecX col(rows);
//		col.head<CPARS>() = v.head<CPARS>();
//		for (int f=0 ; f < nFrames ; f++) {
//			col.segment(C + F * f, F) = v.segment(CPARS + FPARS * f, F);
//		}
//		N.col(Nc++) = col.normalized();
//	}
//	for (const VecX &v : lastNullspaces_scale) {
//		VecX col(rows);
//		col.head<CPARS>() = v.head<CPARS>();
//		for (int f=0 ; f < nFrames ; f++) {
//			col.segment<FPARS>(C + F * f) = v.segment<FPARS>(CPARS + FPARS * f);
//		}
//		N.col(Nc++) = col.normalized();
//	}

	// decide to which nullspaces to orthogonalize.
	std::vector<VecX> ns;
	ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
	ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
//	if(setting_affineOptModeA <= 0)
//		ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
//	if(setting_affineOptModeB <= 0)
//		ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());


	// make Nullspaces matrix
	MatXX N2(ns[0].rows(), ns.size());
	for(unsigned int i=0;i<ns.size();i++)
		N2.col(i) = ns[i].normalized();

	if (fullSize && !NFull.isApprox(N2)) {
		std::cout << "N=...\n" << NFull.format(MatlabFmt);
		std::cout << "N2=...\n" << N2.format(MatlabFmt);
		abort();
	}

	// compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
	Eigen::JacobiSVD<MatXX> svdNN(NFull, Eigen::ComputeThinU | Eigen::ComputeThinV);

	VecX SNN = svdNN.singularValues();
	double minSv = 1e10, maxSv = 0;
	for(int i=0;i<SNN.size();i++)
	{
		if(SNN[i] < minSv) minSv = SNN[i];
		if(SNN[i] > maxSv) maxSv = SNN[i];
	}
	for(int i=0;i<SNN.size();i++)
		{ if(SNN[i] > setting_solverModeDelta*maxSv) SNN[i] = 1.0 / SNN[i]; else SNN[i] = 0; }

	// Pseudo Inverse..
	MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); 	// [dim] x 9.
	MatXX NNpiT = NFull*Npi.transpose(); 	// [dim] x [dim].
	MatXX NNpiTS = 0.5*(NNpiT + NNpiT.transpose());	// = N * (N' * N)^-1 * N'.

	if(b!=0) *b -= NNpiTS * *b;
	if(H!=0) *H -= NNpiTS * *H * NNpiTS;


//	std::cout << std::setprecision(16) << "Orth SV: " << SNN.reverse().transpose() << "\n";

//	VecX eigenvaluesPost = H.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());
//	std::cout << "EigPost:: " << eigenvaluesPost.transpose() << "\n";

}

void EnergyFunctional::solveSystemF(int iteration, double lambda)
{
	if(setting_solverMode & SOLVER_USE_GN) lambda=0;
	if(setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	MatXX HL_top, HA_top, H_sc;
	VecX  bL_top, bA_top, b_sc;

	accumulateAF_MT(HA_top, bA_top,multiThreading);
	accumulateLF_MT(HL_top, bL_top,multiThreading);
	accumulateSCF_MT(H_sc, b_sc,multiThreading);


	const int dsoSize = CPARS + nFrames * FPARS;
	/* Allocated and compute top left H with IMU factors.
	 * 4 - DSO's Calib
	 * 1 - scale
	 * 2 - Gravity dir. like in ORB_SLAM3
	 * 6 - IMU Accel & Gyro biases
	 *
	 * 6 - DSO's Pose. Per frame
	 * 3 - Velocity. Per frame
	 * 2 - DSO's Aff. Per frame
	 */
	const int fullSize = ICPARS + nFrames * IFPARS;
	MatXX HFull_top = MatXX::Zero(fullSize, fullSize);
	VecX bFull_top = VecX::Zero(fullSize);
	if (nFrames>=2) {
		double energy = 0;
		for (long unsigned int i = 0; i < frames.size() - 1; i++) {
			energy += addImuFactors(i, HFull_top, bFull_top);
		}
		assert(!HFull_top.hasNaN());
		assert(!bFull_top.hasNaN());
	}

//	std::cout << "HFull_top=...\n" << HFull_top.format(MatlabFmt) << "\n";
//	std::cout << "bFull_top=...\n" << bFull_top.transpose().format(MatlabFmt) << "\n";

	assert(!HMDso.hasNaN());
	assert(!bMDso.hasNaN());

	VecX delta(bMDso.size());
	getStitchedDeltaF(delta);
	MatXX bMDso_top = bMDso + HMDso * delta;
	assert(!bMDso_top.hasNaN());

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM) {
		abort(); // TODO IMU factors.

//		// have a look if prior is there.
//		bool haveFirstFrame = false;
//		for(EFFrame* f : frames) if(f->frameID==0) haveFirstFrame=true;
//
//		MatXX HDso =  HL_top + HA_top - H_sc;
//		VecX bDso =   bL_top + bA_top - b_sc;
//
//		// HM and bM_top are full so this won't work.
//		HDso += HM;
//		bDso += bM_top;
//
//		addDsoToFullH(HFull_top, HDso);
//		addDsoToFullB(bFull_top, bDso);
//
//		if(!haveFirstFrame)
//			orthogonalize(&bDso, &HDso); // needs handle full H and b.
//
//		lastHS = HFull_top;
//		lastbS = bFull_top;
//
//		for(int i=0;i<dsoSize;i++) HFull_top(i, i) *= (1+lambda);
	} else {
		MatXX HDso = HL_top + HA_top + HMDso;
		addDsoToFullH(HDso, HFull_top);
		assert(!HFull_top.hasNaN());

		VecX bDso = bL_top + bA_top - b_sc + bMDso_top;
		addDsoToFullB(bDso, bFull_top);
		assert(!bFull_top.hasNaN());

		if(HMImuCurr->sum() != 0) {
			std::cout << "HMImuCurr=...\n" << HMImuCurr->format(MatlabFmt);
			abort();
		}
		assert(bMImuCurr->sum() == 0);
		HFull_top += *HMImuCurr;
		bFull_top += *bMImuCurr;

		addPrior(HFull_top, bFull_top);

		MatXX HFull_sc = MatXX::Zero(fullSize, fullSize);
		addDsoToFullH(H_sc, HFull_sc);
		assert(!HFull_sc.hasNaN());

		lastHS = HFull_top - HFull_sc;
		lastbS = bFull_top;

		for(int i=0; i<fullSize; i++) HFull_top(i, i) *= (1+lambda);
		HFull_top -= HFull_sc * (1.0f/(1+lambda));
	}

	assert(!HFull_top.hasNaN());
	assert(!bFull_top.hasNaN());

	VecX x;
	if(setting_solverMode & SOLVER_SVD)
	{
		abort(); // Code below not updated to handle IMU expanded H and b
		VecX SVecI = HFull_top.diagonal().cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFull_top * SVecI.asDiagonal();
		VecX bFinalScaled  = SVecI.asDiagonal() * bFull_top;
		Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

		VecX S = svd.singularValues();
		double minSv = 1e10, maxSv = 0;
		for(int i=0;i<S.size();i++)
		{
			if(S[i] < minSv) minSv = S[i];
			if(S[i] > maxSv) maxSv = S[i];
		}

		VecX Ub = svd.matrixU().transpose()*bFinalScaled;
		int setZero=0;
		for(int i=0;i<Ub.size();i++)
		{
			if(S[i] < setting_solverModeDelta*maxSv)
			{ Ub[i] = 0; setZero++; }

			if((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size()-7))
			{ Ub[i] = 0; setZero++; }

			else Ub[i] /= S[i];
		}
		x = SVecI.asDiagonal() * svd.matrixV() * Ub;
	}
	else
	{
		VecX SVecI = (HFull_top.diagonal()+VecX::Constant(HFull_top.cols(), 10)).cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFull_top * SVecI.asDiagonal();
		x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFull_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
	}

	if((setting_solverMode & SOLVER_ORTHOGONALIZE_X) || (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER)))
	{
		orthogonalize(&x, 0);
	}

	lastX = x;

	currentLambda= lambda;
	resubstituteF_MT(x, multiThreading);
	currentLambda=0;
}

const SE3 EnergyFunctional::dsoCamPoseToMetricImuPose(const SE3& T_dso_cam) {
	// Where the IMU is in the metric world for the given camera pose(in DSO world)
	return SE3(HWorld.TMetricDso.matrix() * T_dso_cam.matrix() * HWorld.TDsoMetric.matrix() * HWorld.TMetricCamImu.matrix());
//	Sim3 TdsoCamImu = Sim3(TBaseCam.inverse().matrix());
//	TdsoCamImu.setScale(HWorld.TMetricDso.scale()); // Scale TBaseCam^-1 to DSO Frame.
//	Sim3 expected = Sim3(HWorld.TMetricDso.matrix() * T_dso_cam.matrix() * TdsoCamImu.matrix() * HWorld.TDsoMetric.matrix());
//
//	Sim3 meh = Sim3(HWorld.TMetricDso.matrix() * T_dso_cam.matrix() * HWorld.TDsoMetric.matrix() * HWorld.TMetricCamImu.matrix());
//
//	if(!expected.matrix().isApprox(meh.matrix())) {
//		std::cerr << "TdsoCamImu=...\n" << TdsoCamImu.matrix().format(MatlabFmt);
//		std::cerr << "expected=...\n" << expected.matrix().format(MatlabFmt);
//		std::cerr << "meh=...\n" << meh.matrix().format(MatlabFmt);
//		abort();
//	}
//	return SE3(expected.matrix());
};

/**
 * Responsible for IMU's J and r indexing.
 */
struct JrWrapper {

	// Jacobian of residual errors for frame i to j:
	//   Translation 3
	//   Rotation 3
	//   Velocity 3
	// against state estimate:
	//   Scale 1
	//   Gravity dir. 3(!!)2
	//   Bias Acc 3
	//   Bias Gyro 3
	//   Translation, rotation, velocity for frames i and j. 9 + 9
	Eigen::Matrix<double,TRVPARS,28> J = Eigen::Matrix<double,TRVPARS,28>::Zero();
	Eigen::Matrix<double,TRVPARS,1> r = Eigen::Matrix<double,TRVPARS,1>::Zero();
	// Handle bias random walk separately
	Eigen::Matrix<double,6,1> rBias = Eigen::Matrix<double,6,1>::Zero();

	// r
	auto rTra() { return r.segment<3>(0); }
	auto rRot() { return r.segment<3>(3); }
	auto rVel() { return r.segment<3>(6); }

	auto rBiasA() { return rBias.segment<3>(0); }
	auto rBiasG() { return rBias.segment<3>(3); }


	// J. Ordered by column left to right
	// Names are column or rowColumn
	auto worldscale() { return J.block<9,1>(0,0); }
//	auto worldrotA() { return J.block<9,1>(0,1); }
//	auto worldrotB() { return J.block<9,1>(0,2); }
	auto worldrot() { return J.block<9,3>(0,1); }

	auto biasA() { return J.block<9,3>(0,4); }
	auto biasG() { return J.block<9,3>(0,7); }
	auto trnBiasa() { return J.block<3,3>(0,4); }
	auto velBiasa() { return J.block<3,3>(6,4); }
	auto trnBiasg() { return J.block<3,3>(0,7); }
	auto rotBiasg() { return J.block<3,3>(3,7); }
	auto velBiasg() { return J.block<3,3>(6,7); }

	auto trnroti() { return J.block<9,6>(0,10); }
	auto trni() { return J.block<9,3>(0,10); }
	auto roti() { return J.block<9,3>(0,13); }
	auto veli() { return J.block<9,3>(0,16); }
	auto blockDsoi() { return J.block<6,6>(0,10); }
	auto trnTrni() { return J.block<3,3>(0,10); }
	auto trnRoti() { return J.block<3,3>(0,13); }
	auto rotRoti() { return J.block<3,3>(3,13); }
	auto velRoti() { return J.block<3,3>(6,13); }
	auto trnVeli() { return J.block<3,3>(0,16); }
	auto velVeli() { return J.block<3,3>(6,16); }

	auto trnrotj() { return J.block<9,6>(0,19); }
	auto trnj() { return J.block<9,3>(0,19); }
	auto rotj() { return J.block<9,3>(0,22); }
	auto velj() { return J.block<9,3>(0,25); }
	auto blockDsoj() { return J.block<6,6>(0,19); }
	auto trnTrnj() { return J.block<3,3>(0,19); }
	auto rotRotj() { return J.block<3,3>(3,22); }
	auto velVelj() { return J.block<3,3>(6,25); }

	Eigen::Matrix<double,28,28> H;
	Eigen::Matrix<double,28,1> b;

	[[clang::optnone]]
	double computeHb(const Mat99 measurementWeights, const Vec6 biasWeights) {
//		std::cerr << J.format(StdOutFmt) << "\n";
//		std::cerr << r.transpose().format(StdOutFmt) << "\n";
		assert(!measurementWeights.hasNaN());
		assert(!biasWeights.hasNaN());
		assert(!J.hasNaN());
		assert(!r.hasNaN());
		assert(!rBias.hasNaN());
//		std::cerr << "J = " << J.format(MatlabFmt) << "\n";
//		std::cerr << "r = " << r.format(MatlabFmt) << "\n";


		// Scale J before computing H and b.
		worldscale() *= SCALE_SCALE;
		worldrot() *= SCALE_ORIENTATION;
		biasA() *= SCALE_BIAS_ACC;
		biasG() *= SCALE_BIAS_GYRO;
		trni() *= SCALE_XI_TRANS;
		roti() *= SCALE_XI_ROT;
		veli() *= SCALE_VELOCITY;
		trnj() *= SCALE_XI_TRANS;
		rotj() *= SCALE_XI_ROT;
		velj() *= SCALE_VELOCITY;

		// TODO. Only compute the upper half..
		// Position, Rotation and Velocity errors vs all params.
		H = J.transpose() * measurementWeights * J;
		b = J.transpose() * measurementWeights * r;

		// Accel and Gyro bias model random walk factors.  Bias errors vs bias params.
		H.block<6,6>(4,4) += biasWeights.asDiagonal();
		b.segment<6>(4) += biasWeights.asDiagonal() * rBias;

		assert(!H.hasNaN());
		assert(!b.hasNaN());

		return (r.transpose() * measurementWeights * r + rBias.transpose() * biasWeights.asDiagonal() * rBias)[0];
		//return (r.transpose() * measurementWeights * r)[0];
	}

	// Indexes in local H and b
	// 0 - Scale
	// 1&2&3 - gravity direction
	// 4-6 - Accel biases
	// 7-9 - Gyro biases
	const int iTrvIdx = IPARS;
	const int jTrvIdx = IPARS + TRVPARS;

	Vec10 bWorldAndBiases() { return b.head<IPARS>(); }
	Vec9 bi() { return b.segment<TRVPARS>(iTrvIdx); }
	Vec9 bj() { return b.segment<TRVPARS>(jTrvIdx); }

	Mat1010 HWorld() { return H.topLeftCorner<IPARS,IPARS>(); }
	Mat99 Hii() { return H.block<TRVPARS,TRVPARS>(iTrvIdx,iTrvIdx); }
	Mat99 Hjj() { return H.block<TRVPARS,TRVPARS>(jTrvIdx,jTrvIdx); }

	// Off diagonal blocks.
	Mat99 Hij() { return H.block<TRVPARS,TRVPARS>(iTrvIdx,jTrvIdx); }
	Mat99 Hji() { return H.block<TRVPARS,TRVPARS>(jTrvIdx,iTrvIdx); }

	Mat109 HWorldi() { return H.block<IPARS,TRVPARS>(0,iTrvIdx); }
	Mat109 HWorldj() { return H.block<IPARS,TRVPARS>(0,jTrvIdx); }

	Mat910 HiWorld() { return H.block<TRVPARS,IPARS>(iTrvIdx,0); }
	Mat910 HjWorld() { return H.block<TRVPARS,IPARS>(jTrvIdx,0); }
};

std::ostream& operator<<(std::ostream& os, const JrWrapper& wrapper) {
	os << "J=...\n" << wrapper.J.format(MatlabFmt);
	os << "r=...\n" << wrapper.r.format(MatlabFmt);
	os << "rBias=...\n" << wrapper.rBias.format(MatlabFmt);
	return os;
}

// WIP
[[clang::optnone]]
double EnergyFunctional::addImuFactors(const int frameIndex, MatXX &H, VecX &b) {
	const int i = frameIndex;
	const int j = frameIndex+1;

	const FrameHessian* const frameI = frames[i]->data;
	const FrameHessian* const frameJ = frames[j]->data;

	if (frames[i]->frameID != frames[j]->frameID -1) {
		std::cerr << "Skipping IMU Factors for frames " << frames[i]->frameID << " and " << frameJ->frameID << "\n";
		return 0;
	}
	JrWrapper Jr;

	const ImuIntegration& imuIJ = frameJ->imuIntegration;
	const double dt = imuIJ.deltaTime;
	//std::cerr << std::setprecision (20) << "IMU Times:" << imuIJ.startTime << " " << dt << " count:" << imuIJ.measurementsCount << "\n";
	//std::cerr << "IMU rotAcc:" << imuIJ.rotAcc.format(StdOutFmt) << "\n";
	//std::cerr << "Velocity I:" << frameI->getVelocityScaled().transpose().format(StdOutFmt) << " J:" << frameJ->getVelocityScaled().transpose().format(StdOutFmt) << "\n";
	//std::cerr << "IMU velAcc:" << imuIJ.velAcc.transpose().format(StdOutFmt) << "\n";
	assert(dt > 0);

	std::cout << "Frames " << frameI->frameID << "->" << frameJ->frameID << "\n";

	const Vec3 deltaBa = HBias.get_value_minus_valueZero().segment<3>(0);;
	const Vec3 deltaBg = HBias.get_value_minus_valueZero().segment<3>(3);

	// Where DSO's origin is relative to frameI's evaluation point.
	const SE3& TCamiDsoEval = frameI->get_worldToCam_evalPT();
	const SE3& TCamjDsoEval = frameJ->get_worldToCam_evalPT();
	// Where frameI is relative to DSO's origin. Current estimate.
	const SE3& TDsoCamiEst = frameI->PRE_TWorldCam;
	const SE3& TDsoCamjEst = frameJ->PRE_TWorldCam;

	// IMU/Base poses in Metric world.
	const SE3 TBaseiEval = dsoCamPoseToMetricImuPose(TCamiDsoEval.inverse());
	const SE3 TBasejEval = dsoCamPoseToMetricImuPose(TCamjDsoEval.inverse());
	const SE3 TBaseiEst = dsoCamPoseToMetricImuPose(TDsoCamiEst);
	const SE3 TBasejEst = dsoCamPoseToMetricImuPose(TDsoCamjEst);

	const Mat33 RBaseiEstTranspose = TBaseiEst.rotationMatrix().transpose();

	// Velocities are expressed in metric frame...
	// Metric/IMU world, X+ is up, Z+ is forwards
	const Vec3 veli = frameI->getVelocityScaled();
	const Vec3 velj = frameJ->getVelocityScaled();

	// DSO's estimates in IMU/Base i's frame.
	const Mat33 expectedRot = RBaseiEstTranspose * TBasejEst.rotationMatrix();
	const Vec3 expectedVel = RBaseiEstTranspose * (velj - veli - Gravity * dt);
	const Vec3 expectedTrn = RBaseiEstTranspose * (TBasejEst.translation() - TBaseiEst.translation() - veli * dt - Gravity*dt*dt/2);

	std::cout << "TDsoCamiEst=...\n" << TDsoCamiEst.matrix().format(MatlabFmt);
	std::cout << "TBaseiEst=...\n" << TBaseiEst.matrix().format(MatlabFmt);
	std::cout << "TDsoCamjEst=...\n" << TDsoCamjEst.matrix().format(MatlabFmt);
	std::cout << "TBasejEst=...\n" << TBasejEst.matrix().format(MatlabFmt);

	std::cout << "veli=...\n" << veli.format(MatlabFmt);
	std::cout << "velj=...\n" << velj.format(MatlabFmt);
	std::cout << "Gravity=...\n" << Gravity.format(MatlabFmt);

	// Compute Rotation, Velocity and Position residual errors for current imu measurement.
	// From Forster, IMU Preintegration (37)
	// The residuals are the DSO estimate vs the IMU measurement...
	const Mat33 deltaRot = imuIJ.deltaRot * SO3::exp(imuIJ.jRotBg * deltaBg).matrix();
	const Vec3 resRot = SO3(deltaRot.transpose() * expectedRot).log();
	assert(!resRot.hasNaN());

	const Vec3 deltaVel = imuIJ.deltaVel + imuIJ.jVelBg * deltaBg + imuIJ.jVelBa * deltaBa;
	const Vec3 resVel = expectedVel - deltaVel;

	const Vec3 deltaTrn = imuIJ.deltaTrn + imuIJ.jTrnBg * deltaBg + imuIJ.jTrnBa * deltaBa;
	const Vec3 resTrn = expectedTrn - deltaTrn;

	// Residuals
	Jr.rTra() = resTrn;
	Jr.rRot() = resRot;
	Jr.rVel() = resVel;

	std::cout << "expectedRot=...\n" << expectedRot.format(MatlabFmt);
	std::cout << "Rimu=...\n" << deltaRot.format(MatlabFmt);

	std::cout << "expectedTrn=...\n" << expectedTrn.format(MatlabFmt);
	std::cout << "deltaTrn=...\n" << deltaTrn.format(MatlabFmt);

	std::cout << "expectedVel=...\n" << expectedVel.format(MatlabFmt);
	std::cout << "deltaVel=...\n" << deltaVel.format(MatlabFmt);

	if (resVel.norm() > 1000000) {
		std::cerr << imuIJ;
		std::cerr << "OH!\n";
	}

	// Gyro and Accel residuals are just deltas from previous values.
	// TODO Do they need something to keep them near 0.
	Jr.rBiasA() = deltaBa;
	Jr.rBiasG() = deltaBg;

	assert(!Jr.J.hasNaN());
	assert(!TBaseiEst.rotationMatrix().hasNaN());
	assert(!TBasejEst.rotationMatrix().hasNaN());

	// Position Jacobians. From Forster, IMU Preintegration supplement section 2.1
	Jr.trnRoti() = SO3::hat(expectedTrn);
	Jr.trnTrni() = -Mat33::Identity();
	Jr.trnVeli() = -RBaseiEstTranspose * dt;
	Jr.trnTrnj() = expectedRot;
	Jr.trnBiasa() = -imuIJ.jTrnBa;
	Jr.trnBiasg() = -imuIJ.jTrnBg;
	assert(!Jr.J.hasNaN());

	// Velocity Jacobians. From Forster, IMU Preintegration supplement section 2.2
	Jr.velRoti() = SO3::hat(expectedVel);
	Jr.velVeli() = -RBaseiEstTranspose;
	Jr.velVelj() = RBaseiEstTranspose;
	Jr.velBiasa() = -imuIJ.jVelBa;
	Jr.velBiasg() = -imuIJ.jVelBg;
	assert(!Jr.J.hasNaN());

	// Rotation Jacobians. From Forster, IMU Preintegration supplement section 2.3
	const Mat33 resRotJri = so3RightJacobianInverse(resRot);
	assert(!resRotJri.hasNaN());

	Jr.rotRoti() =-resRotJri * TBasejEst.rotationMatrix().transpose() * TBaseiEst.rotationMatrix();
	Jr.rotRotj() = resRotJri;
	Jr.rotBiasg() = -resRotJri * SO3::exp(resRot).matrix().transpose() * so3RightJacobian(deltaBg) * imuIJ.jRotBg;
	assert(!Jr.J.hasNaN());

	// Metric/DSO Transform...
	// From Stumberg, dynamic Marginalization supplement.
	Mat44 TMetricCamImuInverse = HWorld.TMetricCamImu.matrix().inverse();
	Mat44 baseDso = TMetricCamImuInverse * HWorld.TMetricDso.matrix();
	Mat77 AdjBaseDso = Sim3(baseDso).Adj();

	Mat77 AdjBaseCami = Sim3(baseDso*TCamiDsoEval.matrix()).Adj();
	Mat77 AdjBaseCamj = Sim3(baseDso*TCamjDsoEval.matrix()).Adj();

	// How the transform between the local IMU measurement frames and the Metric frame change WRT the Metric-DSO transform
	Mat77 J_poseb_wd_i = AdjBaseCami - AdjBaseDso;
	Mat77 J_poseb_wd_j = AdjBaseCamj - AdjBaseDso;
	Mat97 JresWorld = Jr.trnroti() * J_poseb_wd_i.topLeftCorner<6,7>() + Jr.trnrotj() * J_poseb_wd_j.topLeftCorner<6,7>();
//		// Only estimate the rotation and the scale.
	Jr.worldscale() = JresWorld.col(6);
//		Jr.worldrotA() = JresWorld.col(3);
//		Jr.worldrotB() = JresWorld.col(4);
	Jr.worldrot() = JresWorld.block<9,3>(0,3);

//		std::cout << "JresWorld=...\n" << JresWorld.format(MatlabFmt);
//		std::cout << "Jtrnroti=...\n" << Jr.trnroti().format(MatlabFmt);
//		std::cout << "J_poseb_wd_i=...\n" << J_poseb_wd_i.format(MatlabFmt);
//		std::cout << "Jtrnrotj=...\n" << Jr.trnrotj().format(MatlabFmt);
//		std::cout << "J_poseb_wd_j=...\n" << J_poseb_wd_j.format(MatlabFmt);

// TODO: What happens to scaling here...
	Mat66 Jreli = -1 * AdjBaseCami.block(0,0,6,6);
	Mat66 Jrelj = -1 * AdjBaseCamj.block(0,0,6,6);
//
//		std::cout << "AdjBaseCami=...\n" << AdjBaseCami.format(MatlabFmt);

	// Express IMU rot & translation for i & j in DSO's world with right increments.
	Jr.blockDsoi() *= Jreli * TCamiDsoEval.Adj().inverse();
	Jr.blockDsoj() *= Jrelj * TCamjDsoEval.Adj().inverse();

	std::cout << Jr;
	std::cout << "biasCovariance=...\n" << imuIJ.biasCovariance.format(MatlabFmt);

	Mat99 measurementWeights = imuWeightSquared * imuIJ.measurementCovariance.inverse();
	Vec6 biasWeights = imuWeightSquared * imuIJ.biasCovariance.cwiseInverse();
	std::cout << "biasH=...\n" << biasWeights.format(MatlabFmt);
	double energy = Jr.computeHb(measurementWeights, biasWeights);

	// Accumulate IMU factors into DSO's H and b.
	// Indexes in expanded DSO H and b
	int iTrvIdx = ICPARS + IFPARS * i;
	int jTrvIdx = ICPARS + IFPARS * j;

	assert(!H.hasNaN());
	assert(!b.hasNaN());

	b.segment<IPARS>(CPARS) += Jr.bWorldAndBiases();
	b.segment<TRVPARS>(iTrvIdx) += Jr.bi();
	b.segment<TRVPARS>(jTrvIdx) += Jr.bj();

	H.block<IPARS,IPARS>(CPARS,CPARS) += Jr.HWorld();
	H.block<TRVPARS,TRVPARS>(iTrvIdx,iTrvIdx) += Jr.Hii();
	H.block<TRVPARS,TRVPARS>(jTrvIdx,jTrvIdx) += Jr.Hjj();

	// Off diagonal blocks.
	H.block<TRVPARS,TRVPARS>(iTrvIdx,jTrvIdx) += Jr.Hij();
	H.block<TRVPARS,TRVPARS>(jTrvIdx,iTrvIdx) += Jr.Hji();
	H.block<IPARS,TRVPARS>(CPARS,iTrvIdx) += Jr.HWorldi();
	H.block<IPARS,TRVPARS>(CPARS,jTrvIdx) += Jr.HWorldj();
	H.block<TRVPARS,IPARS>(iTrvIdx,CPARS) += Jr.HiWorld();
	H.block<TRVPARS,IPARS>(jTrvIdx,CPARS) += Jr.HjWorld();

	return energy;
}


[[clang::optnone]]
void EnergyFunctional::addPrior(MatXX &H, VecX &b) {
	// Camera, IMU World and IMU Bias priors.
	H.diagonal().head<CPARS>() += cPrior;
	b.head<CPARS>() += cPrior.cwiseProduct(cDelta);

	H.diagonal().segment<IWPARS>(CPARS) += wPrior;
	//b.segment<IWPARS>(CPARS) += wPrior.cwiseProduct(wDelta);

	H.diagonal().segment<IBPARS>(CPARS + IWPARS) += bPrior;
	//b.segment<IBPARS>(CPARS + IWPARS) += bPrior.cwiseProduct(bDelta);

	for(int h=0;h<frames.size();h++) {
		H.diagonal().segment<11>(ICPARS+h*IFPARS) += frames[h]->prior;
		b.segment<11>(ICPARS+h*IFPARS) += frames[h]->prior.cwiseProduct(frames[h]->delta_prior);
	}
}

void EnergyFunctional::makeIDX()
{
	for(unsigned int idx=0;idx<frames.size();idx++)
		frames[idx]->idx = idx;

	allPoints.clear();

	for(EFFrame* f : frames)
		for(EFPoint* p : f->points)
		{
			allPoints.push_back(p);
			for(EFResidual* r : p->residualsAll)
			{
				r->hostIDX = r->host->idx;
				r->targetIDX = r->target->idx;
			}
		}


	EFIndicesValid=true;
}

void EnergyFunctional::getStitchedDeltaF(VecX &delta) {
	if (delta.size() == ICPARS+nFrames*IFPARS) {
		abort();
		delta.head<CPARS>() = cDelta;
		for (int h = 0; h < nFrames; h++)
			delta.segment<IFPARS>(ICPARS + IFPARS * h) = frames[h]->delta;
	} else {
		assert(delta.size() == CPARS+nFrames*FPARS);
		delta.head<CPARS>() = cDelta;
		for(int h=0;h<nFrames;h++)
			delta.segment<FPARS>(CPARS+FPARS*h) = frames[h]->dsoDelta();
	}
	assert(!delta.hasNaN());
}

}
