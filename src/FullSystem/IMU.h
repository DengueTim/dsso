//
// Created by tp on 29/11/23.
//

#pragma once

#include "util/NumType.h"
#include "util/ImuCalib.h"
#include "util/ImuMeasurement.h"
#include <gtsam/navigation/ImuFactor.h>

namespace dso {
    /**
     * Integrates IMU measurements, as in GTSAM's "IMU Preintegration".
     */
    class ImuIntegrator {
        boost::shared_ptr<gtsam::PreintegrationParams> params;
        boost::shared_ptr<gtsam::PreintegratedImuMeasurements> preintegratedMeasurements;
		Vec9 result;
		double interval;

	public:
		ImuIntegrator(const dso::ImuCalib &imuCalib);

        void integrateImuMeasurements(const ImuMeasurements &measurements);

		const Vec9& get();
		const Vec9 getPerSecond();
	};
};