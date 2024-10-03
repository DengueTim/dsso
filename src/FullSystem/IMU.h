//
// Created by tp on 29/11/23.
//

#pragma once

#include "util/NumType.h"
#include "util/ImuMeasurement.h"
#include <gtsam/navigation/ImuFactor.h>

namespace dso {
	const Vec3 Gravity(0,0,-9.8082);

	struct ImuCalib {
		double integration_sigma = 0.316227;
		double accel_sigma = 0.316227;
		double gyro_sigma = 0.1;
	};

	struct ImuIntegration {
		// Preintegrated i to j deltas.
		Mat33 deltaRot;
		Vec3 deltaVel;
		Vec3 deltaPos;

		// Bias correction. From Forster, IMU Preintegration MAP supplement. A.20
		Mat33 jRotBg;
		Mat33 jVelBa;
		Mat33 jVelBg;
		Mat33 jPosBa;
		Mat33 jPosBg;

		Mat99 covariance;
	};

    /**
     * Integrates IMU measurements, as in GTSAM's "IMU Preintegration".
     */
    class ImuIntegrator {
        boost::shared_ptr<gtsam::PreintegrationParams> params;
        boost::shared_ptr<gtsam::PreintegratedImuMeasurements> preintegratedMeasurements;

    public:
        ImuIntegrator(const dso::ImuCalib &imuCalib);

        void integrateImuMeasurements(const ImuMeasurements &measurements);


		void getCurrentIntegration(ImuIntegration& integration);
	};

};