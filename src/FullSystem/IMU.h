//
// Created by tp on 29/11/23.
//

#pragma once

#include "util/NumType.h"
#include "util/ImuMeasurement.h"
#include <gtsam/navigation/ImuFactor.h>

namespace dso {
	// Gravity in Metrix frame. Along positive Y-Axis. Same orientation as camera.
	const Vec3 Gravity(0,9.8082,0);

	struct ImuCalib {
		// Error committed in integrating position from velocities. Value from GTSAM's ImuFactorsExample.cpp. A bit of a mystery..
		double integration_covariance = 1e-8;
		// Values from EuRoC sensor.yaml...
		// The units for sigma/stddev are σ = rad/s/√Hz
		double gyro_noise_density = 0.00016968;
		// The units for sigma/stddev are σ = m/s²/√Hz
		double accel_noise_density = 0.002;
		// TODO: Use these values in EnergyFunctional IMU Bias calc.
		double gyro_random_walk = 0.000019393;
		double accel_random_walk = 0.003;
	};

	struct ImuIntegration {
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		double startTime = 0; // seconds.
		double deltaTime = 0; // seconds.

		// Preintegrated i to j deltas.
		Vec3 deltaTrn;
		Mat33 deltaRot;
		Vec3 deltaVel;

		// Bias correction. From Forster, IMU Preintegration MAP supplement. A.20
		Mat33 jTrnBa;
		Mat33 jTrnBg;
		Mat33 jRotBg;
		Mat33 jVelBa;
		Mat33 jVelBg;

		// Only the diagonal elements
		Vec3 covTrn;
		Vec3 covRot;
		Vec3 covVel;
		Vec3 covBa;
		Vec3 covBg;

		// Covariances for integration interval.
		Mat99 measurementCovariance; // Order translation, rotation, velocity.
		Vec6 biasCovariance; // Order acceleration, gyration.

		// Accumulations over measurementsCount measurements, not averages.
		Vec3 linAccAcc;
		Vec3 rotRateAcc;
		int measurementsCount;

		std::vector<std::shared_ptr<ImuMeasurements>> imuMeasurementsPerFrame;
	};
	std::ostream& operator<<(std::ostream& os, const ImuIntegration& ii);

    /**
     * Integrates IMU measurements.
     */
    class ImuIntegrator {
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	protected:
		const double accel_random_walk;
		const double gyro_random_walk;
		double intervalAccumulator;
		Vec3 linAccAcc;
		Vec3 rotRateAcc;
		int measurementsCounter;
		double startTime = 0;

		std::vector<std::shared_ptr<ImuMeasurements>> imuMeasurementsPerFrame;

		ImuIntegrator(const dso::ImuCalib& imuCalib) :
			accel_random_walk(imuCalib.accel_random_walk), gyro_random_walk(imuCalib.gyro_random_walk),
			intervalAccumulator(0.0), measurementsCounter(0) {}
    public:

		virtual void integrateImuMeasurements(const std::shared_ptr<ImuMeasurements> measurements) = 0;

		virtual void getCurrentIntegration(ImuIntegration& integration) = 0;

		virtual const Mat33 computeGravityInitialiser() = 0;

		virtual void reset(const Vec6& biases = Vec6::Zero()) = 0;

		virtual ~ImuIntegrator() {};
	};

	/**
	 * Integrates IMU measurements, see in Forster's IMU Preintegration Supplement.
	 */
	class FosterImuIntegrator : public ImuIntegrator {
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		Vec3 accelBias;
		Vec3 gyroBias;

		double deltaTime = 0; // seconds.

		// IMU noise covariance.
		Mat33 accelCovariance;
		Mat33 gyroCovariance;
		Mat33 integrationCovariance;

		// Preintegrated i to j deltas accumulators.
		Vec3 trnAcc;
		SO3 rotAcc;
		Vec3 velAcc;

		Vec3 covTrnAcc;
		Vec3 covRotAcc;
		Vec3 covVelAcc;

		// Bias correction. From Forster, IMU Preintegration MAP supplement. A.20
		Mat33 jTrnBa;
		Mat33 jTrnBg;
		Mat33 jRotBg;
		Mat33 jVelBa;
		Mat33 jVelBg;

		Mat93 jBiasA;
		Mat93 jBiasG;

		Mat99 covarianceAcc;

	public:
		FosterImuIntegrator(const dso::ImuCalib& imuCalib);

		void integrateImuMeasurements(const std::shared_ptr<ImuMeasurements> measurements) override;

		void getCurrentIntegration(ImuIntegration& integration) override;

		const Mat33 computeGravityInitialiser() override;

		void reset(const Vec6& biases) override;
	};

	/**
	 * Integrates IMU measurements, as in GTSAM's IMU Preintegration.
	 */
	class GtsamImuIntegrator : public ImuIntegrator {
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		std::shared_ptr<gtsam::PreintegrationParams> params;
        std::shared_ptr<gtsam::PreintegratedImuMeasurements> preintegratedMeasurements;

		void printDebug();

	public:
		GtsamImuIntegrator(const dso::ImuCalib& imuCalib);

		void integrateImuMeasurements(const std::shared_ptr<ImuMeasurements> measurements) override;

		void getCurrentIntegration(ImuIntegration& integration) override;

		const Mat33 computeGravityInitialiser() override;

		void reset(const Vec6& accBias) override;
	};
};