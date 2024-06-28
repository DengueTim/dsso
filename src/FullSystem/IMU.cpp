//
// Created by tp on 29/11/23.
//

#include "IMU.h"

namespace dso {

    ImuIntegrator::ImuIntegrator(const dso::ImuCalib &imuCalib) : result() {
        params.reset(new gtsam::PreintegrationParams((gtsam::Vector(3) << 0, 0, imuCalib.gravity).finished()));
        params->setIntegrationCovariance(imuCalib.integration_sigma * imuCalib.integration_sigma * Eigen::Matrix3d::Identity());
        params->setAccelerometerCovariance(imuCalib.accel_sigma * imuCalib.accel_sigma * Eigen::Matrix3d::Identity());
        params->setGyroscopeCovariance(imuCalib.gyro_sigma * imuCalib.gyro_sigma * Eigen::Matrix3d::Identity());
        preintegratedMeasurements.reset(new gtsam::PreintegratedImuMeasurements(params));
    }

    void ImuIntegrator::integrateImuMeasurements(const ImuMeasurements &measurements) {
		preintegratedMeasurements->resetIntegration();
        for (const auto &measurement: measurements) {
//			std::cerr << "IMU Lin:" << measurement.accLin.transpose().format(MatlabFmt) << "\n";
//			std::cerr << "IMU Rot:" << measurement.accRot.transpose().format(MatlabFmt) << "\n";
//			std::cerr << "IMU Int ns:" << measurement.interval << "\n";
//			std::cerr << "IMU Time ns:" << measurement.timestamp << "\n";
			const double intervalSeconds = measurement.interval / 1000000000;
            preintegratedMeasurements->integrateMeasurement(gtsam::Vector(measurement.accLin),
                                                           	gtsam::Vector(measurement.accRot),
															intervalSeconds);
        }

		result = preintegratedMeasurements->preintegrated();
		interval = preintegratedMeasurements->deltaTij();
		// Print results in seconds.
//		const Vec9 resultSeconds = result / interval;
//		std::cerr << "Preintegrated IMU:" << resultSeconds.format(MatlabFmt) << "\n";
    }

	const Vec9& ImuIntegrator::get() {
		return result;
	}

	const Vec9 ImuIntegrator::getPerSecond() {
		return result/interval;
	}
}