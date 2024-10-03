//
// Created by tp on 29/11/23.
//

#include "IMU.h"

namespace dso {

    ImuIntegrator::ImuIntegrator(const dso::ImuCalib &imuCalib) {
        params.reset(new gtsam::PreintegrationParams((gtsam::Vector(3) << 0, 0, Gravity[3]).finished()));
        params->setIntegrationCovariance(imuCalib.integration_sigma * imuCalib.integration_sigma * Eigen::Matrix3d::Identity());
        params->setAccelerometerCovariance(imuCalib.accel_sigma * imuCalib.accel_sigma * Eigen::Matrix3d::Identity());
        params->setGyroscopeCovariance(imuCalib.gyro_sigma * imuCalib.gyro_sigma * Eigen::Matrix3d::Identity());
        preintegratedMeasurements.reset(new gtsam::PreintegratedImuMeasurements(params));
    }

    void ImuIntegrator::integrateImuMeasurements(const ImuMeasurements &measurements) {
        for (const auto &measurement: measurements) {
            preintegratedMeasurements->integrateMeasurement(gtsam::Vector(measurement.accLin),
                                                           gtsam::Vector(measurement.accRot),
                                                           measurement.interval);
        }
        printf("Ms: %zu\n", measurements.size());
    }
}