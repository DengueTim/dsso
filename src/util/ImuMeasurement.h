//
// Created by tp on 29/11/23.
//

#pragma once

#include <vector>
#include <Eigen/Core>

namespace dso {
    struct ImuMeasurement {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        const Eigen::Vector3d linAcc; // m/s^2
        const Eigen::Vector3d rotRate; // rads/s
        const double timestamp;
        const double interval; // Time since last measurement in nano seconds.

        ImuMeasurement(const Eigen::Vector3d linAcc, const Eigen::Vector3d rotRate, const double timestamp,
                       const double interval) :
			linAcc(linAcc), rotRate(rotRate), timestamp(timestamp), interval(interval) {}

        ImuMeasurement(const double wx, const double wy, const double wz, const double ax, const double ay,
                       const double az, const double timestamp, const double interval) :
			linAcc(ax, ay, az), rotRate(wx, wy, wz), timestamp(timestamp), interval(interval) {}

		const double intervalAsSeconds() const {
			return interval / 1000000000.0; // from nano-seconds.
		}
    };

    typedef std::vector<ImuMeasurement, Eigen::aligned_allocator<ImuMeasurement>> ImuMeasurements;
};