//
// Created by tp on 29/11/23.
//

#pragma once

#include <vector>
#include <Eigen/Core>

namespace dso {
    struct ImuMeasurement {
        const Eigen::Vector3d accLin;
        const Eigen::Vector3d accRot;
        const double timestamp;
        const double interval; // Time since last measurement

        ImuMeasurement(const Eigen::Vector3d accLin, const Eigen::Vector3d accRot, const double timestamp,
                       const double interval) :
                accLin(accLin), accRot(accRot), timestamp(timestamp), interval(interval) {}

        ImuMeasurement(const double wx, const double wy, const double wz, const double ax, const double ay,
                       const double az, const double timestamp, const double interval) :
                accLin(ax, ay, az), accRot(wx, wy, wz), timestamp(timestamp), interval(interval) {}
    };

    typedef std::vector<ImuMeasurement> ImuMeasurements;
};