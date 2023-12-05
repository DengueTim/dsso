//
// Created by tp on 29/11/23.
//

#pragma once

namespace dso {
    struct ImuCalib {
        const double gravity = -9.8082;
        const double integration_sigma = 0.316227;
        const double accel_sigma = 0.316227;
        const double gyro_sigma = 0.1;
    };
};