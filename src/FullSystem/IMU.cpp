//
// Created by tp on 29/11/23.
//

#include "IMU.h"
//#include <gtsam/inference/Symbol.h>

namespace dso {
	FosterImuIntegrator::FosterImuIntegrator(const dso::ImuCalib& imuCalib) : ImuIntegrator(imuCalib){
		// Assuming a sample rate of 200hz.
		accelCovariance = 200 * imuCalib.accel_noise_density  * imuCalib.accel_noise_density * Mat33::Identity();
		gyroCovariance = 200 * imuCalib.gyro_noise_density * imuCalib.gyro_noise_density * Mat33::Identity();
		integrationCovariance = imuCalib.integration_covariance * Mat33::Identity();

		reset(Vec6::Zero());
	}

	[[clang::optnone]]
	void FosterImuIntegrator::integrateImuMeasurements(const std::shared_ptr<ImuMeasurements> measurements) {
		if (startTime == 0 && !measurements->empty()) {
			startTime = measurements->at(0).timestamp / 1e9; // Nano to seconds.
		}
		for (const auto &measurement : *measurements) {
			double dt = measurement.intervalAsSeconds();
			double dt2 = dt * dt;

			Vec3 linAcc = measurement.linAcc - accelBias;
			Vec3 rotRate = measurement.rotRate - gyroBias;
			Mat33 linAccHat = SO3::hat(linAcc);
			Vec3 w = (rotRate) * dt;

			{ // From VI-Stereo_DSO

				SO3 dR = Sophus::SO3d::exp(w).matrix();// Expmap();
				Mat33 Jr;
				{ //Mat33 Jr = JacobianR(w);
					double theta = w.norm();
					if(theta<0.00001) {
						Jr = Mat33::Identity();
					} else {
						Vec3 k = w.normalized();  // k - unit direction vector of w
						Mat33 K = Sophus::SO3d::hat(k);
//						 Jr =   Mat33::Identity()
//								 - (1-cos(theta))/theta*K
//								 + (1-sin(theta)/theta)*K*K;
						Jr = sin(theta)/theta*Mat33::Identity()+(1-sin(theta)/theta)*k*k.transpose()-(1-cos(theta))/theta*K;
					}
				}


				// noise covariance propagation of delta measurements
				// err_k+1 = A*err_k + B*err_gyro + C*err_acc
				Mat33 I3x3 = Mat33::Identity();
				Mat99 A = Mat99::Identity();
				A.block<3, 3>(3, 3) = dR.matrix().transpose();
				A.block<3, 3>(6, 3) = -rotAcc.matrix() * linAccHat * dt;
				A.block<3, 3>(0, 3) = -0.5 * rotAcc.matrix() * linAccHat * dt2;
				A.block<3, 3>(0, 6) = I3x3 * dt;
				Mat93 Bg = Mat93::Zero();
				Bg.block<3, 3>(3, 0) = Jr * dt;
				Mat93 Ca = Mat93::Zero();
				Ca.block<3, 3>(6, 0) = rotAcc.matrix() * dt;
				Ca.block<3, 3>(0, 0) = 0.5 * rotAcc.matrix() * dt2;
				covarianceAcc = A * covarianceAcc * A.transpose() +
					Bg * gyroCovariance * Bg.transpose() +
					Ca * accelCovariance * Ca.transpose();

				covarianceAcc.block<3, 3>(0, 0).noalias() += integrationCovariance * dt;


				// jacobian of delta measurements w.r.t bias of gyro/acc
				// update P first, then V, then R
				jTrnBa += jVelBa * dt - 0.5 * rotAcc.matrix() * dt2;
				jTrnBg += jVelBg * dt - 0.5 * rotAcc.matrix() * linAccHat * jRotBg * dt2;
				jVelBa += -rotAcc.matrix() * dt;
				jVelBg += -rotAcc.matrix() * linAccHat * jRotBg * dt;
				jRotBg = dR.matrix().transpose() * jRotBg - Jr * dt;

				jBiasA = A * jBiasA - Ca;
				jBiasG = A * jBiasG - Bg;

				assert(jTrnBa.isApprox(jBiasA.block<3,3>(0,0)));
				assert(jTrnBg.isApprox(jBiasG.block<3,3>(0,0)));
				assert(jVelBa.isApprox(jBiasA.block<3,3>(6,0)));
				assert(jVelBg.isApprox(jBiasG.block<3,3>(6,0)));
				assert(jRotBg.isApprox(jBiasG.block<3,3>(3,0)));
				assert((jBiasA.block<3,3>(3,0)).isZero());


				// delta measurements, position/velocity/rotation(matrix)
				// update P first, then V, then R. because P's update need V&R's previous state
				trnAcc += velAcc * dt + 0.5 * rotAcc.matrix() * linAcc * dt2;    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
				velAcc += rotAcc.matrix() * linAcc * dt;
				rotAcc = (rotAcc * dR).matrix();  // normalize rotation, in case of numerical error accumulation

				// delta time
				deltaTime += dt;
			}

//			SO3 dR = SO3::exp(w);
//			Mat33 rightJacobianOfR = Mat33::Identity();Å
//			double wNorm = w.norm();
//			if (wNorm >= 10e-8) { // Epsilon from gtsam Rot3
//				Vec3 wUnit = w / wNorm;
//				Mat33 K = SO3::hat(wUnit);
//				rightJacobianOfR *= sin(wNorm) / wNorm;
//				rightJacobianOfR += (1 - sin(wNorm) / wNorm) * wUnit * wUnit.transpose();
//				rightJacobianOfR -= (1 - cos(wNorm)) / wNorm * K;
//			}
//			{
//				// Code from GTSAM
//				Vec3 x = w;
//				double normx = x.norm();//norm_2(x); // rotation angle
//				Mat33 Jr;
//				if (normx < 10e-8){
//					Jr = Mat33::Identity();
//				}
//				else{
//					const Mat33 X = SO3::hat(x); // element of Lie algebra so(3): X = x^
//					Jr = Mat33::Identity() - ((1-cos(normx))/(normx*normx)) * X +
//						((normx-sin(normx))/(normx*normx*normx)) * X * X; // right Jacobian
//				}
//				assert(Jr.isApprox(rightJacobianOfR));
//			}

			// Translation rotation velocity Propagation.
//			const Mat33 I3 = Mat33::Identity();
//			const Mat33 propRR = dR.matrix().transpose();
//			const Mat33 propVR = -rotAcc.matrix() * linAccHat * dt;
//			const Mat33 propPR = -rotAcc.matrix() * linAccHat * dt2 * 0.5;
//			const Mat33 propPV = Mat33::Identity() * dt; // dVij bit that gets added to dPij in paper..
//
//			const Mat33 propRG = rightJacobianOfR * dt;
//			const Mat33 propVA = rotAcc.matrix() * dt;
//			const Mat33 propPA = 0.5 * rotAcc.matrix() * dt2;

//			covarianceAcc = C * covarianceAcc * C.transpose()
//				+ A * accelCovariance * A.transpose()
//				+ G * gyroCovariance * G.transpose();
//			covarianceAcc.block<3,3>(6,6) += integrationCovariance * dt;

//			jTrnBa += jVelBa * dt + dR.matrix().transpose();
//			jTrnBg += jVelBg * dt - rotAcc.matrix() * linAccHat * jRotBg * dt2 * 0.5;
//			jVelBa -= rotAcc.matrix() * dt;
//			jVelBg -= rotAcc.matrix() * linAccHat * jRotBg * dt;
//			jRotBg = dR.matrix().transpose() * jRotBg - rightJacobianOfR * dt;
//
//			trnAcc += velAcc * dt + rotAcc * linAcc * dt2 * 0.5;
//			velAcc += rotAcc * linAcc * dt;
//			rotAcc *= dR;
//
//			deltaTime += dt;

			intervalAccumulator += measurement.interval;
			linAccAcc += measurement.linAcc;
			rotRateAcc += measurement.rotRate;
			measurementsCounter++;

			if (rotAcc.matrix().hasNaN()) {
				std::cerr << (long)measurement.timestamp << ":" <<
						  "tmp.integrateMeasurement(Vec3(" <<
						  measurement.linAcc.transpose().format(StdOutFmt) << "),Vec3(" <<
						  measurement.rotRate.transpose().format(StdOutFmt) << ")," <<
						  (long)measurement.interval << ");\n";
				abort();
			}
		}

		imuMeasurementsPerFrame.push_back(measurements);

		assert(!velAcc.hasNaN());
		assert(!trnAcc.hasNaN());
	}

[[clang::optnone]]
	void FosterImuIntegrator::getCurrentIntegration(ImuIntegration& integration) {
		integration.startTime = startTime;
		double dt = intervalAccumulator / 1e9; // Nano to seconds.
		integration.deltaTime = deltaTime;
		integration.deltaRot = rotAcc.matrix();
		integration.deltaVel = velAcc;
		integration.deltaTrn = trnAcc;

		integration.jRotBg = jRotBg;
		integration.jVelBa = jVelBa;
		integration.jVelBg = jVelBg;
		integration.jTrnBa = jTrnBa;
		integration.jTrnBg = jTrnBg;

		integration.measurementCovariance = covarianceAcc;
		assert(!integration.measurementCovariance.hasNaN());

		integration.biasCovariance.segment<3>(0) = Vec3::Ones() * accel_random_walk * accel_random_walk * dt;
		integration.biasCovariance.segment<3>(3) = Vec3::Ones() * gyro_random_walk * gyro_random_walk * dt;

		integration.linAccAcc = linAccAcc;
		integration.rotRateAcc = rotRateAcc;
		integration.measurementsCount = measurementsCounter;

		integration.imuMeasurementsPerFrame = imuMeasurementsPerFrame;
	}

	const Mat33 FosterImuIntegrator::computeGravityInitialiser() {
		Eigen::Quaterniond q;
		q.setFromTwoVectors(linAccAcc, -dso::Gravity);
//		std::cerr << "r:" << r.format(MatlabFmt) << "\nq:" << q.toRotationMatrix().format(MatlabFmt) << "\n";
		return q.toRotationMatrix();
	}

	void FosterImuIntegrator::reset(const Vec6& biases) {
		accelBias = biases.head<3>();
		gyroBias = biases.tail<3>();

		deltaTime = 0;

		rotAcc = SO3();
		velAcc.setZero();
		trnAcc.setZero();

		jRotBg.setZero();
		jVelBa.setZero();
		jVelBg.setZero();
		jTrnBa.setZero();
		jTrnBg.setZero();

		jBiasA.setZero();
		jBiasG.setZero();

		covarianceAcc.setZero();

		intervalAccumulator = 0.0;
		linAccAcc.setZero();
		rotRateAcc.setZero();
		measurementsCounter = 0;

		imuMeasurementsPerFrame.clear();
	}



	// GTSAM...
[[clang::optnone]]
	GtsamImuIntegrator::GtsamImuIntegrator(const dso::ImuCalib& imuCalib) : ImuIntegrator(imuCalib){
		params = boost::make_shared<gtsam::PreintegrationParams>(Gravity);
        params->setIntegrationCovariance(imuCalib.integration_covariance * Eigen::Matrix3d::Identity());
		// GTSAM takes the continuous-time noise covariance.
        params->setAccelerometerCovariance(imuCalib.accel_noise_density * imuCalib.accel_noise_density * Eigen::Matrix3d::Identity());
        params->setGyroscopeCovariance(imuCalib.gyro_noise_density * imuCalib.gyro_noise_density * Eigen::Matrix3d::Identity());
		reset(Vec6::Zero());
    }

	void GtsamImuIntegrator::printDebug() {
//		std::cerr << "timestamp:" << (long)debugMeasurements.at(0).timestamp << " m:" << debugMeasurements.size() << "\n";
//		for (const auto &m: debugMeasurements) {
//			std::cerr << // (long)measurement.timestamp << ":" <<
//					  "tmp.integrateMeasurement(Vec3(" <<
//					  m.accLin.transpose().format(StdOutFmt)  << "),Vec3(" <<
//					  m.rotAcc.transpose().format(StdOutFmt)  << ")," <<
//					  (long)m.interval << ");\n";
//		}
		std::cerr << "accLinAcc=" << linAccAcc.format(MatlabFmt) << "\n";
		std::cerr << "rotRateAcc=" << rotRateAcc.format(MatlabFmt) << "\n";
		std::cerr << "measurementsCounter=" << measurementsCounter << "\n";
		std::cerr << "rotAcc=" << preintegratedMeasurements->deltaRij().matrix().format(MatlabFmt) << "\n";
		std::cerr << "velAcc=" << preintegratedMeasurements->deltaVij().format(MatlabFmt) << "\n";
		std::cerr << "trnAcc=" << preintegratedMeasurements->deltaPij().format(MatlabFmt) << "\n";
		std::cerr << "preintMeasCov=" << preintegratedMeasurements->preintMeasCov().format(MatlabFmt) << "\n";
	}

[[clang::optnone]]
    void GtsamImuIntegrator::integrateImuMeasurements(const std::shared_ptr<ImuMeasurements> measurements) {
		if (startTime == 0 && !measurements->empty()) {
			startTime = measurements->at(0).timestamp / 1e9; // Nano to seconds.
		}
		for (const auto &measurement : *measurements) {
			preintegratedMeasurements
				->integrateMeasurement(measurement.linAcc, measurement.rotRate, measurement.intervalAsSeconds());
			intervalAccumulator += measurement.interval;

			if (preintegratedMeasurements->deltaRij().matrix().hasNaN() || preintegratedMeasurements->preintMeasCov().hasNaN()) {
				printDebug();
				std::cerr << (long)measurement.timestamp << ":" <<
						  "tmp.integrateMeasurement(Vec3(" <<
						  measurement.linAcc.transpose().format(StdOutFmt) << "),Vec3(" <<
						  measurement.rotRate.transpose().format(StdOutFmt) << ")," <<
						  (long)measurement.interval << ");\n";
				abort();
			}

			linAccAcc += measurement.linAcc;
			rotRateAcc += measurement.rotRate;
			measurementsCounter++;
		}

		imuMeasurementsPerFrame.push_back(measurements);

		assert(!preintegratedMeasurements->deltaVij().hasNaN());
		assert(!preintegratedMeasurements->deltaPij().hasNaN());
	}

[[clang::optnone]]
	void GtsamImuIntegrator::getCurrentIntegration(ImuIntegration& integration) {
//		gtsam::Key poseCurrentKey = gtsam::Symbol('p', 0);
//		gtsam::Key velCurrentKey = gtsam::Symbol('v', 0);
//		gtsam::Key biasCurrentKey = gtsam::Symbol('b', 0);
//
//		gtsam::Key posePrevKey = gtsam::Symbol('p', 0);
//		gtsam::Key velPrevKey = gtsam::Symbol('v', 0);
//		gtsam::Key biasPrevKey = gtsam::Symbol('b', 0);
//
//		gtsam::ImuFactor f = gtsam::ImuFactor(posePrevKey, velPrevKey,
//						 poseCurrentKey, velCurrentKey, biasPrevKey,
//												  *preintegratedMeasurements);


		integration.startTime = startTime;
		double dt = intervalAccumulator / 1e9; // Nano to seconds.
		integration.deltaTime = dt;
		integration.deltaRot = preintegratedMeasurements->deltaRij().matrix();
		integration.deltaVel = preintegratedMeasurements->deltaVij();
		integration.deltaTrn = preintegratedMeasurements->deltaPij();

		// Order is: theta, position, velocity.
		//Vec9 p = preintegratedMeasurements->preintegrated();
		const gtsam::Matrix93 &biasAcc = preintegratedMeasurements->preintegrated_H_biasAcc();
		assert(!biasAcc.hasNaN());
		const gtsam::Matrix93 &biasOmega = preintegratedMeasurements->preintegrated_H_biasOmega();
		assert(!biasOmega.hasNaN());
		integration.jRotBg = biasOmega.block<3,3>(0,0);
		integration.jVelBa = biasAcc.block<3,3>(6,0);
		integration.jVelBg = biasOmega.block<3,3>(6,0);
		integration.jTrnBa = biasAcc.block<3,3>(3,0);
		integration.jTrnBg = biasOmega.block<3,3>(3,0);

		// GTSAM order is rotation, position, velocity.
		const Mat99 &measCov = preintegratedMeasurements->preintMeasCov();
		// We need position, rotation, velocity.
		// Swap rotation and position.
		integration.measurementCovariance.block<3,3> (0,0) = measCov.block<3,3>(3,3);
		integration.measurementCovariance.block<3,3> (3,0) = measCov.block<3,3>(0,3);
		integration.measurementCovariance.block<3,3> (6,0) = measCov.block<3,3>(6,3);
		integration.measurementCovariance.block<3,3> (0,3) = measCov.block<3,3>(3,0);
		integration.measurementCovariance.block<3,3> (3,3) = measCov.block<3,3>(0,0);
		integration.measurementCovariance.block<3,3> (6,3) = measCov.block<3,3>(6,0);
		integration.measurementCovariance.block<3,3> (0,6) = measCov.block<3,3>(3,6);
		integration.measurementCovariance.block<3,3> (3,6) = measCov.block<3,3>(0,6);
		integration.measurementCovariance.block<3,3> (6,6) = measCov.block<3,3>(6,6);
		assert(!integration.measurementCovariance.hasNaN());

//		assert(f.preintegratedMeasurements().preintMeasCov().isApprox(measCov));

		integration.biasCovariance.segment<3>(0) = Vec3::Ones() * accel_random_walk * accel_random_walk * dt;
		integration.biasCovariance.segment<3>(3) = Vec3::Ones() * gyro_random_walk * gyro_random_walk * dt;
		assert(!integration.biasCovariance.hasNaN());

		integration.linAccAcc = linAccAcc;
		integration.rotRateAcc = rotRateAcc;
		integration.measurementsCount = measurementsCounter;

		integration.imuMeasurementsPerFrame = imuMeasurementsPerFrame;
	}

	const Mat33 GtsamImuIntegrator::computeGravityInitialiser() {
		//const Vec3 unitLinAccel = TBaseCam.rotationMatrix().inverse() * linAccAcc/linAccAcc.norm();
		const Vec3 unitLinAccel = linAccAcc/linAccAcc.norm();
		const Vec3 unitNegativeG = dso::Gravity / -dso::Gravity.norm();
		Vec3 axis = Sophus::SO3d::hat(unitLinAccel)*unitNegativeG; // Cross product. Axis, length = sin(theta)
		double sin_theta = axis.norm();
		axis = axis / sin_theta;
		double cos_theta = unitLinAccel.dot(unitNegativeG); // Dot product of unit vectors -> cos(theta);

		Mat33 r = cos_theta*Mat33::Identity()+(1-cos_theta)*axis*axis.transpose()+sin_theta*Sophus::SO3d::hat(axis);

		Eigen::Quaterniond q;
		q.setFromTwoVectors(linAccAcc, -dso::Gravity);
		//std::cerr << "r:" << r.format(MatlabFmt) << "\nq:" << q.toRotationMatrix().format(MatlabFmt) << "\n";
		return q.toRotationMatrix();
	}

	// Biases. Acc then Gyro.
	void GtsamImuIntegrator::reset(const Vec6& biases) {
		//preintegratedMeasurements->resetIntegrationAndSetBias(gtsam::imuBias::ConstantBias());
        // preintegratedMeasurements.reset(new gtsam::PreintegratedImuMeasurements(params));
//        if (preintegratedMeasurements != NULL) {
//            delete preintegratedMeasurements;
//        }
//		if (measurementsCounter > 1000) {
//			printDebug();
//		}

		preintegratedMeasurements.reset(new gtsam::PreintegratedImuMeasurements(params, gtsam::imuBias::ConstantBias(biases.head<3>(), biases.tail<3>())));
        intervalAccumulator = 0.0;
		linAccAcc.setZero();
		rotRateAcc.setZero();
		measurementsCounter = 0;
		startTime = 0;
		imuMeasurementsPerFrame.clear();
	}

	std::ostream& operator<<(std::ostream& os, const ImuIntegration& ii) {
		os << "deltaTime=" << ii.deltaTime << ";\nmeasurementsCount=" << ii.measurementsCount << ";\n";
		//os << ii.deltaRot.transpose().format(StdOutFmt);
		os << " linAccAcc=" << (ii.linAccAcc / ii.measurementsCount).transpose().format(MatlabFmt);
		os << "  deltaTrn=" << (ii.deltaTrn / ii.deltaTime).transpose().format(MatlabFmt);
		os << "  deltaVel=" << ii.deltaVel.transpose().format(MatlabFmt);

		os << "rotRateAcc=" << (ii.rotRateAcc / ii.measurementsCount).transpose().format(MatlabFmt);
		os << "deltaRot=\n" << (ii.deltaRot).format(MatlabFmt);
		os << "measurementCovariance=\n" << ii.measurementCovariance.transpose().format(MatlabFmt);

		os << "jRotBg=\n" << (ii.jRotBg).format(MatlabFmt);
		os << "jTrnBa=\n" << (ii.jTrnBa).format(MatlabFmt);
		os << "jTrnBg=\n" << (ii.jTrnBg).format(MatlabFmt);
		os << "jTrnBg=\n" << (ii.jTrnBg).format(MatlabFmt);
		os << "jVelBg=\n" << (ii.jVelBg).format(MatlabFmt);
		os << "biasCovariance=" << ii.biasCovariance.transpose().format(MatlabFmt);

		return os;
	}
}