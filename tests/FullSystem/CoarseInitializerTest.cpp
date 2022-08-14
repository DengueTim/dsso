#include <fstream>
#include <gtest/gtest.h>

#include "util/NumType.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/CoarseInitializer.h"

using namespace dso;

class CoarseInitializerTest: public ::testing::Test {
protected:
	const int imageWidth = 752;
	const int imageHeight = 480;
	CalibHessian *Hcalib;
	FrameHessian *frameHessian;
	CoarseInitializer *coarseInitializer;
	std::vector<IOWrap::Output3DWrapper*> outputWrapper;

	void SetUp() override {
		setGlobalCalib(imageWidth, imageHeight);

		const Mat44 leftExtrinsics =
				(Mat44() << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975, 0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768, -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949, 0.0, 0.0, 0.0, 1.0).finished();
		const Mat44 rightExtrinsics =
				(Mat44() << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556, 0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024, -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038, 0.0, 0.0, 0.0, 1.0).finished();
		// leftToRight is the transform for a point between the cameras, not the camera transform..
		const SE3 leftToRight = SE3(rightExtrinsics).inverse() * SE3(leftExtrinsics);
		const Mat33 leftK = (Mat33() << 458.652, 0, 366.717, 0, 457.296, 247.871, 0, 0, 1).finished();
		const Mat33 rightK = (Mat33() << 457.584, 0, 379.493, 0, 456.13, 254.74, 0, 0, 1).finished();
		Hcalib = new CalibHessian(leftK, rightK, leftToRight);

		// Load image data from dump from commented code in makeImages()
		float *leftImage = new float[imageWidth * imageHeight];
		float *rightImage = new float[imageWidth * imageHeight];
		std::ifstream is;
		is.open("tests/LeftRightDump.bin", std::ios::binary);
		ASSERT_TRUE(is.is_open());
		for (int i = 0; i < imageWidth * imageHeight; i++) {
			is.read((char*) &leftImage[i], sizeof(float));
			is.read((char*) &rightImage[i], sizeof(float));
		}
		is.close();
		frameHessian = new FrameHessian();
		frameHessian->makeImages<true>(leftImage, rightImage, Hcalib);
		delete leftImage;
		delete rightImage;

		coarseInitializer = new CoarseInitializer(imageWidth, imageHeight);
	}

	void TearDown() {
		delete coarseInitializer;
		delete frameHessian;
		delete Hcalib;
	}
};

TEST_F(CoarseInitializerTest, setFirst) {
	coarseInitializer->setFirst(Hcalib, frameHessian, outputWrapper);
}

