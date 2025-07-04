#!/bin/bash


./build/bin/dso_dataset \
	files=/Users/tp/datasets/EuRoC/MH_01_easy/cam0/data \
	calib=/Users/tp/src/dsso/EuRoC_MAV/cameraLeft.txt \
	gamma=/Users/tp/src/dsso/EuRoC_MAV/pcalib.txt \
	vignette=/Users/tp/src/dsso/EuRoC_MAV/vignette.png \
	imuData=/Users/tp/datasets/EuRoC/MH_01_easy/imu0/data.csv \
	preset=0 \
	mode=0
