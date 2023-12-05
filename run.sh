#!/bin/bash


./build/Debug/bin/dso_dataset \
	files=$HOME/src/dso/EuRoC_MAV/sequence/cam0/data \
	calib=$HOME/src/dso/EuRoC_MAV/cameraLeft.txt \
	gamma=$HOME/src/dso/EuRoC_MAV/pcalib.txt \
	vignette=$HOME/src/dso/EuRoC_MAV/vignette.png \
	preset=0
	mode=0

