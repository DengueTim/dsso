#!/bin/bash


./build/Debug/bin/dsso_dataset \
	leftFiles=$HOME/src/dsso/EuRoC_MAV/sequence/cam0/data \
	rightFiles=$HOME/src/dsso/EuRoC_MAV/sequence/cam1/data \
	leftCalib=$HOME/src/dsso/EuRoC_MAV/cameraLeft.txt \
	rightCalib=$HOME/src/dsso/EuRoC_MAV/cameraRight.txt \
	gamma=$HOME/src/dsso/EuRoC_MAV/pcalib.txt \
	vignette=$HOME/src/dsso/EuRoC_MAV/vignette.png \
	preset=0
	mode=0

