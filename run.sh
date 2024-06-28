#!/bin/bash


./build/bin/dsso_dataset \
	leftFiles=$HOME/datasets/EuRoC/MH_01_easy/cam0/data \
	rightFiles=$HOME/datasets/EuRoC/MH_01_easy/cam1/data \
	leftCalib=$HOME/src/dsso/EuRoC_MAV/cameraLeft.txt \
	rightCalib=$HOME/src/dsso/EuRoC_MAV/cameraRight.txt \
	gamma=$HOME/src/dsso/EuRoC_MAV/pcalib.txt \
	vignette=$HOME/src/dsso/EuRoC_MAV/vignette.png \
	preset=0
	mode=0

