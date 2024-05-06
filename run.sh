#!/bin/bash


./build/bin/dso_dataset \
 	files=$HOME/datasets/EuRoC/MH_01_easy/cam0/data \
        calib=$HOME/src/dsso/EuRoC_MAV/cameraLeft.txt \
        gamma=$HOME/src/dsso/EuRoC_MAV/pcalib.txt \
        vignette=$HOME/src/dsso/EuRoC_MAV/vignette.png \
	preset=0 \
	mode=0 \
	nomt=1

