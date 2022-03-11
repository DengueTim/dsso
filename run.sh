#!/bin/bash


./build/Debug/bin/dso_dataset \
	files=~/Downloads/sequence_01/images.zip \
	calib=~/Downloads/sequence_01/camera.txt \
	gamma=~/Downloads/sequence_01/pcalib.txt \
	vignette=~/Downloads/sequence_01/vignette.png \
	preset=0 \
	mode=0

