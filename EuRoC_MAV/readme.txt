The camera files are derived from the sensor.yaml files that come with the dataset.
The pcalib and vignette files are the result of running online_photometric_calibration on the dataset.

Run it something like:

./bin/dso_dataset \
	leftFiles=/data/datasets/EuRoC/MH_01_easy/cam0/data \
	rightFiles=/data/datasets/EuRoC/MH_01_easy/cam1/data \
	leftCalib=$HOME/src/dsso/EuRoC_MAV/cameraLeft.txt \
	rightCalib=$HOME/src/dsso/EuRoC_MAV/cameraRight.txt \
	gamma=$HOME/src/dsso/EuRoC_MAV/pcalib.txt \
	vignette=$HOME/src/dsso/EuRoC_MAV/vignette.png \
	preset=0
	mode=0