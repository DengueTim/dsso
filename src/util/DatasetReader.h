/**
 * This file is part of DSO.
 * 
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/globalCalib.h"

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

#include "util/Undistort.h"
#include "IOWrapper/ImageRW.h"

#if HAS_ZIPLIB
	#include "zip.h"
#endif

#include <boost/thread.hpp>

using namespace dso;

inline int getdir(std::string dir, std::vector<std::string> &files) {
	DIR *dp;
	struct dirent *dirp;
	if ((dp = opendir(dir.c_str())) == NULL) {
		return -1;
	}

	while ((dirp = readdir(dp)) != NULL) {
		std::string name = std::string(dirp->d_name);

		if (name != "." && name != "..")
			files.push_back(name);
	}
	closedir(dp);

	std::sort(files.begin(), files.end());

	if (dir.at(dir.length() - 1) != '/')
		dir = dir + "/";
	for (unsigned int i = 0; i < files.size(); i++) {
		if (files[i].at(0) != '/')
			files[i] = dir + files[i];
	}

	return files.size();
}

struct PrepImageItem {
	int id;
	bool isQueud;
	ImageAndExposure *pt;

	inline PrepImageItem(int _id) {
		id = _id;
		isQueud = false;
		pt = 0;
	}

	inline void release() {
		if (pt != 0)
			delete pt;
		pt = 0;
	}
};

class ImageFolderReader {
public:
	ImageFolderReader(std::string pathL, std::string pathR, std::string calibFileL, std::string calibFileR, std::string gammaFile,
			std::string vignetteFile) {
		this->pathL = pathL;
		this->pathR = pathR;

#if HAS_ZIPLIB
		ziparchive=0;
		ziparchiveR=0;
		databuffer=0;
#endif

		isZipped = (pathL.length() > 4 && pathL.substr(pathL.length() - 4) == ".zip");

		if (isZipped) {
#if HAS_ZIPLIB
			int ziperror = 0;
			ziparchive = zip_open(pathL.c_str(), ZIP_RDONLY, &ziperror);
			if (ziperror != 0) {
				printf("ERROR %d reading archive %s!\n", ziperror, pathL.c_str());
				exit(1);
			}

			ziparchiveR = zip_open(pathR.c_str(), ZIP_RDONLY, &ziperror);
			if (ziperror != 0) {
				printf("ERROR %d reading archive %s!\n", ziperror, pathR.c_str());
				exit(1);
			}

			files.clear();
			filesR.clear();
			int numEntries = zip_get_num_entries(ziparchive, 0);
			for (int k = 0; k < numEntries; k++) {
				const char *name = zip_get_name(ziparchive, k, ZIP_FL_ENC_STRICT);
				std::string nstr = std::string(name);
				if (nstr == "." || nstr == "..")
					continue;
				files.push_back(name);

				const char *nameR = zip_get_name(ziparchiveR, k, ZIP_FL_ENC_STRICT);
				std::string nstrR = std::string(nameR);
				if (nstrR == "." || nstrR == "..")
					continue;
				filesR.push_back(nameR);
			}

			printf("got %d entries and %d files!\n", numEntries, (int) files.size());
			std::sort(files.begin(), files.end());
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		} else {
			getdir(pathL, files);
			getdir(pathR, filesR);
		}

		undistortL = Undistort::getUndistorterForFile(calibFileL, gammaFile, vignetteFile, false);
		undistortR = Undistort::getUndistorterForFile(calibFileR, gammaFile, vignetteFile, true);

		widthOrg = undistortL->getOriginalSize()[0];
		heightOrg = undistortL->getOriginalSize()[1];
		width = undistortL->getSize()[0];
		height = undistortL->getSize()[1];

		// load timestamps if possible.
		loadTimestamps();
		printf("ImageFolderReader: got %d file pairs in %s & %s!\n", (int) files.size(), pathL.c_str(), pathR.c_str());

	}
	~ImageFolderReader() {
#if HAS_ZIPLIB
		if(ziparchive!=0) {
			zip_close(ziparchive);
			zip_close(ziparchiveR);
		}
		if(databuffer!=0) delete databuffer;
#endif

		delete undistortL;
		delete undistortR;
	}
	;

	Eigen::VectorXf getOriginalCalib() {
		return undistortL->getOriginalParameter().cast<float>();
	}
	Eigen::Vector2i getOriginalDimensions() {
		return undistortL->getOriginalSize();
	}

	void setGlobalCalibration() {
		int w = undistortL->getSize()[0];
		int h = undistortL->getSize()[1];
		setGlobalCalib(w, h);
	}

	int getNumImages() {
		return files.size();
	}

	double getTimestamp(int id) {
		if (timestamps.size() == 0)
			return id * 0.1f;
		if (id >= (int) timestamps.size())
			return 0;
		if (id < 0)
			return 0;
		return timestamps[id];
	}

	void prepImage(int id, bool as8U = false) {

	}

	// By default allocates new ImageAndExposure else reuses what's passed in.
	ImageAndExposure* getImage(int id, ImageAndExposure *iae = 0) {
		MinimalImageB *minimgL = getImageRaw_internal(id, false);
		MinimalImageB *minimgR = getImageRaw_internal(id, true);

        if (iae == 0) {
            iae = new ImageAndExposure(0, 0);
        }

		iae->timestamp = (timestamps.size() == 0 ? 0.0 : timestamps[id]);
		iae->exposure_time = (!setting_useExposure || exposures.size() == 0) ? 1.0f : exposures[id];

		//ImageAndExposure *ret2 = undistortL->undistort<unsigned char>(minimg, minimgR);
		float factor = 1.0;
		undistortL->undistort(minimgL, iae, factor);
		undistortR->undistort(minimgR, iae, factor);

		delete minimgL;
		delete minimgR;
		return iae;
	}

	inline float* getPhotometricGamma() {
		if (undistortL == 0 || undistortL->photometricUndist == 0)
			return 0;
		return undistortL->photometricUndist->getG();
	}

	// undistorter. [0] always exists, [1-2] only when MT is enabled.
	Undistort *undistortR;
	Undistort *undistortL;
private:

	MinimalImageB* getImageRaw_internal(int id, bool rightNotLeft) {
		if (!isZipped) {
			// CHANGE FOR ZIP FILE
			return IOWrap::readImageBW_8U(rightNotLeft ? filesR[id] : files[id]);
		} else {
#if HAS_ZIPLIB
			zip_t* zip = rightNotLeft ? ziparchiveR : ziparchive;
			std::vector<std::string> &files_ = rightNotLeft ? filesR : files;

			if(databuffer==0) databuffer = new char[widthOrg*heightOrg*6+10000];
			zip_file_t* fle = zip_fopen(zip, files_[id].c_str(), 0);
			long readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*6+10000);

			if(readbytes > (long)widthOrg*heightOrg*6)
			{
				printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes,(long)widthOrg*heightOrg*6+10000, files[id].c_str());
				delete[] databuffer;
				databuffer = new char[(long)widthOrg*heightOrg*30];
				fle = zip_fopen(zip, files_[id].c_str(), 0);
				readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*30+10000);

				if(readbytes > (long)widthOrg*heightOrg*30)
				{
					printf("buffer still to small (read %ld/%ld). abort.\n", readbytes,(long)widthOrg*heightOrg*30+10000);
					exit(1);
				}
			}

			return IOWrap::readStreamBW_8U(databuffer, readbytes);
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		}
	}

	inline void loadTimestamps() {
		std::ifstream tr;
		std::string timesFile = pathL.substr(0, pathL.find_last_of('/')) + "/times.txt";
		tr.open(timesFile.c_str());
		while (!tr.eof() && tr.good()) {
			std::string line;
			char buf[1000];
			tr.getline(buf, 1000);

			int id;
			double stamp;
			float exposure = 0;

			if (3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure)) {
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}

			else if (2 == sscanf(buf, "%d %lf", &id, &stamp)) {
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}
		}
		tr.close();

		// check if exposures are correct, (possibly skip)
		bool exposuresGood = ((int) exposures.size() == (int) getNumImages());
		for (int i = 0; i < (int) exposures.size(); i++) {
			if (exposures[i] == 0) {
				// fix!
				float sum = 0, num = 0;
				if (i > 0 && exposures[i - 1] > 0) {
					sum += exposures[i - 1];
					num++;
				}
				if (i + 1 < (int) exposures.size() && exposures[i + 1] > 0) {
					sum += exposures[i + 1];
					num++;
				}

				if (num > 0)
					exposures[i] = sum / num;
			}

			if (exposures[i] == 0)
				exposuresGood = false;
		}

		if ((int) getNumImages() != (int) timestamps.size()) {
			printf("set timestamps and exposures to zero!\n");
			exposures.clear();
			timestamps.clear();
		}

		if ((int) getNumImages() != (int) exposures.size() || !exposuresGood) {
			printf("set EXPOSURES to zero!\n");
			exposures.clear();
		}

		printf("got %d images and %d timestamps and %d exposures.!\n", (int) getNumImages(), (int) timestamps.size(),
				(int) exposures.size());
	}

	std::vector<ImageAndExposure*> preloadedImages;
	std::vector<std::string> files;
	std::vector<std::string> filesR;
	std::vector<double> timestamps;
	std::vector<float> exposures;

	int width, height;
	int widthOrg, heightOrg;

	std::string pathL;
	std::string pathR;

	bool isZipped;

#if HAS_ZIPLIB
	zip_t* ziparchive;
	zip_t* ziparchiveR;
	char* databuffer;
#endif
};

