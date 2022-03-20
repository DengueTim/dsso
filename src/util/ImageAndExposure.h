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
#include <cstring>
#include <iostream>

namespace dso {

class ImageAndExposure {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	float *imageL;			// irradiance. between 0 and 256
	float *imageR;			// Right image.
	int w, h;				// width and height;
	double timestamp;
	float exposure_time;	// exposure time in ms.

	inline ImageAndExposure(int w_, int h_, double timestamp = 0, float exposure_time = 1) :
			w(w_), h(h_), timestamp(timestamp), exposure_time(exposure_time) {
		imageL = new float[w * h];
		imageR = new float[w * h];
	}
	inline ~ImageAndExposure() {
		delete[] imageL;
		delete[] imageR;
	}

	inline ImageAndExposure* getDeepCopy() {
		ImageAndExposure *img = new ImageAndExposure(w, h, timestamp, exposure_time);
		memcpy(img->imageL, imageL, w * h * sizeof(float));
		memcpy(img->imageR, imageR, w * h * sizeof(float));
		return img;
	}
};

}
