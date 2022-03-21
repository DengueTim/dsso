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

#include "util/globalCalib.h"
#include "stdio.h"
#include <iostream>

namespace dso {
int wG[PYR_LEVELS], hG[PYR_LEVELS];

float wM3G;
float hM3G;

void setGlobalCalib(int w, int h) {
	int wlvl = w;
	int hlvl = h;
	pyrLevelsUsed = 1;
	while (wlvl % 2 == 0 && hlvl % 2 == 0 && wlvl * hlvl > 5000 && pyrLevelsUsed < PYR_LEVELS) {
		wlvl /= 2;
		hlvl /= 2;
		pyrLevelsUsed++;
	}
	printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n", pyrLevelsUsed - 1, wlvl, hlvl);
	if (wlvl > 100 && hlvl > 100) {
		printf("\n\n===============WARNING!===================\n "
				"using not enough pyramid levels.\n"
				"Consider scaling to a resolution that is a multiple of a power of 2.\n");
	}
	if (pyrLevelsUsed < 3) {
		printf("\n\n===============WARNING!===================\n "
				"I need higher resolution.\n"
				"I will probably segfault.\n");
	}

	wM3G = w - 3;
	hM3G = h - 3;

	wG[0] = w;
	hG[0] = h;

	for (int level = 1; level < pyrLevelsUsed; ++level) {
		wG[level] = w >> level;
		hG[level] = h >> level;

	}
}

}
