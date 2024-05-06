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


#include "FullSystem/PixelSelector2.h"
#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"

namespace dso
{

template<class Image_Type> PixelSelector<Image_Type>::PixelSelector(int w, int h)
{
	randomIndexes = new unsigned int[w*h];
	for (unsigned int i=0 ; i < w*h ; i++) {
		randomIndexes[i] = i;
	}
	std::srand(6128);
	std::random_shuffle(&randomIndexes[0], &randomIndexes[w*h - 1]); // want to be deterministic.

	ths = new float[(w/32)*(h/32)+100];
	thsSmoothed = new float[(w/32)*(h/32)+100];

	gradHistFrame=0;
}

template<class Image_Type> PixelSelector<Image_Type>::~PixelSelector()
{
	delete[] randomIndexes;
	delete[] ths;
	delete[] thsSmoothed;
}

int computeHistQuantil(int* hist, int bins, float below)
{
	int th = hist[0]*below+0.5f;
	for(int i=0;i<bins;i++)
	{
		th -= hist[i+1];
		if(th<0) return i;
	}
	return bins-1;
}


template<class Image_Type> void PixelSelector<Image_Type>::makeHists(const Image_Type* const fh)
{
	gradHistFrame = fh;
	float * mapmax0 = fh->absSquaredGrad[0];

	int w = wG[0];
	int h = hG[0];

	int w32 = w/32;
	int h32 = h/32;
	thsStep = w32;

	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			int hist0[50];
			memset(hist0,0,sizeof(int)*50);

			for(int j=0;j<32;j++) for(int i=0;i<32;i++)
			{
				int it = i+32*x;
				int jt = j+32*y;
				if(it>w-2 || jt>h-2 || it<1 || jt<1) continue;
				int g = sqrtf(mapmax0[it+jt*w]);
				if(g>48) g=48;
				hist0[g+1]++;
				hist0[0]++;
			}

			ths[x+y*w32] = computeHistQuantil(hist0, 50, setting_minGradHistCut) + setting_minGradHistAdd;
		}

	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			float sum=0,num=0;
			if(x>0)
			{
				if(y>0) 	{num++; 	sum+=ths[x-1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x-1+(y+1)*w32];}
				num++; sum+=ths[x-1+(y)*w32];
			}

			if(x<w32-1)
			{
				if(y>0) 	{num++; 	sum+=ths[x+1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x+1+(y+1)*w32];}
				num++; sum+=ths[x+1+(y)*w32];
			}

			if(y>0) 	{num++; 	sum+=ths[x+(y-1)*w32];}
			if(y<h32-1) {num++; 	sum+=ths[x+(y+1)*w32];}
			num++; sum+=ths[x+y*w32];

			thsSmoothed[x+y*w32] = (sum/num) * (sum/num);

		}





}
template<class Image_Type> int PixelSelector<Image_Type>::makeMaps(
		const Image_Type* const fh,
		float* map_out, float density, int recursionsLeft, bool plot, float thFactor)
{
	float numWant = density; //Oversample.


	// the number of selected pixels behaves approximately as
	// K / (pot+1)^2, where K is a scene-dependent constant.
	// we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.

	if(fh != gradHistFrame) makeHists(fh);

	// select!
	Eigen::Vector3i n = this->select(fh, map_out, numWant, thFactor);

	// sub-select!
	float numHave = n[0]+n[1]+n[2];

	printf("PixelSelector: want %.2f, have %.2f. n0 %d, n1 %d, n2 %d\n", numWant, numHave, n[0], n[1], n[2]);

	if(plot)
	{
		int w = wG[0];
		int h = hG[0];


		MinimalImageB3 img(w,h);

		for(int i=0;i<w*h;i++)
		{
			float c = fh->dI[i][0]*0.7;
			if(c>255) c=255;
			img.at(i) = Vec3b(c,c,c);
		}
		IOWrap::displayImage("Selector Image", &img);

		for(int y=0; y<h;y++)
			for(int x=0;x<w;x++)
			{
				int i=x+y*w;
				if(map_out[i] == 1)
					img.setPixelCirc(x,y,Vec3b(0,255,0));
				else if(map_out[i] == 2)
					img.setPixelCirc(x,y,Vec3b(128,0,0));
				else if(map_out[i] == 4)
					img.setPixelCirc(x,y,Vec3b(0,0,255));
			}
		IOWrap::displayImage("Selector Pixels", &img);
		IOWrap::waitKey(0);
	}

	return numHave;
}



template<class Image_Type> Eigen::Vector3i PixelSelector<Image_Type>::select(const Image_Type* const fh,
																			 float* map_out, int pot, float thFactor)
{
	float * mapmax0 = fh->absSquaredGrad[0];
	float * mapmax1 = fh->absSquaredGrad[1];
	float * mapmax2 = fh->absSquaredGrad[2];

	int w = wG[0];
	int w1 = wG[1];
	int w2 = wG[2];
	int h = hG[0];

	memset(map_out,0,w*h*sizeof(PixelSelectorStatus));

	int n = 0, n0=0, n1=0, n2=0;
	for (int i=0 ; i < w*h ;i++) {
		if (pot < n)
			break;

		unsigned int idx = randomIndexes[i];
		int x = idx % w;
		int y = idx / w;
		if (x<4 || x>(w-4) || y<4 || y>(h-4)) continue;

		float ag0 = mapmax0[idx];
		float pixelTH0 = thsSmoothed[(x >> 5) + (y >> 5) * thsStep];
		if (ag0 > pixelTH0) {
			map_out[idx] = 1;
			n0++;
			n++;
			continue;
		}

		if (i % 4 != 0) continue;

		float ag1 = mapmax1[(int)(x * 0.5f + 0.25f) + (int)(y * 0.5f + 0.25f) * w1];
		float pixelTH1 = pixelTH0;
		if (ag1 > pixelTH1) {
			map_out[idx] = 2;
			n1++;
			n++;
			continue;
		}

		if (i % 16 != 0) continue;

		float ag2 = mapmax2[(int)(x * 0.25f + 0.125) + (int)(y * 0.25f + 0.125) * w2];
		float pixelTH2 = pixelTH0;
		if (ag2 > pixelTH2) {
			map_out[idx] = 4;
			n2++;
			n++;
		}
	}

	return Eigen::Vector3i(n0,n1,n2);
}

}

