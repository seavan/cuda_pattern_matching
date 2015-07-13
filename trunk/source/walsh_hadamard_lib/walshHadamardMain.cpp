// alg2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <conio.h>
#include <algorithm>
#include <fstream>
#include "bmp/EasyBMP.h"
#include "FreeImage.h"

extern "C" unsigned char* correlationCuda( int argc, char** argv, 
								   unsigned char* _input, int _width, int _height, unsigned char* _pattern, int _ptnwidth, int _ptnheight);

extern "C" unsigned char* correlationCpu( unsigned char* _input, int _width, int _height, unsigned char* _pattern, int _ptnwidth, int _ptnheight);

typedef unsigned char (*TMethod) (RGBQUAD*);

unsigned char grayScale(RGBQUAD* _pixel)
{
	return (_pixel->rgbRed + _pixel->rgbGreen + _pixel->rgbBlue) / 3;
}


TMethod method = grayScale;

unsigned char* loadImage(const char* _fileName, int& _width, int& _height)
{
	FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(_fileName);

	FIBITMAP *dib = FreeImage_Load(fif, _fileName, 0);
	
	int dimX = FreeImage_GetWidth(dib);
	int dimY = FreeImage_GetHeight(dib);

	unsigned char* result = (unsigned char*)malloc(sizeof(unsigned char) * dimX * dimY);

	int index = 0;
	for(unsigned y = 1; y <= dimY; ++y)
	{
		for(unsigned x = 0; x < dimX; ++x)
		{
			RGBQUAD pixel;
			bool ok = FreeImage_GetPixelColor(dib, x, dimY - y, &pixel);
			result[index] = method( &pixel );
			++index;
		}
	}
	_width = dimX;
	_height = dimY;
	return result;
}

void saveImage(const char* _fileName, unsigned char* _image, int _width, int _height)
{
	FIBITMAP* resultImage = FreeImage_Allocate(_width, _height, 24);

	for(int y = 1; y <= _height; ++y)
		for(int x = 0; x < _width; ++x)
		{
			int color = *_image;
			color = (color << 8) | (color << 16) | (color) | (color << 24);
			FreeImage_SetPixelColor(resultImage, x, _height - y, (RGBQUAD*)(&color));
			++_image;
		}

	FreeImage_Save(FIF_BMP, resultImage, _fileName);
}

int _tmain(int argc, _TCHAR* argv[])
{
	printf("Filtration\n");

	printf("Usage: correlationCuda.exe <input.bmp> <pattern.bmp> <output> <coeff> <method>\n");

	
	_TCHAR* featureFileName = argv[1];
	_TCHAR* patternFileName = argv[2];
	_TCHAR* coeffStr = argv[4];
	_TCHAR* outputFileName = argv[3];
	_TCHAR* methodStr = argv[5];

	method = grayScale;
	int coeff = (100 - atoi(coeffStr)) * 255 / 100;

	printf("source image: %s\n", featureFileName);
	int imwidth, imheight;
	unsigned char* image = loadImage(featureFileName, imwidth, imheight);
	printf("width: %d, height: %d\n", imwidth, imheight);

	printf("pattern image: %s\n", patternFileName);
	int ptnwidth, ptnheight;
	unsigned char* pattern = loadImage(patternFileName, ptnwidth, ptnheight);
	printf("width: %d, height: %d\n", ptnwidth, ptnheight);

	unsigned char* cudaResult = NULL;

	if( strstr(methodStr, "cuda") != NULL )
	{
		printf("===\nMethod - CUDA\n===\nInitializing CUDA kernel...\n");
		cudaResult = correlationCuda(1, argv, image, imwidth, imheight, pattern, ptnwidth, ptnheight);
	}
	else
	{
		printf("===\nMethod - CPU\n===\nCPU emulation...\n");
		cudaResult = correlationCpu(image, imwidth, imheight, pattern, ptnwidth, ptnheight);
	}

	unsigned char* cudaOriginal = cudaResult;

	printf("Processing finished\nWriting output data\n");
	//saveImage(outputFileName, cudaResult, imwidth, imheight);

	FILE* file = fopen(outputFileName, "wt");

	for(int y = 0; y < imheight; ++y)
		for(int x = 0; x < imwidth; ++x)
		{
			unsigned char value = *cudaResult;

			if( value < coeff )
				fprintf(file, "%d %d %d\n", x, y, value);
			++cudaResult;
		}

	fclose(file);
	free(cudaOriginal);
	free(image);
	free(pattern);

	return 0;
}
