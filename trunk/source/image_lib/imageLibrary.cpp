#include "imageLibrary.h"

#include <stdlib.h>
#include "FreeImage.h"

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
	for(int y = 1; y <= dimY; ++y)
	{
		for(int x = 0; x < dimX; ++x)
		{
			RGBQUAD pixel;
			bool ok = FreeImage_GetPixelColor(dib, x, dimY - y, &pixel);
			result[index] = method( &pixel );
			++index;
		}
	}
	_width = dimX;
	_height = dimY;
	FreeImage_Unload(dib);
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
