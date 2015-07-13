// alg2.cpp : Defines the entry point for the console application.
//

#include <conio.h>
#include <algorithm>
#include <fstream>
#include "bmp/EasyBMP.h"
#include "pm/whimport.h"

extern "C" Image *createImage2(pixelT *pixels, coordT rows, coordT cols);
Image* loadImage(cppBMP* _bitmap);
Image* loadPatternImage(cppBMP* _bitmap);

typedef pixelT (*TMethod) (RGBApixel);

pixelT grayScale(RGBApixel _pixel)
{
	return (_pixel.Red + _pixel.Green + _pixel.Blue) / 3;
}

pixelT colorScaleB(RGBApixel _pixel)
{
	return _pixel.Blue;
}

pixelT colorScaleG(RGBApixel _pixel)
{
	return _pixel.Green;
}

pixelT colorScaleR(RGBApixel _pixel)
{
	return _pixel.Red;
}

TMethod method = colorScaleB;

WHSetup* doProcess(const char* featureFileName, const char* patternFileName, int corrpercent);

int main(int argc, char* argv[])
{
	printf("Поиск совпадению по методу Walsh-Hadamart\n");
	if(argc < 6)
	{
		printf("Неправильное число аргументов.\n\t Использование: alg2.exe <input.bmp> <pattern.bmp> <result.txt> <corrpercent> <method>\n");
		return 1;
	}
	
	char* featureFileName = argv[1];
	char* patternFileName = argv[2];
	char* outputResult = argv[3];
	int corrpercent = atoi(argv[4]);
	char* strMethod = argv[5];
	
	WHSetup* setup = NULL;
	Match m;

	corrpercent = 200;

	if( strstr(strMethod, "rgb") == NULL )
	{
		method = colorScaleB;
		setup = doProcess(featureFileName, patternFileName, corrpercent);

	}
	else
	{

		method = colorScaleR;
		setup = doProcess(featureFileName, patternFileName, corrpercent);

	}


	printf("Вывод результатов: %s\n", outputResult);
	
	FILE* f = fopen(outputResult, "wt");
	for( unsigned i = 0; i < numOfMatches(setup); ++i )
	{
		if(matches(setup)[i].x != -1)
			fprintf(f, "%d %d %d\r\n", matches(setup)[i].x, matches(setup)[i].y, matches(setup)[i].distance);
	}
	fclose(f);

	return 0;
}

WHSetup* doProcess(const char* featureFileName, const char* patternFileName, int corrpercent)
{
	printf("Загрузка исходного файла: %s\n", featureFileName);
	cppBMP srcBmp;
	srcBmp.ReadFromFile(featureFileName);
	Image* image = loadImage(&srcBmp);

	printf("Загрузка файла шаблона: %s\n", patternFileName);

	cppBMP ptnBmp;
	ptnBmp.ReadFromFile(patternFileName);
	Image* pattern = loadPatternImage(&ptnBmp);

	// Инициализация алгоритма
	WHSetup* setup = createWHSetup(image->rows, image->cols, pattern->rows, 50);
	setSourceImage(setup, image);
	setPatternImage(setup, pattern);

	int distance = pattern->rows * pattern->rows * (100 - corrpercent) / 100;

	whPatternMatch(setup, corrpercent);
	printf("Количество совпадений: %d\n", numOfMatches(setup));
	return setup;
}

Image* loadImage(cppBMP* _bitmap)
{
	unsigned dim = std::max(_bitmap->TellWidth(), _bitmap->TellHeight());
	unsigned dimX = _bitmap->TellWidth();
	unsigned dimY = _bitmap->TellHeight();

	pixelT* result = new pixelT[dimX * dimY];

	int index = 0;
	for(unsigned y = 0; y < dimY; ++y)
	{
		for(unsigned x = 0; x < dimX; ++x)
		{
			result[index] = method(_bitmap->GetPixel(x, y));
			++index;
		}
	}
	return createImage(result, dimY, dimX);
}


Image* loadPatternImage(cppBMP* _bitmap)
{
	unsigned dim = std::min(_bitmap->TellWidth(), _bitmap->TellHeight());

	if( dim > 128 )
	{
		dim = 128;
	}
	else
	if( dim > 64 )
	{
		dim = 64;
	}
	else
	if( dim > 32 )
	{
		dim = 32;
	}


	pixelT* result = new pixelT[dim * dim];

	int index = 0;
	for(unsigned y = 0; y < dim; ++y)
	{
		for(unsigned x = 0; x < dim; ++x)
		{
			result[index] = method(_bitmap->GetPixel(x, y));
			//printf(result[index] > 100 ? "*":" ");
			++index;
		
		}
		//printf("\n");
	}
	return createImage(result, dim, dim);
}

