// alg2.cpp : Defines the entry point for the console application.
//

#include <conio.h>
#include <algorithm>
#include <fstream>
#include "simple/simple.h"
#include "../image_lib/imageLibrary.h"

extern "C" unsigned char* simpleMatchingCuda( int argc, char** argv, 
								   unsigned char* _input, int _width, int _height, 
				unsigned char* _pattern, int _ptnWidth, int _ptnHeight);

int main(int argc, char* argv[])
{
	printf("Simple pattern matching\n");
	if(argc < 4)
	{
		printf("Usage: alg2.exe <input.bmp> <pattern.bmp> <result.txt> <parameter>\n");
		return 1;
	}
	
	char* featureFileName = argv[1];
	char* patternFileName = argv[2];
	char* outputResult = argv[3];
	int corrpercent = atoi(argv[4]);
	printf("Corrpercent: %d\n", corrpercent);
	unsigned char corrlimit = corrpercent * 255.0 / 100;
	printf("Corrlimit: %d\n", corrlimit);
	const char* strMethod = argv[5];

	printf("source image: %s\n", featureFileName);
	int imwidth, imheight;
	unsigned char* image = loadImage(featureFileName, imwidth, imheight);

	printf("pattern image: %s\n", patternFileName);

	int ptnWidth, ptnHeight;
	unsigned char* pattern = loadImage(patternFileName, ptnWidth, ptnHeight);

	printf("corrpercent: %d\n", corrpercent);
	printf("writing results to: %s\n", outputResult);

	FILE* f = fopen(outputResult, "wt");

	printf("===\nInitializing CUDA kernel...\n");

	unsigned char* cudaResult = NULL;
	
	if( strstr(strMethod, "cuda") != NULL )
	{
		cudaResult = simpleMatchingCPU(1, argv, image, imwidth, imheight, pattern, ptnWidth, ptnHeight);
	}
	else
	{
		cudaResult = simpleMatchingCPU(1, argv, image, imwidth, imheight, pattern, ptnWidth, ptnHeight);
	}

	for(int y = 0; y < imheight; ++y)
		for(int x = 0; x < imwidth; ++x)
		{
			if( (*cudaResult > corrlimit ) && (y < imheight - ptnHeight) && (x < imwidth - ptnWidth) )
			{
				//printf("%d %d %d\r\n", x, y, *cudaResult);
				fprintf(f, "%d %d %d %d\n", x, y, *cudaResult, *image);
			}
			++cudaResult;
			++image;

		}

	fclose(f);

	return 0;
}
