#include <conio.h>
#include <algorithm>
#include <fstream>

#include "../correlation_cuda_lib/correlationCudaLibrary.h"
#include "../image_lib/imageLibrary.h"

int main(int argc, char* argv[])
{
	printf("Usage: correlationCuda.exe <input.bmp> <pattern.bmp> <output> <coeff> <method>\n");

	
	const char* featureFileName = argv[1];
	const char* patternFileName = argv[2];
	const char* coeffStr = argv[4];
	const char* outputFileName = argv[3];
	const char* methodStr = argv[5];

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
		cudaResult = correlationCudaL(image, imwidth, imheight, pattern, ptnwidth, ptnheight);
	}
	else
	{
		printf("===\nMethod - CPU\n===\nCPU emulation...\n");
		cudaResult = correlationCpuL(image, imwidth, imheight, pattern, ptnwidth, ptnheight);
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

	printf("Writing results");
	fclose(file);

	return 0;
}
