// alg2.cpp : Defines the entry point for the console application.
//

#include <conio.h>
#include <algorithm>
#include <fstream>
#include "bmp/EasyBMP.h"
#include "../filtration_cuda_lib/filtrationCudaLibrary.h"
#include "../image_lib/imageLibrary.h"


int main(int argc, char* argv[])
{
	printf("Filtration\n");

	printf("Usage: filtrationCuda.exe <input.bmp> <result.bmp> <method> <scale>\n");

	
	char* featureFileName = argv[1];
	char* outputFileName = argv[2];
	char* filtrationMethod = argv[3];
	char* scale = argv[4];

	float scaleF = atoi(scale);
	scaleF /= 100.0;

	printf("source image: %s\n", featureFileName);
	int imwidth, imheight;
	unsigned char* image = loadImage(featureFileName, imwidth, imheight);

	printf("===\nInitializing CUDA kernel...\n");

	unsigned char* cudaResult = NULL;
	
	int method = 0;

	if( strstr(filtrationMethod, "sobel") )
		method = 0;
	else if( strstr(filtrationMethod, "puritt") )
		method = 1;
	else if( strstr(filtrationMethod, "laplas") )
		method = 2;

	cudaResult = filtrationCudaL(image, imwidth, imheight, method, scaleF);

	saveImage(outputFileName, cudaResult, imwidth, imheight);
	return 0;
}
