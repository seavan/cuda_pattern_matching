#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../image_lib/imageLibrary.h"
#include "../filtration_lib/filtrationLibrary.h"

int main(int argc, char* argv[])
{
	int width, height;
	int methodIndex = 0;
	unsigned char* input = loadImage(argv[1], width, height);
	

	float coeff = 0.25;

	if( argc > 3 )
	{	
		const char* method = argv[3];
		const char* scale = argv[4];
		float scaleF = atoi(scale);
		scaleF /= 100.0;
		coeff = scaleF;

		if( strstr(method, "puritt") != NULL )
		{
			methodIndex = 1;
		}
		else
			if( strstr(method, "laplas") != NULL )
		{
			methodIndex = 2;
		}
		else
		{
			methodIndex = 0;
		}
	}

	unsigned char* result = filtration(input, width, height, methodIndex, coeff);
	saveImage(argv[2], result, width, height);

	return 0;
}

