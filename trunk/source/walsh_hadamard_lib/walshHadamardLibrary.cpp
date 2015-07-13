#include "walshHadamardLibrary.h"

#include <stdlib.h>
#include <algorithm>
#include "pm/whimport.h"


WHSetup* doProcess(unsigned char* _image, int _width, int _height, unsigned char* _pattern, int _patternWidth, int _patternHeight, int corrpercent)
{
	unsigned dim = std::min(_patternWidth, _patternHeight);

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

	unsigned char* patternNormalized = (unsigned char*)malloc(dim * dim * sizeof(unsigned char));

	unsigned char* c = patternNormalized;

	for(int y = 0; y < dim; ++y)
	{
		for(int x = 0; x < dim; ++x)
		{
			*c = _pattern[x + y * _patternWidth];
			++c;
		}
	}
	Image* image = createImage(_image, _height, _width);
	Image* pattern = createImage(patternNormalized, dim, dim);

	WHSetup* setup = createWHSetup(image->rows, image->cols, pattern->rows, 50);
	setSourceImage(setup, image);
	setPatternImage(setup, pattern);

	int distance = pattern->rows * pattern->rows * (100 - corrpercent) / 100;

	whPatternMatch(setup, corrpercent);
	printf("Количество совпадений: %d\n", numOfMatches(setup));
	return setup;
}