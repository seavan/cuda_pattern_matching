
#include "filtrationCudaLibrary.h"


extern "C" unsigned char* filtrationCuda( int argc, char** argv, 
								   unsigned char* _input, int _width, int _height, int _method, float _scale);


unsigned char* filtrationCudaL( unsigned char* _input, int _width, int _height, int _method, float _scale)
{
	return filtrationCuda(0, 0, _input, _width, _height, _method, _scale);
}

