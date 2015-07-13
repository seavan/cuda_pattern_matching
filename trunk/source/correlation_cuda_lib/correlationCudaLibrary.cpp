
#include "correlationCudaLibrary.h"



extern "C" unsigned char* correlationCpu( unsigned char* _input, int _width, int _height, unsigned char* _pattern, int _ptnwidth, int _ptnheight);
extern "C" unsigned char* correlationCuda( int argc, char** argv, 
								   unsigned char* _input, int _width, int _height, unsigned char* _pattern, int _ptnwidth, int _ptnheight);


unsigned char* correlationCpuL( unsigned char* _input, int _width, int _height, unsigned char* _pattern, int _ptnwidth, int _ptnheight)
{
	return correlationCpu(_input, _width, _height, _pattern, _ptnwidth, _ptnheight);
}

unsigned char* correlationCudaL( unsigned char* _input, int _width, int _height, unsigned char* _pattern, int _ptnwidth, int _ptnheight)
{
	return correlationCuda(0, 0, _input, _width, _height, _pattern, _ptnwidth, _ptnheight);
}

