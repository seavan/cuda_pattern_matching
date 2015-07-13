// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

extern __shared__ int sdata[];

#define PATNDATA(index) sdata[index]
#define PATNDATAY(index) sdata[index + 27]
#define KERNELDATAX(index) PATNDATA(index + 9)
#define KERNELDATAY(index) PATNDATA(index + 18)

#define SHAREDSIZE 300

#define uchar unsigned char

#define THREAD_SIZE 9


#define filterName filterSobel
#define K_UL   1
#define K_UM   0
#define K_UR  -1
#define K_ML   2
#define K_MM   0
#define K_MR   -2
#define K_BL   1
#define K_BM   0
#define K_BR   -1


#include "filterTypes.cu"

#define filterName filterPuritt
#define K_UL   1
#define K_UM   0
#define K_UR  -1
#define K_ML   1
#define K_MM   0
#define K_MR   -1
#define K_BL   1
#define K_BM   0
#define K_BR   -1
#include "filterTypes.cu"

#define K_UL   0
#define K_UM   1
#define K_UR   0
#define K_ML   1
#define K_MM   -4
#define K_MR   1
#define K_BL   0 
#define K_BM   1
#define K_BR   0
#define filterName filterLaplas
#include "filterTypes.cu"


extern "C" unsigned char* filtrationCuda( int argc, char** argv, 
								   unsigned char* _input, int _width, int _height, int _method, float _scale)
{
	CUT_DEVICE_INIT(argc, argv);
    unsigned int timer = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    // allocate device memory
	int imsize = _width * _height;

	uchar* fImage = new uchar[imsize];
	uchar* fResult = new uchar[imsize];

	memcpy( fImage, _input, imsize );

	uchar* d_idata = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, imsize * sizeof(unsigned char) ));

	// copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_idata, fImage, imsize * sizeof(unsigned char),
                                cudaMemcpyHostToDevice) );
    
	// allocate device memory for result
    uchar* d_odata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, imsize));

    // setup execution parameters
    dim3  grid( _width, _height, 1);
    dim3  threads( THREAD_SIZE, 1, 1);

    // execute the kernel
	switch( _method )
	{
	case 0: filterSobel<<< grid, THREAD_SIZE, SHAREDSIZE >>>( d_idata, d_odata, _width, _height, _scale); break;
	case 1: filterPuritt<<< grid, THREAD_SIZE, SHAREDSIZE >>>( d_idata, d_odata, _width, _height, _scale); break;
	case 2: filterLaplas<<< grid, THREAD_SIZE, SHAREDSIZE >>>( d_idata, d_odata, _width, _height, _scale); break;
	}
    

    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( fImage, d_odata, imsize,
                                cudaMemcpyDeviceToHost) );

	CUT_CHECK_ERROR("Output copy failed");

    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    CUT_SAFE_CALL( cutDeleteTimer( timer));
	
	return fImage;
}

