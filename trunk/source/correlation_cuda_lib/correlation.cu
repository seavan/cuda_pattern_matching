// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <dos.h>
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


__global__ void
correlation( uchar* g_idata, uchar* g_pdata, uchar* g_odata, int _width, int _height, int _ptnWidth, int _ptnHeight, int i, int j) 
{
	// определить смещение обрабатываемого элемента
	int x = (blockIdx.x) * blockDim.x + threadIdx.x + i;
	int y = (blockIdx.y) * blockDim.y + threadIdx.y + j;

	// проверить границы
	if( (x >= _width) || (y >= _height) )
	{
		return;
	}

	// определить смещение дл€ записи результата
	uchar* curResult = g_odata + y * _width + x;

	// проверить границы
	if( (x > _width - _ptnWidth) || (y > _height - _ptnHeight) )
	{
		*curResult = 0xFF;
		return;
	}
	
	// задать начальный указатель дл€ шаблона
	uchar* curPattern = g_pdata;

	// результат просчета коррел€ции
	float result = 0;

	// сумма квадратов значений дл€ изображени€ и шаблона
	float nom1 = 0;
	float nom2 = 0;

	// просчитать коррел€цию
	for(int i = 0; i < _ptnHeight; ++i)
	{
		uchar* curRow = g_idata + x + (y + i) * _width;
		for(int j = 0; j < _ptnWidth; ++j)
		{	
			// текущие пиксели изображени€ и шаблона
			uchar cv = *curRow;
			uchar cp = *curPattern;	
			
			// разность
			int delta = cv - cp;

			// евклидово рассто€ние
			result += delta * delta;
			
			// энерги€ шаблона и изображени€
			nom1 += cv * cv;
			nom2 += cp * cp;

			// перейти дальше
			++curRow;
			++curPattern;
		}
	}

	// просчитать дополнение коэффициента коррел€ции (1 - коэфф)
	result /= sqrt(nom1 * nom2);

	// нормализовать по базе 255 * 4
	result *= 1024;

	// отсечь запредельные значени€
	result = result > 255 ? 255 : result;

	// вернуть результат
	*curResult = (uchar)(result);
}

void
correlationCpu( uchar* g_idata, uchar* g_pdata, uchar* g_odata, int _width, int _height, int _ptnWidth, int _ptnHeight, int i, int j) 
{
	int x = i;//(blockIdx.x) * blockDim.x + threadIdx.x + i;
	int y = j;//(blockIdx.y) * blockDim.y + threadIdx.y + j;

	if( (x >= _width) || (y >= _height) )
	{
		return;
	}

	uchar* curResult = g_odata + y * _width + x;

	if( (x > _width - _ptnWidth) || (y > _height - _ptnHeight) )
	{
		*curResult = 0xFF;
		return;
	}

	uchar* curPattern = g_pdata;
	float result = 0;
	float nom1 = 0;
	float nom2 = 0;

	for(int i = 0; i < _ptnHeight; ++i)
	{
		uchar* curRow = g_idata + x + (y + i) * _width;
		for(int j = 0; j < _ptnWidth; ++j)
		{			
			uchar cv = *curRow;
			uchar cp = *curPattern;	
			int delta = cv - cp;
			result += delta * delta;
			//result += cv;
			nom1 += cv * cv;
			nom2 += cp * cp;
			++curRow;
			++curPattern;
		}
	}

	result /= sqrt(nom1 * nom2);
	result *= 1024; // expanding value range
	result = result > 255 ? 255 : result;
	*curResult = (uchar)(result);
}


extern "C" unsigned char* correlationCpu( unsigned char* _input, int _width, int _height, unsigned char* _pattern, int _ptnwidth, int _ptnheight)
{
	uchar* fResult = new uchar[_width * _height];

	for(int i = 0; i < _width; ++i)
		for(int j = 0; j < _height; ++j)
		{
			correlationCpu( _input, _pattern, fResult, _width, _height, _ptnwidth, _ptnheight, i, j);
		}

	return fResult;
}

extern "C" unsigned char* correlationCuda( int argc, char** argv, 
								   unsigned char* _input, int _width, int _height, unsigned char* _pattern, int _ptnwidth, int _ptnheight)
{
	// инициализировать устройство
	CUT_DEVICE_INIT(argc, argv);

	// инициализировать таймер
    unsigned int timer = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    // подготовка к загрузке данных
	int imsize = _width * _height;
	int ptnsize = _ptnwidth * _ptnheight;
	uchar* fImage = _input;
	uchar* fPattern = _pattern;

	// выделить пам€ть дл€ изображени€
	uchar* d_idata = NULL;
	printf("Allocating device memory for image\n");
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, imsize * sizeof(unsigned char)));
	printf("Address: %d\n", d_idata);

	// выделить пам€ть дл€ шаблона
	uchar* d_pdata = NULL;
	printf("Allocating device memory for pattern\n");
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_pdata, ptnsize * sizeof(unsigned char)));
	printf("Address: %d\n", d_pdata);

	// загрузить изображение
    CUDA_SAFE_CALL( cudaMemcpy( d_idata, fImage, imsize * sizeof(unsigned char),
                                cudaMemcpyHostToDevice) );
	CUT_CHECK_ERROR("Image loading failed\n");

	// загрузить шаблон
	CUDA_SAFE_CALL( cudaMemcpy( d_pdata, fPattern, ptnsize * sizeof(unsigned char),
                                cudaMemcpyHostToDevice) );   
	CUT_CHECK_ERROR("Pattern loading failed\n");

	// выделить пам€ть дл€ результата
    uchar* d_odata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, imsize * sizeof(unsigned char) ));
	printf("Address: %d\n", d_odata);

	// установить размер блока потоков
	#define BLOCK_SIZE 4
	#define THREAD_SIZE_X 16
	#define THREAD_SIZE_Y 16

	// задать параметры вызова €дра
    dim3  grid( BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3  threads( THREAD_SIZE_X, THREAD_SIZE_Y, 1);

	// установить размер шага
	int stepSizeX = (THREAD_SIZE_X * BLOCK_SIZE);
	int stepSizeY = (THREAD_SIZE_Y * BLOCK_SIZE);

	

	// вывести оповещение о начале обработки
	printf("starting\n");

    // выполнить проход по изображению
	for(int i = 0; i < _width; i += stepSizeX)
		for(int j = 0; j < _height; j += stepSizeY)
		{
			// вызвать €дро (16 блоков по 256 потоков)
			correlation<<< grid, threads >>>( d_idata, d_pdata, d_odata, _width, _height, _ptnwidth, _ptnheight, i, j);
			cudaThreadSynchronize();
		}   

    // проверить успех выполнени€
    CUT_CHECK_ERROR("Kernel execution failed");

    // выгрузить данные с устройства в пам€ть
	uchar* fResult = (uchar*)malloc(imsize * sizeof(unsigned char));
    CUDA_SAFE_CALL( cudaMemcpy( fResult, d_odata, imsize,
                                cudaMemcpyDeviceToHost) );
	CUT_CHECK_ERROR("Output copy failed");

	// остановить таймер, вывести оценку времени
    CUT_SAFE_CALL( cutStopTimer( timer)); 
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    CUT_SAFE_CALL( cutDeleteTimer( timer));
	
	// вернуть результат
	return fResult;
}

