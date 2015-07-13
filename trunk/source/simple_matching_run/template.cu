// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

#define THREADNO 16

// эмуляция многоядерности куды на цпу. используется один поток с идентификатором (0,0)
#define THREADNO 1
#define THREADIDX_Y threadIdx.x
#define THREADIDX_X threadIdx.y

/**
 @brief Расчитать таблицу квадратичной суммации
 @param _input Входное построчное изображение
 @param _width Ширина
 @param _height Высота
 **/
int* calculateSquareSumTableCuda(unsigned char* _input, int _width, int _height)
{
	int size = (_width + 1) * (_height + 1) * sizeof(int);
	int* result = (int*) malloc( size );
	int* origResult = result;
	memset(result, 0, size);
	_width += 1;
	result += _width;
	int* origResult2 = result;

	int* current = result;
	unsigned char* currentInput = _input;
	for( int i = 0; i < _height; ++i )
	{
		++current;

		for ( int j = 0; j < _width - 1; ++j )
		{
			*current = (*currentInput) * (*currentInput) + *(current - 1) -  *(current - _width - 1) + *(current - _width);
			++current;
			++currentInput;
		}	
	}

	return origResult;
}

/**
 @brief Получить значение суммы области из просчитанной таблицы суммации
 @param _input Построчное входное изображение
 @param _width Ширина изображения
 @param _height Высота изображения
 @param _sx Координата X левого верхнего угла области
 @param _sy Координата Y левого верхнего угла области
 @param _sw Ширина области
 @param _sh Высота области
 **/
__device__ int getSumFromTable(int* _input, int _width, int _height, int _sx, int _sy, int _sw, int _sh)
{
	_width += 1;
	int* corner = _input + (_width * (_sy)) + _sx;
	return *corner - *(corner + _sw) - *(corner + _sh * _width) + *(corner + _sh * _width + _sw);
}

/**
 @brief Интерфейс к методу корреляции
 @param _input Построчное входное изображение оригинала
 @param _width Ширина изображения
 @param _height Высота изображения
 @param _pattern Построчное входное изображение шаблона
 @param _ptnWidth Ширина изображения шаблона
 @param _ptnHeight Высота изображения шаблона
 @param _output Результат
 **/
__global__ void doMatching(unsigned char* _input, int _width, int _height, 
				unsigned char* _pattern, int _ptnWidth, int _ptnHeight,
				unsigned char* _output, int * squareSumTable)
{
	// нижеприведенный блок будет использоваться в кудовском коде, здесь для сравнения дан просто

	int blockHeight = _height / THREADNO;
	int blockWidth = _width / THREADNO;
	int startY = blockHeight * THREADIDX_Y;
	int startX = blockWidth * THREADIDX_X;
	int endY = min(_height - _ptnHeight + 1, startY + blockHeight);
	int endX = min(_width - _ptnWidth + 1, startX + blockWidth);

	// конец блока

	// входное изображение
	unsigned char* input = _input;

	// паттерн
	unsigned char* input2 = _pattern;

	int WIDTH = _width;
	int HEIGHT = _height;
	int PTNWIDTH = _ptnWidth;
	int PTNHEIGHT = _ptnHeight;



	// стартовые указатели для оригинала и шаблона
	unsigned char* inputCurrent = input;
	unsigned char* ptnCurrent = input2;

	for(int py = startY; py < endY; ++py)
	{
		unsigned char* curInputRow = _input + _width * py;
		unsigned char* curOutputRow = _output + _width * py;
		for(int px = startX; px < endX; ++px)
		{	
			// расчет числителя как суммы квадратов отклонений
			double nom = 0;

			// расчет знаменателя как суммы квадратов области оригинала
			double denom = sqrt( (double)getSumFromTable(squareSumTable, WIDTH, HEIGHT, px, py, PTNWIDTH, PTNHEIGHT) )
			;

			unsigned char* curPattern = _pattern;
			for(int y = 0; y < _ptnHeight; ++y)
			{
				unsigned char* ptCurInputRow =  curInputRow + _width * y;
				for(int x = 0; x < _ptnWidth; ++x)
				{
					int val = (*ptCurInputRow) - (*curPattern);
					nom += val * val;					
					++ptCurInputRow;
					++curPattern;
				}
				
			}

			// коэффициент корреляции
			double val = 1 - sqrt(nom) / denom;

			// форматирование для вывода
			*curOutputRow = 255.0 * val;	

			++curInputRow;
			++curOutputRow;
		}
	}
}

/**
 @brief Интерфейс к методу корреляции
 @param argc Не используется. Для совместимости с CUDA
 @param argv Не используется. Для совместимости с CUDA
 @param _input Построчное входное изображение оригинала
 @param _width Ширина изображения
 @param _height Высота изображения
 @param _pattern Построчное входное изображение шаблона
 @param _ptnWidth Ширина изображения шаблона
 @param _ptnHeight Высота изображения шаблона
 **/
extern "C" unsigned char* simpleMatchingCuda( int argc, char** argv, 
								   unsigned char* _input, int _width, int _height, 
				unsigned char* _pattern, int _ptnWidth, int _ptnHeight)
{
	CUT_DEVICE_INIT(argc, argv);
    unsigned int timer = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    // allocate device memory
	int imsize = _width * _height;
	int ptnsize = _ptnWidth * _ptnHeight;

	unsigned char* fImage = new unsigned char[imsize];
	unsigned char* fResult = new unsigned char[imsize];
	//memset( fImage, 0, imsize );
	memcpy( fImage, _input, imsize );

	unsigned char* d_pdata = NULL;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_pdata, ptnsize));
    // copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_pdata, _pattern, ptnsize,
                                cudaMemcpyHostToDevice) );
    
	unsigned char* d_idata = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, imsize * sizeof(unsigned char) ));

	unsigned char* d_resdata = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_resdata, imsize * sizeof(unsigned char) ));
	
	cudaMemset(d_resdata, 255, imsize * sizeof(unsigned char));
	// copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_idata, fImage, imsize * sizeof(unsigned char),
                                cudaMemcpyHostToDevice) );

	
	// таблица суммации квадратичная
	int* squareSumTable = calculateSquareSumTableCuda(_input, _width, _height);

	int* d_sumdata = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_sumdata, imsize * sizeof(int) ));
	// copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_sumdata, squareSumTable, imsize * sizeof(int),
                                cudaMemcpyHostToDevice) );

    // execute the kernel
    // setup execution parameters
    dim3  threads( THREADNO, THREADNO, 1 );

    doMatching<<< 1, threads>>>(d_idata, _width, _height, d_pdata, _ptnWidth, _ptnHeight, d_resdata, d_sumdata);

    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( fImage, d_resdata, imsize,
                                cudaMemcpyDeviceToHost) );

	CUT_CHECK_ERROR("Output copy failed");

    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    CUT_SAFE_CALL( cutDeleteTimer( timer));
	
	
	return fImage;
}

