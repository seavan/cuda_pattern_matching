/**
 @brief Алгоритм быстрой ненормализованной корреляции
 @author Масленников К. 
 **/

#include "simple.h"
#include "math.h"

#include <algorithm>

// эмуляция многоядерности куды на цпу. используется один поток с идентификатором (0,0)
#define THREADNO 1
#define THREADIDX_Y 0
#define THREADIDX_X 0

/**
 @brief Расчитать таблицу суммации
 @param _input Входное построчное изображение
 @param _width Ширина
 @param _height Высота
 **/
int* calculateSumTable(unsigned char* _input, int _width, int _height)
{
	// таблица создается следующего формата:

	// 0  0  0  0  0  0  0 
	// 0  x  x  x  x  x  x
	// 0  x  x  x  x  x  x 

	// т.е. первая строка и первый столбец - нулевые, для увеличения эффективности расчетов все данные сдвинуты вправо и вниз
	int size = (_width + 1) * (_height + 1) * sizeof(int);
	int* result = (int*) malloc( size );
	int* origResult = result;
	memset(result, 0, size);
	_width += 1;
	result += _width;

	int* current = result;
	unsigned char* currentInput = _input;
	for( int i = 0; i < _height; ++i )
	{
		++current;

		for ( int j = 0; j < _width - 1; ++j )
		{
			// формулу см. в ПДФ
			*current = *currentInput + *(current - 1) -  *(current - _width - 1) + *(current - _width);
			++current;
			++currentInput;
		}	
	}

	return origResult;
}

/**
 @brief Расчитать таблицу квадратичной суммации
 @param _input Входное построчное изображение
 @param _width Ширина
 @param _height Высота
 **/
int* calculateSquareSumTable(unsigned char* _input, int _width, int _height)
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
 @brief Выровнять по среднему значению яркости изображение. На данный момент не используется.
 @param _input Входное построчное изображение
 @param _width Ширина
 @param _height Высота
 */
unsigned char* calculateMeanImage(unsigned char* _input, int _width, int _height)
{
	unsigned char* current = _input;
	int sum = 0;
	for( int i = 0; i < _height; ++i )
	{	
		for ( int j = 0; j < _width; ++j )
		{
			sum += *current;
			++current;
		}	
	}

	unsigned char mean = sum / (_width * _height);

	current = _input;
	
	for( int i = 0; i < _height; ++i )
	{	
		for ( int j = 0; j < _width; ++j )
		{
			*current = *current - sum;
			++current;
		}	
	}	

	return _input;
}

/**
 @brief Расчитать энергию изображения. На данный момент не используется. Формулу см. ПДФ
 @param _input Входное построчное изображение
 @param _width Ширина
 @param _height Высота
 */
int calculateEnergy(unsigned char* _input, int _width, int _height)
{
	unsigned char* current = _input;
	int result = 0;
	
	for( int i = 0; i < _height; ++i )
	{	
		for ( int j = 0; j < _width; ++j )
		{
			result += (*current) * (*current);
			++current;
		}	
	}	
	return result;
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
int getSumFromTable(int* _input, int _width, int _height, int _sx, int _sy, int _sw, int _sh)
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
void doMatching(unsigned char* _input, int _width, int _height, 
				unsigned char* _pattern, int _ptnWidth, int _ptnHeight,
				unsigned char* _output)
{
	// нижеприведенный блок будет использоваться в кудовском коде, здесь для сравнения дан просто

	int blockHeight = _height / THREADNO;
	int blockWidth = _width / THREADNO;
	int startY = blockHeight * THREADIDX_Y;
	int startX = blockWidth * THREADIDX_X;
	int endY = std::min(_height - _ptnHeight + 1, startY + blockHeight);
	int endX = std::min(_width - _ptnWidth + 1, startX + blockWidth);

	// конец блока

	// входное изображение
	unsigned char* input = _input;

	// паттерн
	unsigned char* input2 = _pattern;

	int WIDTH = _width;
	int HEIGHT = _height;
	int PTNWIDTH = _ptnWidth;
	int PTNHEIGHT = _ptnHeight;

	// таблица суммации обычная (не используется)
	//int* sumTable = calculateSumTable(input, WIDTH, HEIGHT);

	// таблица суммации квадратичная
	int* squareSumTable = calculateSquareSumTable(input, WIDTH, HEIGHT);

	// энергия шаблона (не используется)
	//double energy = calculateEnergy(input2, PTNWIDTH, PTNHEIGHT);

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
unsigned char* simpleMatchingCPU( int argc, char** argv, 
								   unsigned char* _input, int _width, int _height, 
				unsigned char* _pattern, int _ptnWidth, int _ptnHeight)
{
    // allocate device memory
	int imsize = _width * _height;
	unsigned char* fResult = new unsigned char[imsize];


	doMatching(_input, _width, _height, _pattern, _ptnWidth, _ptnHeight, fResult);
	return fResult;
}
