/**
 @brief �������� ������� ����������������� ����������
 @author ����������� �. 
 **/

#include "simple.h"
#include "math.h"

#include <algorithm>

// �������� �������������� ���� �� ���. ������������ ���� ����� � ��������������� (0,0)
#define THREADNO 1
#define THREADIDX_Y 0
#define THREADIDX_X 0

/**
 @brief ��������� ������� ��������
 @param _input ������� ���������� �����������
 @param _width ������
 @param _height ������
 **/
int* calculateSumTable(unsigned char* _input, int _width, int _height)
{
	// ������� ��������� ���������� �������:

	// 0  0  0  0  0  0  0 
	// 0  x  x  x  x  x  x
	// 0  x  x  x  x  x  x 

	// �.�. ������ ������ � ������ ������� - �������, ��� ���������� ������������� �������� ��� ������ �������� ������ � ����
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
			// ������� ��. � ���
			*current = *currentInput + *(current - 1) -  *(current - _width - 1) + *(current - _width);
			++current;
			++currentInput;
		}	
	}

	return origResult;
}

/**
 @brief ��������� ������� ������������ ��������
 @param _input ������� ���������� �����������
 @param _width ������
 @param _height ������
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
 @brief ��������� �� �������� �������� ������� �����������. �� ������ ������ �� ������������.
 @param _input ������� ���������� �����������
 @param _width ������
 @param _height ������
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
 @brief ��������� ������� �����������. �� ������ ������ �� ������������. ������� ��. ���
 @param _input ������� ���������� �����������
 @param _width ������
 @param _height ������
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
 @brief �������� �������� ����� ������� �� ������������ ������� ��������
 @param _input ���������� ������� �����������
 @param _width ������ �����������
 @param _height ������ �����������
 @param _sx ���������� X ������ �������� ���� �������
 @param _sy ���������� Y ������ �������� ���� �������
 @param _sw ������ �������
 @param _sh ������ �������
 **/
int getSumFromTable(int* _input, int _width, int _height, int _sx, int _sy, int _sw, int _sh)
{
	_width += 1;
	int* corner = _input + (_width * (_sy)) + _sx;
	return *corner - *(corner + _sw) - *(corner + _sh * _width) + *(corner + _sh * _width + _sw);
}

/**
 @brief ��������� � ������ ����������
 @param _input ���������� ������� ����������� ���������
 @param _width ������ �����������
 @param _height ������ �����������
 @param _pattern ���������� ������� ����������� �������
 @param _ptnWidth ������ ����������� �������
 @param _ptnHeight ������ ����������� �������
 @param _output ���������
 **/
void doMatching(unsigned char* _input, int _width, int _height, 
				unsigned char* _pattern, int _ptnWidth, int _ptnHeight,
				unsigned char* _output)
{
	// ��������������� ���� ����� �������������� � ��������� ����, ����� ��� ��������� ��� ������

	int blockHeight = _height / THREADNO;
	int blockWidth = _width / THREADNO;
	int startY = blockHeight * THREADIDX_Y;
	int startX = blockWidth * THREADIDX_X;
	int endY = std::min(_height - _ptnHeight + 1, startY + blockHeight);
	int endX = std::min(_width - _ptnWidth + 1, startX + blockWidth);

	// ����� �����

	// ������� �����������
	unsigned char* input = _input;

	// �������
	unsigned char* input2 = _pattern;

	int WIDTH = _width;
	int HEIGHT = _height;
	int PTNWIDTH = _ptnWidth;
	int PTNHEIGHT = _ptnHeight;

	// ������� �������� ������� (�� ������������)
	//int* sumTable = calculateSumTable(input, WIDTH, HEIGHT);

	// ������� �������� ������������
	int* squareSumTable = calculateSquareSumTable(input, WIDTH, HEIGHT);

	// ������� ������� (�� ������������)
	//double energy = calculateEnergy(input2, PTNWIDTH, PTNHEIGHT);

	// ��������� ��������� ��� ��������� � �������
	unsigned char* inputCurrent = input;
	unsigned char* ptnCurrent = input2;

	for(int py = startY; py < endY; ++py)
	{
		unsigned char* curInputRow = _input + _width * py;
		unsigned char* curOutputRow = _output + _width * py;
		for(int px = startX; px < endX; ++px)
		{	
			// ������ ��������� ��� ����� ��������� ����������
			double nom = 0;

			// ������ ����������� ��� ����� ��������� ������� ���������
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

			// ����������� ����������
			double val = 1 - sqrt(nom) / denom;

			// �������������� ��� ������
			*curOutputRow = 255.0 * val;	

			++curInputRow;
			++curOutputRow;
		}
	}
}

/**
 @brief ��������� � ������ ����������
 @param argc �� ������������. ��� ������������� � CUDA
 @param argv �� ������������. ��� ������������� � CUDA
 @param _input ���������� ������� ����������� ���������
 @param _width ������ �����������
 @param _height ������ �����������
 @param _pattern ���������� ������� ����������� �������
 @param _ptnWidth ������ ����������� �������
 @param _ptnHeight ������ ����������� �������
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
