#include "filtrationLibrary.h"

#include <stdlib.h>
#include <math.h>

void copyMatrix(int _src[3][3], int _dest[3][3])
{
	for(int i = 0; i < 3; ++i)
	{
		for(int j = 0; j < 3; ++j)
		{
			_dest[i][j] = _src[i][j];
		}

	}
}

#define SOBEL 0
#define PURITT 1
#define LAPLAS 2

#define GET_PIXEL(_input, _x, _y, _width) (_input)[(_x) + (_y) * (_width)]

unsigned char* filtration(unsigned char* _input, int _width, int _height, int _method, float _coeff)
{
	unsigned char* result = (unsigned char*)malloc(_width * _height * sizeof(unsigned char));


	int	sobel_GX[3][3];
	int	sobel_GY[3][3];

	/* 3x3 sobel_GX Sobel mask. */
	sobel_GX[0][0] = -1; sobel_GX[0][1] = 0; sobel_GX[0][2] = 1;
	sobel_GX[1][0] = -2; sobel_GX[1][1] = 0; sobel_GX[1][2] = 2;
	sobel_GX[2][0] = -1; sobel_GX[2][1] = 0; sobel_GX[2][2] = 1;

	/* 3x3 sobel_GY Sobel mask. */
	sobel_GY[0][0] =  1; sobel_GY[0][1] =  2; sobel_GY[0][2] =  1;
	sobel_GY[1][0] =  0; sobel_GY[1][1] =  0; sobel_GY[1][2] =  0;
	sobel_GY[2][0] = -1; sobel_GY[2][1] = -2; sobel_GY[2][2] = -1;

	/* 3x3 puritt_GX puritt mask. */
	int	puritt_GX[3][3];
	int	puritt_GY[3][3];

	puritt_GX[0][0] = -1; puritt_GX[0][1] = 0; puritt_GX[0][2] = 1;
	puritt_GX[1][0] = -1; puritt_GX[1][1] = 0; puritt_GX[1][2] = 1;
	puritt_GX[2][0] = -1; puritt_GX[2][1] = 0; puritt_GX[2][2] = 1;

	/* 3x3 puritt_GY puritt mask*/
	puritt_GY[0][0] =  1; puritt_GY[0][1] =  1; puritt_GY[0][2] =  1;
	puritt_GY[1][0] =  0; puritt_GY[1][1] =  0; puritt_GY[1][2] =  0;
	puritt_GY[2][0] = -1; puritt_GY[2][1] = -1; puritt_GY[2][2] = -1;

	/* 3x3 laplas_GX laplas mask.*/
	int	laplas_GX[3][3];
	int	laplas_GY[3][3];
	laplas_GX[0][0] = 0; laplas_GX[0][1] = 1; laplas_GX[0][2] = 0;
	laplas_GX[1][0] = 1; laplas_GX[1][1] = -4; laplas_GX[1][2] = 1;
	laplas_GX[2][0] = 0; laplas_GX[2][1] = 1; laplas_GX[2][2] = 0;

	/* 3x3 laplas_GY laplas mask.*/
	laplas_GY[0][0] =  0; laplas_GY[0][1] =  1; laplas_GY[0][2] =  0;
	laplas_GY[1][0] =  1; laplas_GY[1][1] =  -4; laplas_GY[1][2] =  1;
	laplas_GY[2][0] = 0; laplas_GY[2][1] = 1; laplas_GY[2][2] = 0;

	double coeff = _coeff;
	int GX[3][3];
	int GY[3][3];

	switch(_method)
	{
	case SOBEL: 
		copyMatrix(sobel_GX, GX);
		copyMatrix(sobel_GY, GY);	
		break;
	case PURITT: 
		copyMatrix(puritt_GX, GX);
		copyMatrix(puritt_GY, GY);	
		break;
	case LAPLAS: 
		copyMatrix(laplas_GX, GX);
		copyMatrix(laplas_GY, GY);		
		break;
	}

	/*---------------------------------------------------
	convolution
	---------------------------------------------------*/

	unsigned char* current = result;

	for(int Y = 0; Y < _height; ++Y)  {
		for(int X = 0; X < _width; ++X)  {
			int SUM;
			int sumX = 0;
			int sumY = 0;

			/* image boundaries */
			if(Y == 0 || Y == _height - 1)
				SUM = 0;
			else 
				if( X == 0 || X == _width - 1)
					SUM = 0;
			/* Convolution starts here */
				else   {

					/*-------X GRADIENT APPROXIMATION------*/
					for(int I = -1; I <= 1; I++)  {
						for(int J=-1 ; J<=1; J++)  {
							sumX = sumX + GET_PIXEL(_input, X + I, Y + J, _width) * GX[I+1][J+1];
						}
					}

					/*-------Y GRADIENT APPROXIMATION-------*/
					for(int I=-1; I<=1; I++)  {
						for(int J=-1; J<=1; J++)  {
							sumY = sumY + GET_PIXEL(_input, X + I, Y + J, _width) * GY[I+1][J+1];
						}
					}

					/*---GRADIENT MAGNITUDE APPROXIMATION----*/
					SUM = coeff * sqrt(float(sumX * sumX + sumY * sumY));
				}

			if(SUM>255) SUM=255;
			if(SUM<0) SUM=0;

			unsigned char val = 255 - (unsigned char)(SUM);
			*current = val;
			++current;
		}
	}
	return result;
}

