__global__ void
filterName( uchar* g_idata, uchar* g_odata, int _width, int _height, float _scale) 
{
	unsigned int patternOffset = blockIdx.x + blockIdx.y * _width;

	if( (blockIdx.x < 1) || (blockIdx.y < 1) || (blockIdx.x >= _width - 2 ) || (blockIdx.y >= _height - 2) )
	{
		g_odata[patternOffset] = 0xFF;
		return;
	}

	unsigned int tlOffset = blockIdx.x - 1 + (blockIdx.y - 1)* _width;
	
	if( threadIdx.x == 0 )
	{
		PATNDATA(0) = g_idata[tlOffset];
		PATNDATA(1) = g_idata[tlOffset + 1];
		PATNDATA(2) = g_idata[tlOffset + 2];

		tlOffset += _width;

		PATNDATA(3) = g_idata[tlOffset];
		PATNDATA(4) = g_idata[tlOffset + 1];
		PATNDATA(5) = g_idata[tlOffset + 2];

		tlOffset += _width;

		PATNDATA(6) = g_idata[tlOffset];
		PATNDATA(7) = g_idata[tlOffset + 1];
		PATNDATA(8) = g_idata[tlOffset + 2];

		KERNELDATAY(0) = K_UR; KERNELDATAY(1) = K_MR; KERNELDATAY(2) = K_BR;
		KERNELDATAY(3) = K_UM; KERNELDATAY(4) = K_MM; KERNELDATAY(5) = K_BM;	
		KERNELDATAY(6) = K_UL; KERNELDATAY(7) = K_ML; KERNELDATAY(8) = K_BL;

		KERNELDATAX(0) = K_UL; KERNELDATAX(1) = K_UM; KERNELDATAX(2) = K_UR;
		KERNELDATAX(3) = K_ML; KERNELDATAX(4) = K_MM; KERNELDATAX(5) = K_MR;	
		KERNELDATAX(6) = K_BL; KERNELDATAX(7) = K_BM; KERNELDATAX(8) = K_BR;
	}
	__syncthreads();

	PATNDATAY( threadIdx.x ) = PATNDATA(threadIdx.x) * KERNELDATAY(threadIdx.x);
	PATNDATA( threadIdx.x ) = PATNDATA(threadIdx.x) * KERNELDATAX(threadIdx.x);
	//KERNELDATAY( threadIdx.x );

	__syncthreads();

	if( threadIdx.x == 0 )
	{
		int result = 0;
		int resulty = 0;
#pragma unroll 9
		for(int i = 0; i < 9; ++i)
		{
			result += PATNDATA(i);
			resulty += PATNDATAY(i);
		}
		result = _scale * ( abs(result) + abs(resulty) );
		if( result > 0xFF ) result = 0xFF;
		else if( result < 0 ) result = 0;

		g_odata[patternOffset] = 0xFF - result;//(uchar)result;
	}
	__syncthreads();
}