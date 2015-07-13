#ifdef CORRELATIONCUDALIBRARY_EXPORTS
#define CORRELATIONCUDALIBRARY_API __declspec(dllexport)
#else
#define CORRELATIONCUDALIBRARY_API __declspec(dllimport)
#endif

/**
 * Просчет корреляции средствами CPU
 *\param _input Построчный массив с исходным изображением
 *\param _width Ширина исходного изображения
 *\param _height Высота исходного изображения
 *\param _pattern Построчный массив с шаблоном
 *\param _ptnwidth Ширина шаблона
 *\param _ptnheight Высота шаблона
 *\return 
 */
CORRELATIONCUDALIBRARY_API 
unsigned char* correlationCpuL( unsigned char* _input, int _width, int _height, unsigned char* _pattern, int _ptnwidth, int _ptnheight);

/**
 *! Просчет корреляции средствами видеокарты
 *\param _input Построчный массив с исходным изображением
 *\param _width Ширина исходного изображения
 *\param _height Высота исходного изображения
 *\param _pattern Построчный массив с шаблоном
 *\param _ptnwidth Ширина шаблона
 *\param _ptnheight Высота шаблона
 *\return 
 */
CORRELATIONCUDALIBRARY_API 
unsigned char* correlationCudaL( unsigned char* _input, int _width, int _height, unsigned char* _pattern, int _ptnwidth, int _ptnheight);

