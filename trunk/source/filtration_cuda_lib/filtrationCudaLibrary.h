#ifdef FILTRATIONCUDALIBRARY_EXPORTS
#define FILTRATIONCUDALIBRARY_API __declspec(dllexport)
#else
#define FILTRATIONCUDALIBRARY_API __declspec(dllimport)
#endif

/**
 * Фильтрация изображения заданным методом с использованием платформы GPU
 *\param _input Построчный массив с исходным изображением
 *\param _width Ширина исходного изображения
 *\param _height Высота исходного изображения
 *\param _method Метод (0 - Собель, 1 - Пюритт, 2 - Лаплас)
 *\param _scale Резкость фильтрации
 *\return Массив байт с отфильтрованным изображением с глубиной цвета - 8 (grayscale)
 */
FILTRATIONCUDALIBRARY_API 
unsigned char* filtrationCudaL( unsigned char* _input, int _width, int _height, int _method, float _scale);

