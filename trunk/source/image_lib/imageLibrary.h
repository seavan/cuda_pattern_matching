#ifdef IMAGELIBRARY_EXPORTS
#define IMAGELIBRARY_API __declspec(dllexport)
#else
#define IMAGELIBRARY_API __declspec(dllimport)
#endif

/**
 * јвтоматическое определение формата и загрузка изображени€ из указанного файла
 *\param _fileName ѕуть к файлу с изображением
 *\param _width Ўирина загруженного изображени€
 *\param _height ¬ысота загруженного изображени€
 *\return ћассив байт загруженного изображени€ в режиме 8 bit (grayscale), ширину и высоту полученного изображени€ по ссылке
 */
IMAGELIBRARY_API 
unsigned char* loadImage(const char* _fileName, int& _width, int& _height);

/**
 * —охранение изображени€ в формате BMP
 *\param _fileName ѕуть к файлу с изображением
 *\param _width Ўирина загруженного изображени€
 *\param _height ¬ысота загруженного изображени€
 *\return ћассив байт загруженного изображени€ в режиме 8 bit (grayscale), ширину и высоту полученного изображени€ по ссылке
 */
IMAGELIBRARY_API
void saveImage(const char* _fileName, unsigned char* _image, int _width, int _height);
