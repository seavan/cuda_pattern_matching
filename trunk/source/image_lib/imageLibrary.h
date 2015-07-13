#ifdef IMAGELIBRARY_EXPORTS
#define IMAGELIBRARY_API __declspec(dllexport)
#else
#define IMAGELIBRARY_API __declspec(dllimport)
#endif

/**
 * �������������� ����������� ������� � �������� ����������� �� ���������� �����
 *\param _fileName ���� � ����� � ������������
 *\param _width ������ ������������ �����������
 *\param _height ������ ������������ �����������
 *\return ������ ���� ������������ ����������� � ������ 8 bit (grayscale), ������ � ������ ����������� ����������� �� ������
 */
IMAGELIBRARY_API 
unsigned char* loadImage(const char* _fileName, int& _width, int& _height);

/**
 * ���������� ����������� � ������� BMP
 *\param _fileName ���� � ����� � ������������
 *\param _width ������ ������������ �����������
 *\param _height ������ ������������ �����������
 *\return ������ ���� ������������ ����������� � ������ 8 bit (grayscale), ������ � ������ ����������� ����������� �� ������
 */
IMAGELIBRARY_API
void saveImage(const char* _fileName, unsigned char* _image, int _width, int _height);
