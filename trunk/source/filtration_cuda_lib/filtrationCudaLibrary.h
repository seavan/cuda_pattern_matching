#ifdef FILTRATIONCUDALIBRARY_EXPORTS
#define FILTRATIONCUDALIBRARY_API __declspec(dllexport)
#else
#define FILTRATIONCUDALIBRARY_API __declspec(dllimport)
#endif

/**
 * ���������� ����������� �������� ������� � �������������� ��������� GPU
 *\param _input ���������� ������ � �������� ������������
 *\param _width ������ ��������� �����������
 *\param _height ������ ��������� �����������
 *\param _method ����� (0 - ������, 1 - ������, 2 - ������)
 *\param _scale �������� ����������
 *\return ������ ���� � ��������������� ������������ � �������� ����� - 8 (grayscale)
 */
FILTRATIONCUDALIBRARY_API 
unsigned char* filtrationCudaL( unsigned char* _input, int _width, int _height, int _method, float _scale);

