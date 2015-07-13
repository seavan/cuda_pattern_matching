#ifdef FILTRATIONLIBRARY_EXPORTS
#define FILTRATIONLIBRARY_API __declspec(dllexport)
#else
#define FILTRATIONLIBRARY_API __declspec(dllimport)
#endif

/**
 * ���������� ����������� �������� ������� (CPU)
 *\param _input ���������� ������ � �������� ������������
 *\param _width ������ ��������� �����������
 *\param _height ������ ��������� �����������
 *\param _method ����� (0 - ������, 1 - ������, 2 - ������)
 *\param _scale �������� ����������
 *\return ������ ���� � ��������������� ������������ � �������� ����� - 8 (grayscale)
 */
FILTRATIONLIBRARY_API 
unsigned char* filtration(unsigned char* _input, int _width, int _height, int _method, float _coeff);

