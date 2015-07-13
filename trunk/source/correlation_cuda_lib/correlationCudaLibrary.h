#ifdef CORRELATIONCUDALIBRARY_EXPORTS
#define CORRELATIONCUDALIBRARY_API __declspec(dllexport)
#else
#define CORRELATIONCUDALIBRARY_API __declspec(dllimport)
#endif

/**
 * ������� ���������� ���������� CPU
 *\param _input ���������� ������ � �������� ������������
 *\param _width ������ ��������� �����������
 *\param _height ������ ��������� �����������
 *\param _pattern ���������� ������ � ��������
 *\param _ptnwidth ������ �������
 *\param _ptnheight ������ �������
 *\return 
 */
CORRELATIONCUDALIBRARY_API 
unsigned char* correlationCpuL( unsigned char* _input, int _width, int _height, unsigned char* _pattern, int _ptnwidth, int _ptnheight);

/**
 *! ������� ���������� ���������� ����������
 *\param _input ���������� ������ � �������� ������������
 *\param _width ������ ��������� �����������
 *\param _height ������ ��������� �����������
 *\param _pattern ���������� ������ � ��������
 *\param _ptnwidth ������ �������
 *\param _ptnheight ������ �������
 *\return 
 */
CORRELATIONCUDALIBRARY_API 
unsigned char* correlationCudaL( unsigned char* _input, int _width, int _height, unsigned char* _pattern, int _ptnwidth, int _ptnheight);

