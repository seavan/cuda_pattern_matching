#ifdef FILTRATIONLIBRARY_EXPORTS
#define FILTRATIONLIBRARY_API __declspec(dllexport)
#else
#define FILTRATIONLIBRARY_API __declspec(dllimport)
#endif

FILTRATIONLIBRARY_API 
unsigned char* filtration(unsigned char* _input, int _width, int _height, int _method, float _coeff);

