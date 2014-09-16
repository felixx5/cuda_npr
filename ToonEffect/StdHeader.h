#ifndef STD_HEADER_H_
#define STD_HEADER_H_

#pragma warning(disable:4244)

#include <d3dx9.h>

#define DBG_FILE_PATH "myDebug.txt"
#define DBG_LOG(s, x) {FILE* file = fopen(DBG_FILE_PATH, "a"); fprintf(file, "%s: %d\n", s, x); fclose(file);}

#endif