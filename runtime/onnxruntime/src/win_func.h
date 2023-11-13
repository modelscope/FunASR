#ifndef WIN_FUNC_
#define WIN_FUNC_
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define  WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winsock.h>
#include<io.h>

#ifndef R_OK
#define R_OK 4
#endif
#ifndef W_OK
#define W_OK 2
#endif
#ifndef X_OK
#define X_OK 0 
#endif
#ifndef F_OK
#define F_OK 0
#endif
#define access _access

static inline int gettimeofday(struct timeval* tv, void* /*tz*/) {
    FILETIME ft;
    ULARGE_INTEGER li;
    ULONGLONG tt;
    GetSystemTimeAsFileTime(&ft);
    li.LowPart = ft.dwLowDateTime;
    li.HighPart = ft.dwHighDateTime;
    tt = (li.QuadPart - 116444736000000000ULL) / 10;
    tv->tv_sec = tt / 1000000;
    tv->tv_usec = tt % 1000000;
    return 0;
}
#endif
#endif