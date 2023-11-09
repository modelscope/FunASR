#ifndef WIN_FUNC_
#define WIN_FUNC_
#ifdef _WIN32
#include <windows.h>
#include <winsock.h>
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