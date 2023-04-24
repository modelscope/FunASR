#include "precomp.h"
void *AlignedMalloc(size_t alignment, size_t required_bytes)
{
    void *p1;  // original block
    void **p2; // aligned block
    int offset = alignment - 1 + sizeof(void *);
    if ((p1 = (void *)malloc(required_bytes + offset)) == NULL) {
        return NULL;
    }
    p2 = (void **)(((size_t)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
}

void AlignedFree(void *p)
{
    free(((void **)p)[-1]);
}
