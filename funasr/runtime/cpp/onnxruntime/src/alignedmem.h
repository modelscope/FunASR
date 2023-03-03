
#ifndef ALIGNEDMEM_H
#define ALIGNEDMEM_H



extern void *aligned_malloc(size_t alignment, size_t required_bytes);
extern void aligned_free(void *p);

#endif
