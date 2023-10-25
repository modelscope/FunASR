
#ifndef ALIGNEDMEM_H
#define ALIGNEDMEM_H

namespace funasr {
extern void *AlignedMalloc(size_t alignment, size_t required_bytes);
extern void AlignedFree(void *p);

} // namespace funasr
#endif
