/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

#ifndef ALIGNEDMEM_H
#define ALIGNEDMEM_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void *aligned_malloc(size_t alignment, size_t required_bytes);
extern void aligned_free(void *p);

#endif
