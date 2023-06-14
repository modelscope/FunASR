/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  online_utils.hpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#ifndef online_utils_hpp
#define online_utils_hpp

#include <stdio.h>
#include <vector>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include "cblas.h"
#endif

//using namespace QNN_NS;

inline void MatMul(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
#if defined(__APPLE1__) && TARGET_OS_IPHONE
//    QNN_CONCURRENCY_BEGIN(tid, 1){
        vDSP_mmul(A, 1, B, 1, C, 1, M, N, K);
//    }
//    QNN_CONCURRENCY_END();
    
#else
    int lda = K; // A column
    int ldb = N; // B column
    int ldc = N; // C column
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

inline void MatMul(int M, int N, int K, float alpha, float *A, const enum CBLAS_TRANSPOSE __TransA, float *B, const enum CBLAS_TRANSPOSE __TransB, float beta, float *C) {
#if defined(__APPLE1__) && TARGET_OS_IPHONE
//    QNN_CONCURRENCY_BEGIN(tid, 1){
        vDSP_mmul(A, 1, B, 1, C, 1, M, N, K);
//    }
//    QNN_CONCURRENCY_END();
    
#else
    int lda = K; // A column
    int ldb = N; // B column
    int ldc = N; // C column
    cblas_sgemm(CblasRowMajor, __TransA, __TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

inline void softmax(float *din, int mask, int len)
{
    float *tmp = (float *)malloc(mask * sizeof(float));
    int i;
    float sum = 0;
    float max = -INFINITY;

    for (i = 0; i < mask; i++) {
        max = max < din[i] ? din[i] : max;
    }

    for (i = 0; i < mask; i++) {
        tmp[i] = exp(din[i] - max);
        sum += tmp[i];
    }
    for (i = 0; i < mask; i++) {
        din[i] = tmp[i] / sum;
    }
    free(tmp);
    for (i = mask; i < len; i++) {
        din[i] = 0;
    }
}

#endif /* online_utils_hpp */
