#ifndef __CUDA_DEBUG_H__
#define __CUDA_DEBUG_H__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#ifdef DEBUG
#define CUDA_CHECK(value) {                                             \
    cudaError_t _m_cudaStat = value;                                    \
    if (_m_cudaStat != cudaSuccess) {                                   \
        fprintf(stderr, "Error %s at line %d in file %s\n",             \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1); }}

#else
#define CUDA_CHECK(value) { value; }
#endif
#endif
