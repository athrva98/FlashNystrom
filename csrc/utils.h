/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 *
 * FlashNystrom - CUDA utility helpers, type conversions, warp reductions etc.
 * Most of the heavy lifitng for type dispatch happens here.
 ******************************************************************************/
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cassert>

#include <cutlass/numeric_types.h>

// error checking macros — these just abort on failure, no fancy recovery
// honestly the best thing you can do with a cuda error is die loudly

#define FN_CUDA_CHECK(call)                                                    \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "[FlashNystrom] CUDA error at %s:%d: %s (code %d)\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err), (int)err);     \
            abort();                                                            \
        }                                                                       \
    } while (0)

#define FN_CUDA_KERNEL_CHECK()                                                 \
    do {                                                                        \
        cudaError_t err = cudaGetLastError();                                   \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "[FlashNystrom] Kernel launch error at %s:%d: %s\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            abort();                                                            \
        }                                                                       \
    } while (0)

#define FN_CHECK(cond, msg)                                                    \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "[FlashNystrom] CHECK failed at %s:%d: %s\n",      \
                    __FILE__, __LINE__, (msg));                                 \
            abort();                                                            \
        }                                                                       \
    } while (0)

// hard limits — kernels actualy only support D in {64, 128} and m <= 64
// but these are the theoretical maxiums for validation

namespace flash_nystrom {

constexpr int kMaxHeadDim = 256;
constexpr int kMaxLandmarks = 128;
constexpr float kLog2e = 1.4426950408889634f;  // log2(e) for exp2-based softmax trick

// gpu arch detection — cached after first call so we dont keep querying the driver

inline int get_sm_version() {
    static int cached = -1;
    if (cached < 0) {
        int device = -1;
        FN_CUDA_CHECK(cudaGetDevice(&device));
        int major = 0, minor = 0;
        FN_CUDA_CHECK(cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, device));
        FN_CUDA_CHECK(cudaDeviceGetAttribute(
            &minor, cudaDevAttrComputeCapabilityMinor, device));
        cached = major * 10 + minor;
    }
    return cached;
}

inline size_t get_max_smem_per_block() {
    static size_t cached = 0;
    if (cached == 0) {
        int device = -1;
        FN_CUDA_CHECK(cudaGetDevice(&device));
        int val = 0;
        FN_CUDA_CHECK(cudaDeviceGetAttribute(
            &val, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        cached = static_cast<size_t>(val);
    }
    return cached;
}

// type conversions — cutlass types (half_t, bfloat16_t) to/from float
// these are ABI-compatible with __half / __nv_bfloat16 so reinterpret_cast is fine
// following the same pattern as FlashAttention for type dispatch
__device__ __forceinline__ float to_float(float val) { return val; }
__device__ __forceinline__ float to_float(cutlass::half_t val) {
    return __half2float(reinterpret_cast<const __half&>(val));
}
__device__ __forceinline__ float to_float(cutlass::bfloat16_t val) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(val));
}

// from_float: go the other way. float -> whatever element type we need
template <typename T>
__device__ __forceinline__ T from_float(float val);

template <>
__device__ __forceinline__ float from_float<float>(float val) { return val; }

template <>
__device__ __forceinline__ cutlass::half_t from_float<cutlass::half_t>(float val) {
    __half h = __float2half(val);
    return reinterpret_cast<cutlass::half_t&>(h);
}

template <>
__device__ __forceinline__ cutlass::bfloat16_t from_float<cutlass::bfloat16_t>(float val) {
    __nv_bfloat16 b = __float2bfloat16(val);
    return reinterpret_cast<cutlass::bfloat16_t&>(b);
}

// warp-level reductions — butterfly pattern with shfl_xor
// these get used everwhere in softmax and the backward

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// block-wide reductions — warp reduces first, then inter-warp via smem
// scratch needs at least (blockDim.x / 32) floats, dont forget to allocate
__device__ __forceinline__ float block_reduce_max(float val, float* scratch) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;

    val = warp_reduce_max(val);
    if (lane == 0) scratch[warp] = val;
    __syncthreads();

    val = (threadIdx.x < num_warps) ? scratch[threadIdx.x] : -FLT_MAX;
    if (warp == 0) val = warp_reduce_max(val);
    __syncthreads();
    if (threadIdx.x == 0) scratch[0] = val;
    __syncthreads();
    return scratch[0];
}

__device__ __forceinline__ float block_reduce_sum(float val, float* scratch) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) scratch[warp] = val;
    __syncthreads();

    val = (threadIdx.x < num_warps) ? scratch[threadIdx.x] : 0.0f;
    if (warp == 0) val = warp_reduce_sum(val);
    __syncthreads();
    if (threadIdx.x == 0) scratch[0] = val;
    __syncthreads();
    return scratch[0];
}

} // namespace flash_nystrom
