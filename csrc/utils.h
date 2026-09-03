/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 *
 * FlashNystrom - CUDA utility helpers, type conversions, warp reductions etc.
 * Most of the heavy lifting for type dispatch happens here.
 ******************************************************************************/
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cassert>
#include <stdexcept>
#include <string>
#include <sstream>

#include <cutlass/numeric_types.h>

// Error checking macros. These throw std::runtime_error rather than calling
// abort() so a failure inside the kernel pipeline propagates as a normal
// Python exception. Aborting the host process from a library used inside a
// training loop would kill the user's whole run on an intermittent failure
// (e.g. a transient OOM); a thrown exception lets PyTorch's autograd unwind
// cleanly and surface a recoverable RuntimeError to Python.

#define FN_CUDA_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::ostringstream _fn_oss;                                        \
            _fn_oss << "[FlashNystrom] CUDA error at " << __FILE__             \
                    << ":" << __LINE__ << ": " << cudaGetErrorString(err)      \
                    << " (code " << static_cast<int>(err) << ")";              \
            throw std::runtime_error(_fn_oss.str());                           \
        }                                                                      \
    } while (0)

#define FN_CUDA_KERNEL_CHECK()                                                 \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            std::ostringstream _fn_oss;                                        \
            _fn_oss << "[FlashNystrom] Kernel launch error at " << __FILE__    \
                    << ":" << __LINE__ << ": " << cudaGetErrorString(err);     \
            throw std::runtime_error(_fn_oss.str());                           \
        }                                                                      \
    } while (0)

#define FN_CHECK(cond, msg)                                                    \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::ostringstream _fn_oss;                                        \
            _fn_oss << "[FlashNystrom] CHECK failed at " << __FILE__           \
                    << ":" << __LINE__ << ": " << (msg);                       \
            throw std::runtime_error(_fn_oss.str());                           \
        }                                                                      \
    } while (0)

// Hard limits. The kernels support D in {64, 128} and m <= 64; the
// constants below are the theoretical maxima used for validation.

namespace flash_nystrom {

constexpr int kMaxHeadDim = 256;
// The fused kernels use fixed m=64 tiles; the public entry (flash_nystrom.cu)
// hard-rejects m > 64, and m > 64 dispatches to the PyTorch reference at the
// Python level. This is the real limit, kept in sync with that check.
constexpr int kMaxLandmarks = 64;
constexpr float kLog2e = 1.4426950408889634f;  // log2(e) for exp2-based softmax trick

// GPU arch detection, cached after the first call so the driver is not re-queried.

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

// Total shared memory per multiprocessor (164KB on sm_80, 100KB on sm_86/89
// and consumer Blackwell, 228KB on sm_90/100). Used by launch-time dispatch
// to decide whether a larger SMEM footprint would cost resident CTAs (e.g.
// the pipelined kernel3 forward needs 3 CTAs x kSmemFwdBytes to keep the
// occupancy that __launch_bounds__(kNThreads, 3) targets). Cached.
inline size_t get_smem_per_multiprocessor() {
    static size_t cached = 0;
    if (cached == 0) {
        int device = -1;
        FN_CUDA_CHECK(cudaGetDevice(&device));
        int val = 0;
        FN_CUDA_CHECK(cudaDeviceGetAttribute(
            &val, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
        cached = static_cast<size_t>(val);
    }
    return cached;
}

// SM count for the current device. Used by launch-time dispatch to decide
// whether a grid is large enough to fill the GPU (e.g. grid(BH) starves a
// 108-SM A100 when BH is 4). Cached after first query.
inline int get_sm_count() {
    static int cached = -1;
    if (cached < 0) {
        int device = -1;
        FN_CUDA_CHECK(cudaGetDevice(&device));
        int val = 0;
        FN_CUDA_CHECK(cudaDeviceGetAttribute(
            &val, cudaDevAttrMultiProcessorCount, device));
        cached = val;
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
// Used throughout the softmax and the backward.

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
// scratch needs at least (blockDim.x / 32) floats; the caller must allocate it
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
    // Capture, then sync before returning. Without the trailing barrier a
    // caller that reuses `scratch` immediately (e.g. block_reduce_sum on the
    // same buffer) can write scratch[warp] while another thread is still
    // reading scratch[0] here — a real read/write race (caught by racecheck).
    float result = scratch[0];
    __syncthreads();
    return result;
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
    // See block_reduce_max: trailing barrier so a subsequent reduction reusing
    // `scratch` cannot overwrite scratch[0] before all threads have read it.
    float result = scratch[0];
    __syncthreads();
    return result;
}

} // namespace flash_nystrom
