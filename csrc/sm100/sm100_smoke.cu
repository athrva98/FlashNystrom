/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 *
 * CP-B1: standalone Blackwell (sm_100a) tcgen05 GEMM smoke test.
 *
 * Computes S[64 x 128] = A[64 x D] @ B[128 x D]^T in fp16 with fp32
 * accumulation, at exactly the tile shape the FlashNystrom sm100 kernels
 * will use (kTileM = 64 landmark rows, kTileN = 128 streamed rows,
 * kTileK = D in {64, 128}). Validates, in our build system, the full
 * tcgen05 chain the real kernels depend on:
 *   - CollectiveBuilder-derived TiledMma (SM100_MMA_F16BF16_SS) + UMMA
 *     smem layouts
 *   - TMEM allocation / accumulator residency / deallocation
 *   - single-warp MMA issue with PipelineUmmaAsync completion
 *   - TMEM -> register epilogue via SM100_TMEM_LOAD + make_tmem_copy
 *
 * Idioms follow CUTLASS example 77_blackwell_fmha (Apache-2.0); the
 * gemm_zero_acc helper comes from its collective/fmha_common.hpp, vendored
 * at csrc/sm100/fmha_common.hpp (the Modal image trims the examples tree).
 *
 * This TU is compiled ONLY for sm_100a into the separate extension module
 * flash_nystrom._C_sm100; it is never part of the multi-arch main module.
 ******************************************************************************/

#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/numeric_types.h>

#include "sm100/fmha_common.hpp"  // gemm_zero_acc (vendored from example 77, Apache-2.0)

namespace flash_nystrom_sm100 {

using namespace cute;
using cutlass::fmha::collective::gemm_zero_acc;

template <int kHeadDim_>
struct SmokeTraits {
    using Element = cutlass::half_t;
    using ElementAcc = float;

    // MMA orientation matches the planned sm100 kernels (and example 77's
    // backward): the STREAMED dimension is the MMA M (128 rows, full TMEM
    // lanes) and the landmark dimension (m <= 64) is the MMA N. An M=64
    // accumulator has a packed TMEM layout the 32dp TMEM_LOAD atoms cannot
    // tile, so S is computed transposed: S^T[128 x 64] = K_tile @ Qt^T.
    static constexpr int kTileM = 128;   // streamed K/V rows per tile (MMA M)
    static constexpr int kTileN = 64;    // landmark rows (MMA N)
    static constexpr int kHeadDim = kHeadDim_;

    using ClusterShape = Shape<_1, _1, _1>;
    // Row-major (K-contiguous) operand strides, batch mode unused here.
    using StrideK = Stride<int, _1, Stride<int, int>>;

    using CollectiveMma = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Element, StrideK, 8,
        Element, StrideK, 8,
        ElementAcc,
        Shape<Int<kTileM>, Int<kTileN>, Int<kHeadDim>>,
        ClusterShape, cutlass::gemm::collective::StageCount<2>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

    using TiledMma    = typename CollectiveMma::TiledMma;
    using SmemLayoutA = typename CollectiveMma::SmemLayoutA;  // (BLK_M, BLK_K, PIPE)
    using SmemLayoutB = typename CollectiveMma::SmemLayoutB;  // (BLK_N, BLK_K, PIPE)

    using TmemAllocator = cute::TMEM::Allocator1Sm;
    using PipelineS = cutlass::PipelineUmmaAsync<1>;

    // 8 warps: the TMEM->register epilogue tiles one 32-lane
    // SM100_TMEM_LOAD atom per (32-row x 32-col) patch of the
    // 64 x 128 fp32 accumulator -> 8 warp-atoms = 256 threads.
    static constexpr int kNumThreads = 256;

    struct SharedStorage {
        alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutA>> smem_a;
        alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutB>> smem_b;
        alignas(16) typename PipelineS::SharedStorage pipeline_s;
        uint32_t tmem_base_ptr;
    };
};

template <class Traits>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
sm100_smoke_kernel(
    const typename Traits::Element* __restrict__ a_ptr,  // (kTileM=128, D) streamed rows
    const typename Traits::Element* __restrict__ b_ptr,  // (kTileN=64, D) landmark rows
    float* __restrict__ c_ptr                            // (kTileM, kTileN) row-major
) {
    using Element = typename Traits::Element;
    constexpr int kM = Traits::kTileM;
    constexpr int kN = Traits::kTileN;
    constexpr int kK = Traits::kHeadDim;

    extern __shared__ char smem_raw[];
    auto& storage = *reinterpret_cast<typename Traits::SharedStorage*>(smem_raw);

    const int tidx = threadIdx.x;
    const int warp_idx = tidx / 32;

    // ---- gmem -> smem stage-0 fill (cute tutorial blackwell/01_mma_sm100
    // idiom): partition the gmem tensors with the MMA so they become
    // congruent with the MMA-canonical swizzled smem layouts, then let
    // cooperative_copy handle the swizzle. The real kernels will use TMA,
    // which consumes these layouts natively.
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()),
                            typename Traits::SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()),
                            typename Traits::SmemLayoutB{});
    Tensor mA = make_tensor(make_gmem_ptr(a_ptr),
                            make_shape(Int<kM>{}, Int<kK>{}),
                            make_stride(Int<kK>{}, _1{}));
    Tensor mB = make_tensor(make_gmem_ptr(b_ptr),
                            make_shape(Int<kN>{}, Int<kK>{}),
                            make_stride(Int<kK>{}, _1{}));
    {
        typename Traits::TiledMma fill_mma;
        auto cta_mma = fill_mma.get_slice(0);
        Tensor tCgA = cta_mma.partition_A(mA);   // (MmaA, NumMma_M, NumMma_K)
        Tensor tCgB = cta_mma.partition_B(mB);   // (MmaB, NumMma_N, NumMma_K)
        cooperative_copy<Traits::kNumThreads>(tidx, tCgA, sA(_, _, _, _0{}));
        cooperative_copy<Traits::kNumThreads>(tidx, tCgB, sB(_, _, _, _0{}));
    }
    __syncthreads();

    // ---- TMEM allocation (warp 0 allocates the full capacity; base is 0)
    typename Traits::TmemAllocator tmem_allocator;
    if (warp_idx == 0) {
        tmem_allocator.allocate(
            Traits::TmemAllocator::Sm100TmemCapacityColumns,
            &storage.tmem_base_ptr);
    }
    __syncthreads();

    // ---- UMMA completion pipeline: warp 0 produces, all threads consume
    typename Traits::PipelineS::Params pipeline_params;
    pipeline_params.role = (warp_idx == 0)
        ? Traits::PipelineS::ThreadCategory::ProducerConsumer
        : Traits::PipelineS::ThreadCategory::Consumer;
    pipeline_params.consumer_arv_count = Traits::kNumThreads;
    typename Traits::PipelineS pipeline_s(
        storage.pipeline_s, pipeline_params,
        typename Traits::ClusterShape{}, cute::true_type{}, cute::false_type{});
    typename Traits::PipelineS::PipelineState pipeline_s_producer_state =
        cutlass::make_producer_start_state<typename Traits::PipelineS>();
    typename Traits::PipelineS::PipelineState pipeline_s_consumer_state;
    __syncthreads();

    // ---- MMA: warp 0 issues S = A @ B^T into TMEM column 0
    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(0);
    Tensor tCtC = partition_fragment_C(
        tiled_mma, Shape<Int<kM>, Int<kN>>{});
    tCtC.data() = 0;  // full-capacity allocation => TMEM base is column 0

    if (warp_idx == 0) {
        Tensor tCrA = thr_mma.make_fragment_A(sA);  // (MMA, MMA_M, MMA_K, PIPE)
        Tensor tCrB = thr_mma.make_fragment_B(sB);
        pipeline_s.producer_acquire(pipeline_s_producer_state);
        gemm_zero_acc(tiled_mma,
                      tCrA(_, _, _, _0{}), tCrB(_, _, _, _0{}), tCtC);
        pipeline_s.producer_commit(pipeline_s_producer_state);
        ++pipeline_s_producer_state;
    }

    // ---- wait for the UMMA to land, then TMEM -> registers -> gmem
    pipeline_s.consumer_wait(pipeline_s_consumer_state);

    // Epilogue (tutorial 01 idiom): MMA-partition the gmem C tensor, then
    // partition both sides with the tmem tiled copy and store directly.
    using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b32x;
    Tensor mC  = make_tensor(make_gmem_ptr(c_ptr),
                             make_shape(Int<kM>{}, Int<kN>{}),
                             make_stride(Int<kN>{}, _1{}));
    Tensor tCgC = tiled_mma.get_slice(0).partition_C(mC);  // (MmaC,NumM,NumN)
    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tCtC);
    if (tidx < size(tiled_tmem_load)) {
        auto thr_tmem_load = tiled_tmem_load.get_slice(tidx);
        Tensor tTMtC = thr_tmem_load.partition_S(tCtC);
        Tensor tTMgC = thr_tmem_load.partition_D(tCgC);
        Tensor tTMrC = make_tensor<float>(shape(tTMgC));

        copy(tiled_tmem_load, tTMtC, tTMrC);
        cutlass::arch::fence_view_async_tmem_load();
        copy(tTMrC, tTMgC);
    }

    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;
    __syncthreads();

    // ---- TMEM deallocation (warp 0)
    if (warp_idx == 0) {
        tmem_allocator.free(storage.tmem_base_ptr,
                            Traits::TmemAllocator::Sm100TmemCapacityColumns);
    }
}

template <int kHeadDim>
torch::Tensor run_smoke(const torch::Tensor& a, const torch::Tensor& b) {
    using Traits = SmokeTraits<kHeadDim>;
    const at::cuda::CUDAGuard guard(a.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto c = torch::empty({Traits::kTileM, Traits::kTileN},
                          a.options().dtype(torch::kFloat32));

    constexpr size_t smem = sizeof(typename Traits::SharedStorage);
    auto* kernel = sm100_smoke_kernel<Traits>;
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem));
    }
    kernel<<<1, Traits::kNumThreads, smem, stream>>>(
        reinterpret_cast<const typename Traits::Element*>(a.data_ptr()),
        reinterpret_cast<const typename Traits::Element*>(b.data_ptr()),
        static_cast<float*>(c.data_ptr()));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "sm100 smoke launch failed: ", cudaGetErrorString(err));
    return c;
}

torch::Tensor smoke(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "smoke: tensors must be CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16,
                "smoke: tensors must be fp16");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "smoke: tensors must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "smoke: tensors must be 2D");
    TORCH_CHECK(a.size(0) == 128 && b.size(0) == 64,
                "smoke: A must be (128, D) and B (64, D)");
    TORCH_CHECK(a.size(1) == b.size(1), "smoke: K dims must match");
    const int64_t d = a.size(1);
    TORCH_CHECK(d == 64 || d == 128, "smoke: D must be 64 or 128");

    int major = 0, minor = 0, device = a.get_device();
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    TORCH_CHECK(major == 10,
                "sm100 smoke requires a Blackwell datacenter GPU (sm_100x), "
                "got sm_", major, minor);

    return (d == 64) ? run_smoke<64>(a, b) : run_smoke<128>(a, b);
}

bool available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
    int major = 0, device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    return major == 10;
}

}  // namespace flash_nystrom_sm100

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashNystrom Blackwell-native (sm_100a) kernels";
    m.def("smoke", &flash_nystrom_sm100::smoke,
          "tcgen05 GEMM smoke: S[128x64] = A[128xD] @ B[64xD]^T (fp16 -> fp32)",
          py::arg("a"), py::arg("b"));
    m.def("available", &flash_nystrom_sm100::available,
          "True when the current device is a Blackwell datacenter GPU");
}
