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

// ---------------------------------------------------------------------------
// CP-B2 shape smoke: validates the two mechanisms kernel3_bwd_sm100 adds on
// top of the basic smoke, exactly as example 77's backward composes them:
//   1. SS GEMMs whose B (and A) operands are MN-major reads of buffers that
//      were WRITTEN through a K-major layout from a different collective
//      (the union'd transposed-view trick).
//   2. An M=64 accumulator (the dQ_tilde GEMM) with a 16dp TMEM_LOAD.
// GEMM2: C2[128, D] = P[128 x 64] @ E[64 x D]     (dV/dK shape)
// GEMM3: C3[ 64, D] = P^T        @ K[128 x D]     (dQt shape)
// P is written once through GEMM2's K-major A layout and read again through
// GEMM3's MN-major A layout; E and K are written through the (128, 64, D)
// collective's K-major layouts and read through MN-major B layouts.
template <int kHeadDim_>
struct BwdShapeTraits {
    using Element = cutlass::half_t;
    using ElementAcc = float;
    static constexpr int kHeadDim = kHeadDim_;

    using ClusterShape = Shape<_1, _1, _1>;
    using StrideK  = Stride<int, _1, Stride<int, int>>;   // K-contiguous
    using StrideMN = Stride<_1, int, Stride<int, int>>;   // MN-contiguous

    // Writer collective: same shape as the basic smoke (128, 64, D).
    // Its SmemLayoutA/B are the K-major layouts K and E are staged through.
    using CollectiveKQ = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Element, StrideK, 8, Element, StrideK, 8, ElementAcc,
        Shape<_128, _64, Int<kHeadDim>>,
        ClusterShape, cutlass::gemm::collective::StageCount<2>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

    // GEMM2 (dV/dK shape): (M, N, K) = (128, D, 64); A K-major, B MN-major.
    using CollectivePE = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Element, StrideK, 8, Element, StrideMN, 8, ElementAcc,
        Shape<_128, Int<kHeadDim>, _64>,
        ClusterShape, cutlass::gemm::collective::StageCount<2>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

    // GEMM3 (dQt shape): (M, N, K) = (64, D, 128); A and B MN-major.
    using CollectiveDQ = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Element, StrideMN, 8, Element, StrideMN, 8, ElementAcc,
        Shape<_64, Int<kHeadDim>, _128>,
        ClusterShape, cutlass::gemm::collective::StageCount<2>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

    using TmemAllocator = cute::TMEM::Allocator1Sm;
    using PipelineS = cutlass::PipelineUmmaAsync<1>;
    static constexpr int kNumThreads = 256;

    struct SharedStorage {
        // P through CollectivePE::SmemLayoutA (K-major (128, 64)); GEMM3
        // re-reads it through CollectiveDQ::SmemLayoutA ((64, 128) MN).
        alignas(2048) cute::array<Element,
            cute::cosize_v<typename CollectivePE::SmemLayoutA>> smem_p;
        // E through CollectiveKQ::SmemLayoutB (K-major (64, D)); GEMM2
        // reads it through CollectivePE::SmemLayoutB ((D, 64) MN).
        alignas(2048) cute::array<Element,
            cute::cosize_v<typename CollectiveKQ::SmemLayoutB>> smem_e;
        // K through CollectiveKQ::SmemLayoutA (K-major (128, D)); GEMM3
        // reads it through CollectiveDQ::SmemLayoutB ((D, 128) MN).
        alignas(2048) cute::array<Element,
            cute::cosize_v<typename CollectiveKQ::SmemLayoutA>> smem_k;
        alignas(16) typename PipelineS::SharedStorage pipeline_s;
        uint32_t tmem_base_ptr;
    };
};

template <class Traits>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
sm100_bwd_shapes_kernel(
    const typename Traits::Element* __restrict__ p_ptr,  // (128, 64)
    const typename Traits::Element* __restrict__ e_ptr,  // (64, D)
    const typename Traits::Element* __restrict__ k_ptr,  // (128, D)
    float* __restrict__ c2_ptr,                          // (128, D)
    float* __restrict__ c3_ptr                           // (64, D)
) {
    using Element = typename Traits::Element;
    constexpr int kD = Traits::kHeadDim;

    extern __shared__ char smem_raw[];
    auto& storage = *reinterpret_cast<typename Traits::SharedStorage*>(smem_raw);
    const int tidx = threadIdx.x;
    const int warp_idx = tidx / 32;

    // ---- stage inputs through the WRITER layouts (K-major)
    Tensor mP = make_tensor(make_gmem_ptr(p_ptr),
                            make_shape(_128{}, _64{}), make_stride(_64{}, _1{}));
    Tensor mE = make_tensor(make_gmem_ptr(e_ptr),
                            make_shape(_64{}, Int<kD>{}), make_stride(Int<kD>{}, _1{}));
    Tensor mK = make_tensor(make_gmem_ptr(k_ptr),
                            make_shape(_128{}, Int<kD>{}), make_stride(Int<kD>{}, _1{}));
    {
        Tensor sP = make_tensor(make_smem_ptr(storage.smem_p.begin()),
                                typename Traits::CollectivePE::SmemLayoutA{});
        Tensor sE = make_tensor(make_smem_ptr(storage.smem_e.begin()),
                                typename Traits::CollectiveKQ::SmemLayoutB{});
        Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.begin()),
                                typename Traits::CollectiveKQ::SmemLayoutA{});
        auto mma_pe = typename Traits::CollectivePE::TiledMma{}.get_slice(0);
        auto mma_kq = typename Traits::CollectiveKQ::TiledMma{}.get_slice(0);
        cooperative_copy<Traits::kNumThreads>(tidx, mma_pe.partition_A(mP),
                                              sP(_, _, _, _0{}));
        cooperative_copy<Traits::kNumThreads>(tidx, mma_kq.partition_B(mE),
                                              sE(_, _, _, _0{}));
        cooperative_copy<Traits::kNumThreads>(tidx, mma_kq.partition_A(mK),
                                              sK(_, _, _, _0{}));
    }
    __syncthreads();

    // ---- TMEM + pipeline (same scheme as the basic smoke)
    typename Traits::TmemAllocator tmem_allocator;
    if (warp_idx == 0) {
        tmem_allocator.allocate(
            Traits::TmemAllocator::Sm100TmemCapacityColumns,
            &storage.tmem_base_ptr);
    }
    __syncthreads();

    typename Traits::PipelineS::Params pipeline_params;
    pipeline_params.role = (warp_idx == 0)
        ? Traits::PipelineS::ThreadCategory::ProducerConsumer
        : Traits::PipelineS::ThreadCategory::Consumer;
    pipeline_params.consumer_arv_count = Traits::kNumThreads;
    typename Traits::PipelineS pipeline_s(
        storage.pipeline_s, pipeline_params,
        typename Traits::ClusterShape{}, cute::true_type{}, cute::false_type{});
    typename Traits::PipelineS::PipelineState producer_state =
        cutlass::make_producer_start_state<typename Traits::PipelineS>();
    typename Traits::PipelineS::PipelineState consumer_state;
    __syncthreads();

    // ---- accumulators: C2 at TMEM column 0, C3 at column 256
    typename Traits::CollectivePE::TiledMma mma2;
    typename Traits::CollectiveDQ::TiledMma mma3;
    Tensor tC2 = partition_fragment_C(mma2, Shape<_128, Int<kD>>{});
    Tensor tC3 = partition_fragment_C(mma3, Shape<_64, Int<kD>>{});
    tC2.data() = 0;
    tC3.data() = 256;

    if (warp_idx == 0) {
        // GEMM2: reader views (A K-major as written, B = MN view of E)
        Tensor sP2 = make_tensor(make_smem_ptr(storage.smem_p.begin()),
                                 typename Traits::CollectivePE::SmemLayoutA{});
        Tensor sE2 = make_tensor(make_smem_ptr(storage.smem_e.begin()),
                                 typename Traits::CollectivePE::SmemLayoutB{});
        // GEMM3: MN views of the same physical P and K bytes
        Tensor sP3 = make_tensor(make_smem_ptr(storage.smem_p.begin()),
                                 typename Traits::CollectiveDQ::SmemLayoutA{});
        Tensor sK3 = make_tensor(make_smem_ptr(storage.smem_k.begin()),
                                 typename Traits::CollectiveDQ::SmemLayoutB{});
        auto thr2 = mma2.get_slice(0);
        auto thr3 = mma3.get_slice(0);
        Tensor tArP = thr2.make_fragment_A(sP2);
        Tensor tBrE = thr2.make_fragment_B(sE2);
        Tensor tArPT = thr3.make_fragment_A(sP3);
        Tensor tBrK = thr3.make_fragment_B(sK3);

        pipeline_s.producer_acquire(producer_state);
        gemm_zero_acc(mma2, tArP(_, _, _, _0{}), tBrE(_, _, _, _0{}), tC2);
        gemm_zero_acc(mma3, tArPT(_, _, _, _0{}), tBrK(_, _, _, _0{}), tC3);
        pipeline_s.producer_commit(producer_state);
        ++producer_state;
    }

    pipeline_s.consumer_wait(consumer_state);

    // ---- epilogue C2: M=128 accumulator, 32dp load (as in the basic smoke)
    {
        Tensor mC2 = make_tensor(make_gmem_ptr(c2_ptr),
                                 make_shape(_128{}, Int<kD>{}),
                                 make_stride(Int<kD>{}, _1{}));
        Tensor tCgC2 = mma2.get_slice(0).partition_C(mC2);
        auto tld = make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{}, tC2);
        if (tidx < size(tld)) {
            auto thr_ld = tld.get_slice(tidx);
            Tensor tTMtC = thr_ld.partition_S(tC2);
            Tensor tTMgC = thr_ld.partition_D(tCgC2);
            Tensor tTMrC = make_tensor<float>(shape(tTMgC));
            copy(tld, tTMtC, tTMrC);
            cutlass::arch::fence_view_async_tmem_load();
            copy(tTMrC, tTMgC);
        }
    }
    // ---- epilogue C3: M=64 accumulator, 16dp load
    {
        Tensor mC3 = make_tensor(make_gmem_ptr(c3_ptr),
                                 make_shape(_64{}, Int<kD>{}),
                                 make_stride(Int<kD>{}, _1{}));
        Tensor tCgC3 = mma3.get_slice(0).partition_C(mC3);
        auto tld = make_tmem_copy(SM100_TMEM_LOAD_16dp32b32x{}, tC3);
        if (tidx < size(tld)) {
            auto thr_ld = tld.get_slice(tidx);
            Tensor tTMtC = thr_ld.partition_S(tC3);
            Tensor tTMgC = thr_ld.partition_D(tCgC3);
            Tensor tTMrC = make_tensor<float>(shape(tTMgC));
            copy(tld, tTMtC, tTMrC);
            cutlass::arch::fence_view_async_tmem_load();
            copy(tTMrC, tTMgC);
        }
    }

    pipeline_s.consumer_release(consumer_state);
    ++consumer_state;
    __syncthreads();

    if (warp_idx == 0) {
        tmem_allocator.free(storage.tmem_base_ptr,
                            Traits::TmemAllocator::Sm100TmemCapacityColumns);
    }
}

template <int kHeadDim>
std::vector<torch::Tensor> run_bwd_shapes(
    const torch::Tensor& p, const torch::Tensor& e, const torch::Tensor& k) {
    using Traits = BwdShapeTraits<kHeadDim>;
    const at::cuda::CUDAGuard guard(p.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto c2 = torch::empty({128, kHeadDim}, p.options().dtype(torch::kFloat32));
    auto c3 = torch::empty({64,  kHeadDim}, p.options().dtype(torch::kFloat32));

    constexpr size_t smem = sizeof(typename Traits::SharedStorage);
    auto* kernel = sm100_bwd_shapes_kernel<Traits>;
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem));
    }
    kernel<<<1, Traits::kNumThreads, smem, stream>>>(
        reinterpret_cast<const typename Traits::Element*>(p.data_ptr()),
        reinterpret_cast<const typename Traits::Element*>(e.data_ptr()),
        reinterpret_cast<const typename Traits::Element*>(k.data_ptr()),
        static_cast<float*>(c2.data_ptr()),
        static_cast<float*>(c3.data_ptr()));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "sm100 bwd-shapes launch failed: ", cudaGetErrorString(err));
    return {c2, c3};
}

std::vector<torch::Tensor> smoke_bwd_shapes(
    const torch::Tensor& p, const torch::Tensor& e, const torch::Tensor& k) {
    TORCH_CHECK(p.is_cuda() && e.is_cuda() && k.is_cuda(),
                "smoke_bwd_shapes: tensors must be CUDA");
    TORCH_CHECK(p.dtype() == torch::kFloat16 && e.dtype() == torch::kFloat16
                && k.dtype() == torch::kFloat16,
                "smoke_bwd_shapes: tensors must be fp16");
    TORCH_CHECK(p.is_contiguous() && e.is_contiguous() && k.is_contiguous(),
                "smoke_bwd_shapes: tensors must be contiguous");
    TORCH_CHECK(p.dim() == 2 && p.size(0) == 128 && p.size(1) == 64,
                "smoke_bwd_shapes: P must be (128, 64)");
    const int64_t d = e.size(1);
    TORCH_CHECK(e.dim() == 2 && e.size(0) == 64 && (d == 64 || d == 128),
                "smoke_bwd_shapes: E must be (64, D), D in {64, 128}");
    TORCH_CHECK(k.dim() == 2 && k.size(0) == 128 && k.size(1) == d,
                "smoke_bwd_shapes: K must be (128, D)");

    int major = 0, device = p.get_device();
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    TORCH_CHECK(major == 10, "sm100 bwd-shapes requires sm_100x");

    return (d == 64) ? run_bwd_shapes<64>(p, e, k)
                     : run_bwd_shapes<128>(p, e, k);
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
    m.def("smoke_bwd_shapes", &flash_nystrom_sm100::smoke_bwd_shapes,
          "CP-B2 shape smoke: C2[128xD] = P @ E and C3[64xD] = P^T @ K via "
          "MN-major union views + M=64 TMEM_LOAD",
          py::arg("p"), py::arg("e"), py::arg("k"));
    m.def("available", &flash_nystrom_sm100::available,
          "True when the current device is a Blackwell datacenter GPU");
}
