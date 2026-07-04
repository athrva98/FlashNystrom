/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 *
 * CP-B2: Blackwell-native (sm_100a) kernel3 backward.
 *
 * Same math as kernel3_bwd_tc (csrc/kernels/backward/kernel3_bwd.cuh), in
 * the transposed orientation (streamed K/V rows are the MMA M so every
 * accumulator is TMEM-tileable):
 *   St[Bc, m]  = K_tile @ Qt^T          (GEMM1, SS)
 *   Pt         = exp(St - lse3[col])    (compute, masked)
 *   dPt[Bc, m] = V_tile @ dO3^T         (GEMM2, SS)
 *   dSt        = Pt * (dPt - D3[col])   (compute)
 *   dV[Bc, D]  = Pt  @ dO3              (GEMM3, SS; A = Pt from smem)
 *   dK[Bc, D]  = dSt @ Qt               (GEMM4, SS)
 *   dQt[m, D] += dSt^T @ K_tile         (GEMM5, SS; M = 64, atomicAdd)
 *
 * v1 restrictions (dispatch falls back to the sm80 path otherwise):
 *   m == 64, D in {64, 128}, N % 128 == 0 (TMA in a later checkpoint gives
 *   partial tiles for free via OOB zero-fill).
 *
 * Idioms: cute tutorial blackwell/01 (staging, epilogue) and CUTLASS
 * example 77 backward (per-GEMM CollectiveBuilders, union'd transposed smem
 * views, position-independent-swizzle compute-warp smem store). All five
 * MMA shapes were validated standalone in sm100_smoke.cu.
 ******************************************************************************/
#pragma once

#include <algorithm>

#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/numeric_types.h>

#include "sm100/fmha_common.hpp"

namespace flash_nystrom_sm100 {

using namespace cute;
using cutlass::fmha::collective::gemm_zero_acc;
using cutlass::fmha::collective::gemm_reset_zero_acc;

template <int kHeadDim_>
struct K3BwdSm100Traits {
    using Element = cutlass::half_t;
    using ElementAcc = float;

    static constexpr int kTileK = 128;   // streamed K/V rows per CTA (MMA M)
    static constexpr int kTileM = 64;    // landmark rows (MMA N)
    static constexpr int kHeadDim = kHeadDim_;

    using ClusterShape = Shape<_1, _1, _1>;
    using StrideK  = Stride<int, _1, Stride<int, int>>;   // K-contiguous
    using StrideMN = Stride<_1, int, Stride<int, int>>;   // MN-contiguous
    using Schedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;

    // GEMM1/GEMM2 shape (128, 64, D): St = K @ Qt^T, dPt = V @ dO3^T.
    // SmemLayoutA is the writer layout for K and V, SmemLayoutB for Qt/dO3.
    using CollectiveKQ = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Element, StrideK, 8, Element, StrideK, 8, ElementAcc,
        Shape<Int<kTileK>, Int<kTileM>, Int<kHeadDim>>,
        ClusterShape, cutlass::gemm::collective::StageCount<2>,
        Schedule>::CollectiveOp;

    // GEMM3/GEMM4 shape (128, D, 64): dV = Pt @ dO3, dK = dSt @ Qt.
    // SmemLayoutA is the writer layout for Pt/dSt; SmemLayoutB is the
    // MN-major reader view of the K-major-written Qt/dO3 buffers.
    using CollectivePE = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Element, StrideK, 8, Element, StrideMN, 8, ElementAcc,
        Shape<Int<kTileK>, Int<kHeadDim>, Int<kTileM>>,
        ClusterShape, cutlass::gemm::collective::StageCount<2>,
        Schedule>::CollectiveOp;

    // GEMM5 shape (64, D, 128): dQt = dSt^T @ K. SmemLayoutA is the MN-major
    // reader view of dSt, SmemLayoutB the MN-major reader view of K.
    using CollectiveDQ = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Element, StrideMN, 8, Element, StrideMN, 8, ElementAcc,
        Shape<Int<kTileM>, Int<kHeadDim>, Int<kTileK>>,
        ClusterShape, cutlass::gemm::collective::StageCount<2>,
        Schedule>::CollectiveOp;

    // TMEM column map (fp32 accumulators, no overlaps needed):
    // St 64 + dPt 64 + dV D + dK D + dQt D = 128 + 3D <= 512 for D <= 128.
    static constexpr uint32_t kTmemSt  = 0;
    static constexpr uint32_t kTmemDPt = 64;
    static constexpr uint32_t kTmemDV  = 128;
    static constexpr uint32_t kTmemDK  = 128 + kHeadDim;
    static constexpr uint32_t kTmemDQ  = 128 + 2 * kHeadDim;
    static_assert(128 + 3 * kHeadDim <= 512, "TMEM budget exceeded");

    using TmemAllocator = cute::TMEM::Allocator1Sm;
    using PipelineS = cutlass::PipelineUmmaAsync<1>;
    static constexpr int kNumThreads = 256;

    // Restaged views of the builder smem layouts: K/V are double-buffered
    // for the TMA prefetch pipeline; everything else is single-stage (the
    // builder's default PIPE on all buffers would blow the smem budget).
    using SmemLayoutKV  = decltype(cutlass::fmha::collective::unstageSmemLayout(
        typename CollectiveKQ::SmemLayoutA{}, _2{}));
    using SmemLayoutQD  = decltype(cutlass::fmha::collective::unstageSmemLayout(
        typename CollectiveKQ::SmemLayoutB{}, _1{}));
    using SmemLayoutPDS = decltype(cutlass::fmha::collective::unstageSmemLayout(
        typename CollectivePE::SmemLayoutA{}, _1{}));
    using SmemLayoutQDt = decltype(cutlass::fmha::collective::unstageSmemLayout(
        typename CollectivePE::SmemLayoutB{}, _1{}));
    using SmemLayoutDS5 = decltype(cutlass::fmha::collective::unstageSmemLayout(
        typename CollectiveDQ::SmemLayoutA{}, _1{}));
    using SmemLayoutK5  = decltype(cutlass::fmha::collective::unstageSmemLayout(
        typename CollectiveDQ::SmemLayoutB{}, _2{}));  // reader view of smem_k

    struct SharedStorage {
        alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutKV>> smem_k;
        alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutKV>> smem_v;
        alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutQD>> smem_qt;
        alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutQD>> smem_do3;
        alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutPDS>> smem_p;
        alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutPDS>> smem_ds;
        // Plain row-major staging for the compute-warp P/dS stores. Compute
        // threads write by (row, col) coordinate (order-independent), then a
        // cooperative_copy re-lays them into the swizzled MMA layouts above
        // (the same partition idiom validated by the gmem fills).
        alignas(128) cute::array<Element, kTileK * kTileM> smem_stage_p;
        alignas(128) cute::array<Element, kTileK * kTileM> smem_stage_ds;
        alignas(16) typename PipelineS::SharedStorage pipeline_s;   // GEMM1+2
        alignas(16) typename PipelineS::SharedStorage pipeline_d;   // GEMM3-5
        alignas(16) cute::uint64_t tma_barrier_kv[2];  // one per K/V stage
        alignas(16) cute::uint64_t tma_barrier_qd;     // Qt + dO3 (once)
        uint32_t tmem_base_ptr;
    };

    // Transaction bytes: one K/V stage (K + V tile) and the one-time
    // Qt + dO3 load.
    static constexpr int kTmaBytesKV =
        2 * kTileK * kHeadDim * (int)sizeof(Element);
    static constexpr int kTmaBytesQD =
        2 * kTileM * kHeadDim * (int)sizeof(Element);
};

template <class Traits, class TmaKV, class TmaQD>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
kernel3_bwd_sm100_kernel(
    CUTE_GRID_CONSTANT TmaKV const tma_k,    // (BH*N, D) K_s
    CUTE_GRID_CONSTANT TmaKV const tma_v,    // (BH*N, D) V
    CUTE_GRID_CONSTANT TmaQD const tma_qt,   // (BH*64, D) Q_tilde
    CUTE_GRID_CONSTANT TmaQD const tma_do3,  // (BH*64, D) dO3
    const float*              __restrict__ lse3_ptr,   // (BH, 64)
    const float*              __restrict__ d3_ptr,     // (BH, 64)
    typename Traits::Element* __restrict__ dv_ptr,     // (BH, N, D)
    typename Traits::Element* __restrict__ dk_ptr,     // (BH, N, D)
    float*                    __restrict__ dqt_ptr,    // (BH, 64, D)
    int BH, int N
) {
    using Element = typename Traits::Element;
    constexpr int kBc = Traits::kTileK;    // 128
    constexpr int kM  = Traits::kTileM;    // 64
    constexpr int kD  = Traits::kHeadDim;

    const int bh = blockIdx.y;
    const int num_tiles = N / kBc;
    const int t0 = blockIdx.x;         // this CTA's first tile
    const int stride = gridDim.x;      // persistent stride over tiles
    if (t0 >= num_tiles) return;

    extern __shared__ char smem_raw[];
    auto& storage = *reinterpret_cast<typename Traits::SharedStorage*>(smem_raw);
    const int tidx = threadIdx.x;
    const int warp_idx = tidx / 32;

    // ---- TMA plumbing. Gmem tensors are flattened to (BH*rows, D); Qt and
    // dO3 load once per CTA (coordinate bh), K/V tiles stream through a
    // 2-stage smem pipeline with one transaction barrier per stage.
    if (tidx == 0) {
        cute::initialize_barrier(storage.tma_barrier_kv[0], /*num_threads=*/1);
        cute::initialize_barrier(storage.tma_barrier_kv[1], /*num_threads=*/1);
        cute::initialize_barrier(storage.tma_barrier_qd, /*num_threads=*/1);
    }
    __syncthreads();

    Tensor sK_all = make_tensor(make_smem_ptr(storage.smem_k.begin()),
                                typename Traits::SmemLayoutKV{});  // (.., 2)
    Tensor sV_all = make_tensor(make_smem_ptr(storage.smem_v.begin()),
                                typename Traits::SmemLayoutKV{});
    Tensor sQt  = make_tensor(make_smem_ptr(storage.smem_qt.begin()),
                              typename Traits::SmemLayoutQD{})(_, _, _, _0{});
    Tensor sDO3 = make_tensor(make_smem_ptr(storage.smem_do3.begin()),
                              typename Traits::SmemLayoutQD{})(_, _, _, _0{});

    Tensor mK_t   = tma_k.get_tma_tensor(make_shape(BH * N, Int<kD>{}));
    Tensor mV_t   = tma_v.get_tma_tensor(make_shape(BH * N, Int<kD>{}));
    Tensor mQt_t  = tma_qt.get_tma_tensor(make_shape(BH * kM, Int<kD>{}));
    Tensor mDO3_t = tma_do3.get_tma_tensor(make_shape(BH * kM, Int<kD>{}));
    // Keep the row-tile mode free (underscore) so the loop can index tiles.
    Tensor gK   = local_tile(mK_t,   Shape<Int<kBc>, Int<kD>>{},
                             make_coord(_, _0{}));   // (kBc, kD, row_tiles)
    Tensor gV   = local_tile(mV_t,   Shape<Int<kBc>, Int<kD>>{},
                             make_coord(_, _0{}));
    Tensor gQt  = local_tile(mQt_t,  Shape<Int<kM>, Int<kD>>{},
                             make_coord(bh, _0{}));
    Tensor gDO3 = local_tile(mDO3_t, Shape<Int<kM>, Int<kD>>{},
                             make_coord(bh, _0{}));
    auto cta_kq = typename Traits::CollectiveKQ::TiledMma{}.get_slice(0);
    Tensor tCgK   = cta_kq.partition_A(gK);    // (MmaA, M, K, row_tiles)
    Tensor tCgV   = cta_kq.partition_A(gV);
    Tensor tCgQt  = cta_kq.partition_B(gQt);
    Tensor tCgDO3 = cta_kq.partition_B(gDO3);
    // No structured bindings: they cannot be captured by the issue lambda
    // under C++17.
    auto pKK = tma_partition(tma_k, Int<0>{}, Layout<_1>{},
        group_modes<0, 3>(sK_all), group_modes<0, 3>(tCgK));
    auto pVV = tma_partition(tma_v, Int<0>{}, Layout<_1>{},
        group_modes<0, 3>(sV_all), group_modes<0, 3>(tCgV));
    auto pQQ = tma_partition(tma_qt, Int<0>{}, Layout<_1>{},
        group_modes<0, 3>(sQt), group_modes<0, 3>(tCgQt));
    auto pDD = tma_partition(tma_do3, Int<0>{}, Layout<_1>{},
        group_modes<0, 3>(sDO3), group_modes<0, 3>(tCgDO3));
    auto tKgK = get<0>(pKK); auto tKsK = get<1>(pKK);  // (TMA, tiles)/(TMA, 2)
    auto tVgV = get<0>(pVV); auto tVsV = get<1>(pVV);
    auto tQgQ = get<0>(pQQ); auto tQsQ = get<1>(pQQ);
    auto tDgD = get<0>(pDD); auto tDsD = get<1>(pDD);

    // Issue one K+V tile into smem stage `buf` (thread 0 only).
    auto issue_kv = [&](int t, int b) {
        cute::set_barrier_transaction_bytes(storage.tma_barrier_kv[b],
                                            Traits::kTmaBytesKV);
        const int row_tile = bh * num_tiles + t;
        copy(tma_k.with(storage.tma_barrier_kv[b]), tKgK(_, row_tile),
             tKsK(_, b));
        copy(tma_v.with(storage.tma_barrier_kv[b]), tVgV(_, row_tile),
             tVsV(_, b));
    };
    if (tidx == 0) {
        cute::set_barrier_transaction_bytes(storage.tma_barrier_qd,
                                            Traits::kTmaBytesQD);
        copy(tma_qt.with(storage.tma_barrier_qd),  tQgQ, tQsQ);
        copy(tma_do3.with(storage.tma_barrier_qd), tDgD, tDsD);
        issue_kv(t0, 0);
        if (t0 + stride < num_tiles) issue_kv(t0 + stride, 1);
    }
    cute::wait_barrier(storage.tma_barrier_qd, /*phase=*/0);

    // ---- TMEM allocation + the two UMMA completion pipelines
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
    typename Traits::PipelineS pipeline_d(
        storage.pipeline_d, pipeline_params,
        typename Traits::ClusterShape{}, cute::true_type{}, cute::false_type{});
    typename Traits::PipelineS::PipelineState producer_s =
        cutlass::make_producer_start_state<typename Traits::PipelineS>();
    typename Traits::PipelineS::PipelineState producer_d =
        cutlass::make_producer_start_state<typename Traits::PipelineS>();
    typename Traits::PipelineS::PipelineState consumer_s;
    typename Traits::PipelineS::PipelineState consumer_d;
    __syncthreads();

    // ---- accumulators
    typename Traits::CollectiveKQ::TiledMma mma_kq;
    typename Traits::CollectivePE::TiledMma mma_pe;
    typename Traits::CollectiveDQ::TiledMma mma_dq;
    Tensor tSt  = partition_fragment_C(mma_kq, Shape<Int<kBc>, Int<kM>>{});
    Tensor tDPt = partition_fragment_C(mma_kq, Shape<Int<kBc>, Int<kM>>{});
    Tensor tDV  = partition_fragment_C(mma_pe, Shape<Int<kBc>, Int<kD>>{});
    Tensor tDK  = partition_fragment_C(mma_pe, Shape<Int<kBc>, Int<kD>>{});
    Tensor tDQ  = partition_fragment_C(mma_dq, Shape<Int<kM>,  Int<kD>>{});
    tSt.data()  = Traits::kTmemSt;
    tDPt.data() = Traits::kTmemDPt;
    tDV.data()  = Traits::kTmemDV;
    tDK.data()  = Traits::kTmemDK;
    tDQ.data()  = Traits::kTmemDQ;

    // ---- warp0 MMA fragments (stage-sliced per iteration for K/V)
    auto thr_kq = mma_kq.get_slice(0);
    Tensor tArK   = thr_kq.make_fragment_A(sK_all);   // (MmaA, M, K, 2)
    Tensor tArV   = thr_kq.make_fragment_A(sV_all);
    Tensor tBrQt  = thr_kq.make_fragment_B(sQt);      // (MmaB, N, K)
    Tensor tBrDO3 = thr_kq.make_fragment_B(sDO3);

    Tensor sP  = make_tensor(make_smem_ptr(storage.smem_p.begin()),
                             typename Traits::SmemLayoutPDS{});
    Tensor sDS = make_tensor(make_smem_ptr(storage.smem_ds.begin()),
                             typename Traits::SmemLayoutPDS{});
    Tensor sDO3t = make_tensor(make_smem_ptr(storage.smem_do3.begin()),
                               typename Traits::SmemLayoutQDt{});
    Tensor sQtt  = make_tensor(make_smem_ptr(storage.smem_qt.begin()),
                               typename Traits::SmemLayoutQDt{});
    Tensor sDS5  = make_tensor(make_smem_ptr(storage.smem_ds.begin()),
                               typename Traits::SmemLayoutDS5{});
    Tensor sK5_all = make_tensor(make_smem_ptr(storage.smem_k.begin()),
                                 typename Traits::SmemLayoutK5{});  // (.., 2)
    auto thr_pe = mma_pe.get_slice(0);
    auto thr_dq = mma_dq.get_slice(0);
    Tensor tArP    = thr_pe.make_fragment_A(sP);
    Tensor tArDS   = thr_pe.make_fragment_A(sDS);
    Tensor tBrDO3t = thr_pe.make_fragment_B(sDO3t);
    Tensor tBrQtt  = thr_pe.make_fragment_B(sQtt);
    Tensor tArDSt  = thr_dq.make_fragment_A(sDS5);
    Tensor tBrK5   = thr_dq.make_fragment_B(sK5_all);

    // dV/dK epilogue for one tile: TMEM -> registers -> gmem (direct store;
    // the tensors are zero-initialized and each tile row has one writer).
    auto store_dv_dk = [&](int tile) {
        Tensor mDV = make_tensor(
            make_gmem_ptr(dv_ptr + (int64_t)bh*N*kD + (int64_t)tile*kBc*kD),
            make_shape(Int<kBc>{}, Int<kD>{}), make_stride(Int<kD>{}, _1{}));
        Tensor mDK = make_tensor(
            make_gmem_ptr(dk_ptr + (int64_t)bh*N*kD + (int64_t)tile*kBc*kD),
            make_shape(Int<kBc>{}, Int<kD>{}), make_stride(Int<kD>{}, _1{}));
        Tensor tCgDV = thr_pe.partition_C(mDV);
        Tensor tCgDK = thr_pe.partition_C(mDK);
        auto tld = make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{}, tDV);
        if (tidx < size(tld)) {
            auto thr_ld = tld.get_slice(tidx);
            Tensor tTMgDV = thr_ld.partition_D(tCgDV);
            Tensor tTMgDK = thr_ld.partition_D(tCgDK);
            Tensor rDV = make_tensor<float>(shape(tTMgDV));
            Tensor rDK = make_tensor<float>(shape(tTMgDK));
            copy(tld, thr_ld.partition_S(tDV), rDV);
            copy(tld, thr_ld.partition_S(tDK), rDK);
            cutlass::arch::fence_view_async_tmem_load();
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rDV); i++) {
                tTMgDV(i) = Element(rDV(i));
                tTMgDK(i) = Element(rDK(i));
            }
        }
    };

    // ---- persistent, software-pipelined loop over this CTA's tiles.
    // Each tile's dV/dK epilogue is deferred one iteration so the tensor
    // core runs GEMM3-5(prev) and GEMM1+2(cur) back to back while the
    // threads do the deferred stores and the softmax-backward compute.
    int phase_kv[2] = {0, 0};
    int buf = 0;
    int prev_t = -1;   // tile with a pending dV/dK epilogue
    for (int t = t0; t < num_tiles; t += stride, buf ^= 1) {
        cute::wait_barrier(storage.tma_barrier_kv[buf], phase_kv[buf]);
        phase_kv[buf] ^= 1;

        // GEMM1 + GEMM2: St = K @ Qt^T, dPt = V @ dO3^T
        if (warp_idx == 0) {
            pipeline_s.producer_acquire(producer_s);
            gemm_zero_acc(mma_kq, tArK(_, _, _, buf), tBrQt, tSt);
            gemm_zero_acc(mma_kq, tArV(_, _, _, buf), tBrDO3, tDPt);
            pipeline_s.producer_commit(producer_s);
            ++producer_s;
        }

        // Deferred epilogue for the previous tile; GEMM1+2(t) run under it.
        // The pipe_d wait also frees K[buf^1], so the next tile for that
        // buffer prefetches here.
        if (prev_t >= 0) {
            pipeline_d.consumer_wait(consumer_d);
            if (tidx == 0 && t + stride < num_tiles) {
                issue_kv(t + stride, buf ^ 1);
            }
            store_dv_dk(prev_t);
            pipeline_d.consumer_release(consumer_d);
            ++consumer_d;
        }

        pipeline_s.consumer_wait(consumer_s);

        // compute Pt = exp(St - lse3[col]), dSt = Pt * (dPt - D3[col]) into
        // plain staging by (row, col), then re-lay into the swizzled MMA
        // layouts (order-independent; the compose idiom mis-ordered values).
        {
            auto load_op = SM100_TMEM_LOAD_32dp32b16x{};
            Tensor cSt = make_identity_tensor(Shape<Int<kBc>, Int<kM>>{});
            Tensor tCcSt = thr_kq.partition_C(cSt);
            auto tiled_t2r = make_tmem_copy(load_op, tSt);
            if (tidx < size(tiled_t2r)) {
                auto thr_t2r = tiled_t2r.get_slice(tidx % 128);
                Tensor tTRcSt = thr_t2r.partition_D(tCcSt);
                Tensor tTRrSt  = make_tensor<float>(shape(tTRcSt));
                Tensor tTRrDPt = make_tensor<float>(shape(tTRcSt));
                copy(tiled_t2r, thr_t2r.partition_S(tSt),  tTRrSt);
                copy(tiled_t2r, thr_t2r.partition_S(tDPt), tTRrDPt);
                cutlass::arch::fence_view_async_tmem_load();

                const float* lse = lse3_ptr + bh * kM;
                const float* d3  = d3_ptr  + bh * kM;
                Element* stage_p  = storage.smem_stage_p.begin();
                Element* stage_ds = storage.smem_stage_ds.begin();
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(tTRrSt); i++) {
                    const int row = get<0>(tTRcSt(i));   // streamed K/V row
                    const int col = get<1>(tTRcSt(i));   // landmark index
                    const float p  = expf(tTRrSt(i) - lse[col]);
                    const float ds = p * (tTRrDPt(i) - d3[col]);
                    stage_p[row * kM + col]  = Element(p);
                    stage_ds[row * kM + col] = Element(ds);
                }
            }
        }
        __syncthreads();  // staging complete

        {
            Tensor mPst  = make_tensor(make_smem_ptr(storage.smem_stage_p.begin()),
                                       make_shape(Int<kBc>{}, Int<kM>{}),
                                       make_stride(Int<kM>{}, _1{}));
            Tensor mDSst = make_tensor(make_smem_ptr(storage.smem_stage_ds.begin()),
                                       make_shape(Int<kBc>{}, Int<kM>{}),
                                       make_stride(Int<kM>{}, _1{}));
            cooperative_copy<Traits::kNumThreads>(tidx, thr_pe.partition_A(mPst),
                                                  sP(_, _, _, _0{}));
            cooperative_copy<Traits::kNumThreads>(tidx, thr_pe.partition_A(mDSst),
                                                  sDS(_, _, _, _0{}));
        }
        cutlass::arch::fence_view_async_shared();
        __syncthreads();

        pipeline_s.consumer_release(consumer_s);
        ++consumer_s;

        // GEMM3-5: dV = Pt @ dO3, dK = dSt @ Qt, dQt += dSt^T @ K.
        // dQt accumulates in TMEM across the whole tile loop. The dV/dK
        // epilogue is deferred to the top of the next iteration.
        if (warp_idx == 0) {
            pipeline_d.producer_acquire(producer_d);
            gemm_zero_acc(mma_pe, tArP(_, _, _, _0{}),  tBrDO3t(_, _, _, _0{}), tDV);
            gemm_zero_acc(mma_pe, tArDS(_, _, _, _0{}), tBrQtt(_, _, _, _0{}),  tDK);
            if (prev_t < 0) {
                gemm_zero_acc(mma_dq, tArDSt(_, _, _, _0{}),
                              tBrK5(_, _, _, buf), tDQ);
            } else {
                mma_dq.accumulate_ = UMMA::ScaleOut::One;
                gemm_reset_zero_acc(mma_dq, tArDSt(_, _, _, _0{}),
                                    tBrK5(_, _, _, buf), tDQ);
            }
            pipeline_d.producer_commit(producer_d);
            ++producer_d;
        }
        prev_t = t;
    }

    // Final deferred dV/dK epilogue
    pipeline_d.consumer_wait(consumer_d);
    store_dv_dk(prev_t);
    pipeline_d.consumer_release(consumer_d);
    ++consumer_d;

    // dQt: accumulated over all of this CTA's tiles; one atomicAdd pass
    // (gridDim.x CTAs contribute per bh)
    {
        Tensor cDQ = make_identity_tensor(Shape<Int<kM>, Int<kD>>{});
        Tensor tCcDQ = mma_dq.get_slice(0).partition_C(cDQ);
        auto tld = make_tmem_copy(SM100_TMEM_LOAD_16dp32b32x{}, tDQ);
        if (tidx < size(tld)) {
            auto thr_ld = tld.get_slice(tidx);
            Tensor tTMcDQ = thr_ld.partition_D(tCcDQ);
            Tensor rDQ = make_tensor<float>(shape(tTMcDQ));
            copy(tld, thr_ld.partition_S(tDQ), rDQ);
            cutlass::arch::fence_view_async_tmem_load();
            float* dqt = dqt_ptr + (int64_t)bh * kM * kD;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rDQ); i++) {
                const int r = get<0>(tTMcDQ(i));
                const int c = get<1>(tTMcDQ(i));
                atomicAdd(&dqt[r * kD + c], rDQ(i));
            }
        }
    }

    __syncthreads();  // all threads done reading TMEM

    if (warp_idx == 0) {
        tmem_allocator.free(storage.tmem_base_ptr,
                            Traits::TmemAllocator::Sm100TmemCapacityColumns);
    }
}

// Raw-pointer launcher shared by the pybind test entry and the main-module
// hook. Returns false (without launching) for shapes v1 cannot handle so the
// caller falls back to the sm80 path. dV/dK are fully overwritten for every
// covered row; dQ_tilde is accumulated with atomicAdd (caller zero-inits).
template <int kHeadDim>
inline void launch_kernel3_bwd_sm100_impl(
    const void* q_tilde, const void* k_s, const void* v,
    const float* lse3, const float* d3, const void* do3,
    void* dV, void* dK_s, float* dQ_tilde,
    int BH, int N, cudaStream_t stream
) {
    using Traits = K3BwdSm100Traits<kHeadDim>;
    using Element = typename Traits::Element;
    constexpr int kBc = Traits::kTileK;
    constexpr int kM  = Traits::kTileM;

    // Host-side TMA descriptors (per-launch; encode the tensor pointers).
    auto layout_kv = make_layout(make_shape(BH * N, Int<kHeadDim>{}),
                                 make_stride(Int<kHeadDim>{}, _1{}));
    auto layout_qd = make_layout(make_shape(BH * kM, Int<kHeadDim>{}),
                                 make_stride(Int<kHeadDim>{}, _1{}));
    Tensor mK   = make_tensor(make_gmem_ptr(static_cast<const Element*>(k_s)), layout_kv);
    Tensor mV   = make_tensor(make_gmem_ptr(static_cast<const Element*>(v)), layout_kv);
    Tensor mQt  = make_tensor(make_gmem_ptr(static_cast<const Element*>(q_tilde)), layout_qd);
    Tensor mDO3 = make_tensor(make_gmem_ptr(static_cast<const Element*>(do3)), layout_qd);
    auto smem_kv = typename Traits::SmemLayoutKV{}(_, _, _, _0{});
    auto smem_qd = typename Traits::SmemLayoutQD{}(_, _, _, _0{});
    auto tma_k   = make_tma_atom(SM90_TMA_LOAD{}, mK, smem_kv,
                                 Shape<Int<kBc>, Int<kHeadDim>>{});
    auto tma_v   = make_tma_atom(SM90_TMA_LOAD{}, mV, smem_kv,
                                 Shape<Int<kBc>, Int<kHeadDim>>{});
    auto tma_qt  = make_tma_atom(SM90_TMA_LOAD{}, mQt, smem_qd,
                                 Shape<Int<kM>, Int<kHeadDim>>{});
    auto tma_do3 = make_tma_atom(SM90_TMA_LOAD{}, mDO3, smem_qd,
                                 Shape<Int<kM>, Int<kHeadDim>>{});
    using TmaKV = decltype(tma_k);
    using TmaQD = decltype(tma_qt);

    constexpr size_t smem = sizeof(typename Traits::SharedStorage);
    auto* kernel = kernel3_bwd_sm100_kernel<Traits, TmaKV, TmaQD>;
    static const bool smem_set = [kernel] {
        if (smem > 48 * 1024) {
            cudaFuncSetAttribute(kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem));
        }
        return true;
    }();
    (void)smem_set;
    // Persistent grid: each CTA strides through its share of the row tiles
    // so Qt/dO3 loads and the dQ_tilde accumulation amortize across tiles.
    // Aim for ~8 waves of CTAs: at 1 resident CTA/SM the wave-quantization
    // tail costs ~1/waves, while chains of even a few tiles capture most of
    // the amortization (2 waves measured 33% slower than the single-tile
    // grid; the tail dominated the reuse win).
    static const int sm_count = [] {
        int dev = 0, n = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n > 0 ? n : 1;
    }();
    const int num_tiles = N / kBc;
    const int chains = std::min(num_tiles,
                                std::max(1, (8 * sm_count + BH - 1) / BH));
    dim3 grid((unsigned)chains, (unsigned)BH);
    kernel<<<grid, Traits::kNumThreads, smem, stream>>>(
        tma_k, tma_v, tma_qt, tma_do3,
        lse3, d3,
        static_cast<Element*>(dV),
        static_cast<Element*>(dK_s),
        dQ_tilde, BH, N);
}

inline bool kernel3_bwd_sm100_try_launch(
    const void* q_tilde, const void* k_s, const void* v,
    const float* lse3, const float* d3, const void* do3,
    void* dV, void* dK_s, float* dQ_tilde,
    int BH, int N, int D, int m, cudaStream_t stream
) {
    if (m != 64 || N % 128 != 0 || !(D == 64 || D == 128)) return false;
    if (D == 64) {
        launch_kernel3_bwd_sm100_impl<64>(q_tilde, k_s, v, lse3, d3, do3,
                                          dV, dK_s, dQ_tilde, BH, N, stream);
    } else {
        launch_kernel3_bwd_sm100_impl<128>(q_tilde, k_s, v, lse3, d3, do3,
                                           dV, dK_s, dQ_tilde, BH, N, stream);
    }
    return true;
}

}  // namespace flash_nystrom_sm100
