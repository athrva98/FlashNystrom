/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
// Runtime occupancy probe. For each kernel template instantiation, queries
// cudaOccupancyMaxActiveBlocksPerMultiprocessor at the actual launch config
// (threads, dynamic SMEM) and reports the result.
//
// Why this exists: cuobjdump --dump-resource-usage gives static info but does
// not account for register allocation granularity (regs are rounded up to a
// chunk size that varies by arch) or for the runtime opt-in to large dynamic
// SMEM. The CUDA runtime API has the exact answer the hardware would actually
// give. Use this for any decision about block count or kernel restructuring.
#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstdint>

#include "flash_nystrom.h"
#include "static_switch.h"
#include "kernels/landmark.cuh"
#include "kernels/kernel2_inv.cuh"
#include "kernels/kernel3_output_fused.cuh"
#include "kernels/kernel1_output_fused.cuh"
#include "kernels/backward/precompute_di.cuh"
#include "kernels/backward/kernel1_bwd.cuh"
#include "kernels/backward/kernel3_bwd.cuh"
#include "kernels/backward/compute_dO3.cuh"
#include "kernels/backward/compute_dk2inv.cuh"
#include "kernels/backward/kernel2_inv_bwd.cuh"
#include "kernels/backward/landmark_bwd.cuh"

#include <cutlass/numeric_types.h>

namespace flash_nystrom {

struct OccupancyRow {
    std::string kernel_name;
    int threads_per_block;
    int dynamic_smem_bytes;
    int regs_per_thread;          // from cudaFuncAttributes
    int static_smem_bytes;        // per CTA
    int max_blocks_per_sm;        // runtime answer
    int max_warps_per_sm;
    int regs_per_block;
    int total_smem_per_block;
    // Constraint diagnostics. The runtime returns the min, but we also
    // compute each axis independently so we can flag the binding one.
    int blocks_by_threads;        // hardware max threads per SM / threads
    int blocks_by_regs;           // register file / regs_per_block
    int blocks_by_smem;            // shared mem per SM / smem_per_block
    int blocks_by_hardware;        // hardware max blocks per SM
    std::string binding_constraint;
};

// SM12.0 (Blackwell consumer) limits. Source: NVIDIA arch table for compute
// capability 12.0. If running on a different arch, the runtime API still
// returns the correct max_blocks_per_sm; the per-axis breakdown is for
// SM12.0.
struct SmLimits {
    int max_threads_per_sm;
    int max_blocks_per_sm;
    int max_regs_per_sm;
    int max_smem_per_sm_bytes;
    int reg_alloc_unit;       // register allocation granularity (chunk size)
    int warp_alloc_unit;      // warp allocation granularity
};

inline SmLimits query_sm_limits() {
    SmLimits lim{};
    cudaDeviceProp prop{};
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    lim.max_threads_per_sm  = prop.maxThreadsPerMultiProcessor;
    lim.max_blocks_per_sm   = prop.maxBlocksPerMultiProcessor;
    lim.max_regs_per_sm     = prop.regsPerMultiprocessor;
    lim.max_smem_per_sm_bytes = static_cast<int>(prop.sharedMemPerMultiprocessor);
    // Architecture-dependent. SM 8.x and newer use 256-register chunks and
    // 4-warp granularity. Defaulting to those is correct for everything from
    // Ampere through Blackwell consumer.
    lim.reg_alloc_unit = 256;
    lim.warp_alloc_unit = 4;
    return lim;
}

template <typename KernelFn>
inline OccupancyRow probe(
    const std::string& name, KernelFn fn,
    int threads_per_block, int dynamic_smem_bytes,
    const SmLimits& lim
) {
    OccupancyRow r;
    r.kernel_name = name;
    r.threads_per_block = threads_per_block;
    r.dynamic_smem_bytes = dynamic_smem_bytes;

    // Per-kernel resource attributes. cudaFuncAttributes is exact (matches
    // cuobjdump REG and SHARED, but also includes spill stores/loads).
    cudaFuncAttributes attr{};
    cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(fn));
    r.regs_per_thread = attr.numRegs;
    r.static_smem_bytes = static_cast<int>(attr.sharedSizeBytes);

    // For the SMEM opt-in path: tell the runtime our intended dynamic SMEM
    // size. Without this, cudaOccupancyMaxActiveBlocksPerMultiprocessor
    // assumes the default 48 KB cap and undercounts our blocks/SM.
    if (dynamic_smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(fn),
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_bytes);
    }

    // Runtime occupancy answer. This is what the hardware would actually
    // schedule. Already accounts for register granularity.
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, reinterpret_cast<const void*>(fn),
        threads_per_block, dynamic_smem_bytes);
    r.max_blocks_per_sm = blocks;
    r.max_warps_per_sm = blocks * (threads_per_block / 32);

    // Per-axis breakdown, useful for identifying the binding constraint.
    r.regs_per_block = r.regs_per_thread * threads_per_block;
    r.total_smem_per_block = r.static_smem_bytes + dynamic_smem_bytes;
    r.blocks_by_threads = lim.max_threads_per_sm / threads_per_block;
    r.blocks_by_regs   = (r.regs_per_block > 0)
        ? lim.max_regs_per_sm / r.regs_per_block : lim.max_blocks_per_sm;
    r.blocks_by_smem    = (r.total_smem_per_block > 0)
        ? lim.max_smem_per_sm_bytes / r.total_smem_per_block
        : lim.max_blocks_per_sm;
    r.blocks_by_hardware = lim.max_blocks_per_sm;

    // Identify binding constraint (smallest among the four).
    int min_v = r.blocks_by_threads; std::string min_name = "threads/SM";
    if (r.blocks_by_regs    < min_v) { min_v = r.blocks_by_regs;    min_name = "registers"; }
    if (r.blocks_by_smem    < min_v) { min_v = r.blocks_by_smem;    min_name = "SMEM"; }
    if (r.blocks_by_hardware < min_v) { min_v = r.blocks_by_hardware; min_name = "hw blocks/SM"; }
    r.binding_constraint = min_name;
    return r;
}

// Probe every kernel we ship, at its production launch configuration.
// Returned dtype is FP16/BF16 for tensor-core kernels and FP32 for the
// scalar fallbacks. Override `dtype_str` to "bfloat16" to probe BF16
// instantiations instead of FP16.
inline std::vector<OccupancyRow> probe_all(
    int m, int D, int newton_iter, const std::string& dtype_str = "half"
) {
    auto lim = query_sm_limits();
    std::vector<OccupancyRow> rows;

    auto run = [&](auto Element_tag, const std::string& tag) {
        using Element = decltype(Element_tag);
        const int mm = m * m;

        // Forward. landmark_kernel launches with block = (1024/D)*D threads and
        // dynamic SMEM = 2*tpd*D floats (tpd = 1024/D) for the split segment
        // reduction; mirror that here so the probe reflects the real config.
        int lm_tpd = 1024 / D; if (lm_tpd < 1) lm_tpd = 1;
        const int lm_block = lm_tpd * D;
        const int lm_smem = static_cast<int>(2 * lm_tpd * D * sizeof(float));
        rows.push_back(probe("landmark_kernel<" + tag + ">",
            landmark_kernel<Element>, lm_block, lm_smem, lim));
        rows.push_back(probe("kernel2_inv_kernel<" + tag + ">",
            kernel2_inv_kernel<Element>, 256, 6 * mm * (int)sizeof(float), lim));

        if (D == 64) {
            using Tk1 = K1Traits<64, Element>;
            using Tk3 = K3Traits<64, Element>;
            rows.push_back(probe("kernel1_fused_tc<D=64," + tag + ">",
                kernel1_fused_tc<Tk1>, Tk1::kNThreads, Tk1::kSmemBytes, lim));
            rows.push_back(probe("kernel3_fused_tc<D=64,pipe," + tag + ">",
                kernel3_fused_tc<Tk3, true>, Tk3::kNThreads, Tk3::kSmemFwdBytes, lim));
            rows.push_back(probe("kernel3_fused_tc<D=64,sync," + tag + ">",
                kernel3_fused_tc<Tk3, false>, Tk3::kNThreads, Tk3::kSmemBytes, lim));
        } else if (D == 128) {
            using Tk1 = K1Traits<128, Element>;
            using Tk3 = K3Traits<128, Element>;
            rows.push_back(probe("kernel1_fused_tc<D=128," + tag + ">",
                kernel1_fused_tc<Tk1>, Tk1::kNThreads, Tk1::kSmemBytes, lim));
            rows.push_back(probe("kernel3_fused_tc<D=128,pipe," + tag + ">",
                kernel3_fused_tc<Tk3, true>, Tk3::kNThreads, Tk3::kSmemFwdBytes, lim));
            rows.push_back(probe("kernel3_fused_tc<D=128,sync," + tag + ">",
                kernel3_fused_tc<Tk3, false>, Tk3::kNThreads, Tk3::kSmemBytes, lim));
        }

        // Backward
        rows.push_back(probe("precompute_di_kernel<" + tag + ">",
            precompute_di_kernel<Element>, 256, 0, lim));

        if (D == 64) {
            using Tk1 = K1Traits<64, Element>;
            using Tk3 = K3Traits<64, Element>;
            rows.push_back(probe("kernel1_bwd_tc<D=64,narrow," + tag + ">",
                kernel1_bwd_tc<Tk1, false>, Tk1::kNThreads,
                (Tk1::kSmemQElems + Tk1::kSmemKVElems * 2) * (int)sizeof(Element),
                lim));
            rows.push_back(probe("kernel1_bwd_tc<D=64,wide," + tag + ">",
                kernel1_bwd_tc<Tk1, true>, Tk1::kNThreads,
                (Tk1::kSmemQElems * 2 + Tk1::kSmemKVElems * 2) * (int)sizeof(Element),
                lim));
            rows.push_back(probe("kernel3_bwd_tc<D=64,narrow," + tag + ">",
                kernel3_bwd_tc<Tk3, false>, Tk3::kNThreads, Tk3::kSmemBwdBytes, lim));
            rows.push_back(probe("kernel3_bwd_tc<D=64,wide," + tag + ">",
                kernel3_bwd_tc<Tk3, true>, Tk3::kNThreads, Tk3::kSmemBwdWideBytes, lim));
            // compute_dk2inv_tc SMEM: sQ + sKV in Element + sB in FP32 (m * D)
            int dk2inv_smem = (Tk3::kSmemQElems + Tk3::kSmemKVElems) * (int)sizeof(Element)
                              + Tk3::kBlockM * 64 * (int)sizeof(float);
            rows.push_back(probe("compute_dk2inv_tc<D=64," + tag + ">",
                compute_dk2inv_tc<Tk3>, Tk3::kNThreads, dk2inv_smem, lim));
        } else if (D == 128) {
            using Tk1 = K1Traits<128, Element>;
            using Tk3 = K3Traits<128, Element>;
            rows.push_back(probe("kernel1_bwd_tc<D=128,narrow," + tag + ">",
                kernel1_bwd_tc<Tk1, false>, Tk1::kNThreads,
                (Tk1::kSmemQElems + Tk1::kSmemKVElems * 2) * (int)sizeof(Element),
                lim));
            rows.push_back(probe("kernel1_bwd_tc<D=128,wide," + tag + ">",
                kernel1_bwd_tc<Tk1, true>, Tk1::kNThreads,
                (Tk1::kSmemQElems * 2 + Tk1::kSmemKVElems * 2) * (int)sizeof(Element),
                lim));
            rows.push_back(probe("kernel3_bwd_tc<D=128,narrow," + tag + ">",
                kernel3_bwd_tc<Tk3, false>, Tk3::kNThreads, Tk3::kSmemBwdBytes, lim));
            rows.push_back(probe("kernel3_bwd_tc<D=128,wide," + tag + ">",
                kernel3_bwd_tc<Tk3, true>, Tk3::kNThreads, Tk3::kSmemBwdWideBytes, lim));
            int dk2inv_smem = (Tk3::kSmemQElems + Tk3::kSmemKVElems) * (int)sizeof(Element)
                              + Tk3::kBlockM * 128 * (int)sizeof(float);
            rows.push_back(probe("compute_dk2inv_tc<D=128," + tag + ">",
                compute_dk2inv_tc<Tk3>, Tk3::kNThreads, dk2inv_smem, lim));
        }

        rows.push_back(probe("compute_dO3_kernel<" + tag + ">",
            compute_dO3_kernel<Element>, 256, 0, lim));

        // Scalar default-mode compute_dk2inv (TILE_N = 32)
        constexpr int TILE_N = 32;
        int dk2inv_scalar_smem =
            (m * D + m * D + TILE_N * D + m * TILE_N + m) * (int)sizeof(float);
        rows.push_back(probe("compute_dk2inv_kernel<" + tag + "> (scalar)",
            compute_dk2inv_kernel<Element>, 256, dk2inv_scalar_smem, lim));

        rows.push_back(probe("ns_bwd_step_kernel",
            ns_bwd_step_kernel, 256, 6 * mm * (int)sizeof(float), lim));
        rows.push_back(probe("ns_bwd_final_kernel<" + tag + ">",
            ns_bwd_final_kernel<Element>, 256, (3 * mm + 8) * (int)sizeof(float), lim));

        rows.push_back(probe("landmark_bwd_kernel<" + tag + ">",
            landmark_bwd_kernel<Element>, std::min(256, D), 0, lim));
    };

    if (dtype_str == "half" || dtype_str == "fp16") {
        run(cutlass::half_t{}, "half");
    } else if (dtype_str == "bfloat16" || dtype_str == "bf16") {
        run(cutlass::bfloat16_t{}, "bfloat16");
    } else {
        // FP32 scalar path
        // Note: only the scalar kernels exist here; TC kernels are FP16/BF16 only
        const int mm = m * m;
        constexpr int TILE_N = 32;
        rows.push_back(probe("ns_bwd_step_kernel",
            ns_bwd_step_kernel, 256, 6 * mm * (int)sizeof(float), lim));
        rows.push_back(probe("ns_bwd_final_kernel<float>",
            ns_bwd_final_kernel<float>, 256, (3 * mm + 8) * (int)sizeof(float), lim));
        rows.push_back(probe("compute_dk2inv_kernel<float>",
            compute_dk2inv_kernel<float>, 256,
            (m * D + m * D + TILE_N * D + m * TILE_N + m) * (int)sizeof(float),
            lim));
    }
    return rows;
}

} // namespace flash_nystrom
