/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 *
 * Per-kernel profiling for the forward / backward pipelines.
 *
 * Enabled at runtime with FLASH_NYSTROM_PROFILE=1. When enabled, each kernel
 * launch wrapped in KernelProfiler::run is bracketed by CUDA events and the
 * elapsed time is recorded; report() prints a per-kernel breakdown to stderr.
 *
 * This exists because the cost of the pipeline is wildly hardware- and
 * shape-dependent: a kernel that is fine at high batch*head (many CTAs) can
 * starve a large GPU at low batch*head (few CTAs). The breakdown tells you
 * exactly which kernel dominates for a given (BH, N) on a given GPU, so you
 * are not guessing about where the time goes.
 *
 * Overhead: when enabled, run() synchronizes after every kernel (cudaEvent
 * elapsed-time requires the stop event to complete). That serializes the
 * pipeline and inflates absolute totals slightly, but the per-kernel
 * attribution is what matters. When disabled (the default), run() forwards
 * straight to the launch lambda with zero added work.
 ******************************************************************************/
#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace flash_nystrom {

// Is FLASH_NYSTROM_PROFILE set to a non-empty, non-"0" value?
//
// Deliberately NOT cached: read fresh each time a KernelProfiler is
// constructed (once per forward/backward call). This lets the typical
// "warm up with profiling off, then turn it on for a steady-state call"
// workflow work from Python:
//     for _ in range(5): model(x)          # warmup, flag unset
//     os.environ["FLASH_NYSTROM_PROFILE"] = "1"
//     model(x)                             # this call prints the breakdown
// A single getenv per forward/backward is negligible next to kernel launches.
inline bool profiling_enabled() {
    const char* v = std::getenv("FLASH_NYSTROM_PROFILE");
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

// Times each named kernel launch when profiling is on; no-op otherwise.
class KernelProfiler {
public:
    explicit KernelProfiler(cudaStream_t stream)
        : stream_(stream), on_(profiling_enabled()) {
        if (on_) {
            cudaEventCreate(&start_);
            cudaEventCreate(&stop_);
        }
    }

    ~KernelProfiler() {
        if (on_) {
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
        }
    }

    KernelProfiler(const KernelProfiler&) = delete;
    KernelProfiler& operator=(const KernelProfiler&) = delete;

    // Run `launch` (a void() callable that issues one or more kernel launches).
    // When profiling, brackets it with events and records the elapsed ms.
    template <typename LaunchFn>
    void run(const char* name, LaunchFn&& launch) {
        if (!on_) {
            launch();
            return;
        }
        cudaEventRecord(start_, stream_);
        launch();
        cudaEventRecord(stop_, stream_);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        records_.emplace_back(name, ms);
    }

    // Print the per-kernel breakdown + sum to stderr. No-op when disabled.
    void report(const char* phase) const {
        if (!on_) return;
        float total = 0.0f;
        std::fprintf(stderr, "[FlashNystrom profile] %s\n", phase);
        for (const auto& r : records_) {
            std::fprintf(stderr, "    %-30s %9.4f ms\n", r.first.c_str(), r.second);
            total += r.second;
        }
        std::fprintf(stderr, "    %-30s %9.4f ms\n", "(sum of above)", total);
        std::fflush(stderr);
    }

private:
    cudaStream_t stream_;
    bool on_;
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
    std::vector<std::pair<std::string, float>> records_;
};

} // namespace flash_nystrom
