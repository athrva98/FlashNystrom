load("@rules_cuda//cuda:defs.bzl", "cuda_library")

# =============================================================================
# Core CUDA kernel library (standalone, no PyTorch dependency)
# =============================================================================

cuda_library(
    name = "flash_nystrom_kernels",
    srcs = [
        "flash_nystrom_kernels.cu",
    ],
    hdrs = [
        "flash_nystrom.h",
        "utils.h",
        "static_switch.h",
        "kernels/landmark.cuh",
        "kernels/kernel2_inv.cuh",
        "kernels/kernel3_output_fused.cuh",
        "kernels/kernel1_output_fused.cuh",
        "kernels/dconv_residual.cuh",
    ],
    copts = [
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        "--std=c++17",
    ],
    deps = [
        "@cuda//:cuda_runtime",
        "@cutlass",
    ],
    visibility = ["//visibility:public"],
)
