# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Build and benchmark FlashNystrom on a real A100-80GB via Modal.

This removes the "I can't test on A100" gap: the extension is compiled on a
CUDA 12.8 devel image (sm_80) at image-build time, then the test suite and a
FN-vs-cuBLAS benchmark run on an actual A100-80GB.

Per the project owner's instruction the benchmark compares ONLY against the
cuBLAS path (the pure-PyTorch Nystrom reference, whose every matmul dispatches
to cuBLAS). No SDPA column.

USAGE
-----
One-time auth (opens a browser; you must do this, I can't):
    pip install modal
    modal setup

Run the tests:
    modal run tools/modal_a100.py::test
Run the benchmark:
    modal run tools/modal_a100.py::bench
Both (default entrypoint):
    modal run tools/modal_a100.py

Editing a kernel and re-running `modal run` rebuilds the extension layer
automatically (the local source is part of the image), so the loop is:
    edit -> modal run tools/modal_a100.py::bench -> read A100 numbers.
"""
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
REMOTE = "/root/FlashNystrom"

# CUDA 12.8 devel to match the strict-build nvcc flags (e.g.
# -static-global-template-stub, introduced in 12.8) and the cu128 torch wheels.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    # build-essential = gcc/g++ for nvcc's host compiler; clang because the
    # Modal Python's sysconfig records clang++ as the extension linker (the
    # nvcc compiles use gcc, but the final link shells out to clang++).
    .apt_install("git", "build-essential", "clang")
    # Pin the +cu128 local tag so pip is FORCED to take torch from the cu128
    # index (a bare "torch" with extra_index_url falls back to PyPI's default
    # build, which now bundles CUDA-13 wheels and mismatches the 12.8 nvcc in
    # this image). Pure-python deps (sympy, etc.) still resolve from PyPI.
    .pip_install(
        "torch==2.7.1+cu128",
        "pytest",
        "ninja",
        "numpy",
        # setuptools + wheel are required at build time because we use
        # --no-build-isolation (pip won't provision its own build env, so
        # bdist_wheel must already be importable).
        "setuptools",
        "wheel",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    # Build for A100 (sm_80), H100/H200 (sm_90), and B200 (sm_100): no nvidia-smi
    # at image-build time, so pin all three arches. The SM80 MMA / cp.async atoms
    # compile and run on Hopper and Blackwell in compatibility mode, so one image
    # serves every datacenter GPU (H200 is sm_90 like H100; B200 is sm_100).
    # 90a (not 90): the architecture-specific Hopper target that enables WGMMA +
    # TMA, needed by the native Hopper kernel family. SM80-idiom code still
    # compiles and runs on Hopper under sm_90a. 100 = B200; local builds are sm_120.
    .env({"FLASH_NYSTROM_CUDA_ARCH_LIST": "80 90a 100"})
    .add_local_dir(
        str(REPO),
        remote_path=REMOTE,
        copy=True,  # build-time layer so the run_commands below can compile it
        ignore=[
            "**/.git", "**/__pycache__", "**/*.pyc", "**/*.pyd", "**/*.so",
            "**/*.o", "**/*.a", "build/", "dist/", "**/*.egg-info",
            ".venv*/", "**/.pytest_cache",
            # Trim the CUTLASS tree to the header-only include/ we actually need.
            "third_party/cutlass/test", "third_party/cutlass/examples",
            "third_party/cutlass/tools", "third_party/cutlass/docs",
            "third_party/cutlass/media", "third_party/cutlass/python",
            "third_party/cutlass/.git",
        ],
    )
    .run_commands(
        f"cd {REMOTE} && pip install -e . --no-build-isolation -v"
    )
)

app = modal.App("flash-nystrom-a100", image=image)

# Image with FlashAttention-2 and FlashAttention-3 on top of the FN build, for
# the head-to-head vs exact attention. FA2 ships wheels (falls back to source);
# FA3 (Hopper) has no wheel and must build from the repo's hopper/ subdir. Both
# installs are wrapped with `|| echo` so a failure does NOT abort the image
# build -- bench_fa_h100 imports each at runtime and skips whichever is missing.
fa_image = (
    image
    .env({
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "MAX_JOBS": "4",
        # Trim the FA3 (hopper) build to just what we benchmark: fp16, head
        # dims 64/128, fwd+bwd. Disabling fp8, the larger head dims, paged-KV,
        # split-KV, local/softcap/packgqa, and the sm8x fallback cuts the
        # instantiation count (and the multi-minute compile + log volume that
        # broke the client stream) by most. Unknown flag names are ignored, so
        # this is safe across FA versions.
        # Flag names verified against flash-attention hopper/setup.py. SM80 (not
        # SM8x) is the sm_80 fallback toggle; we run on H100 so skip it, which
        # drops the largest slice of the build. We use flash_attn_func (fixed
        # length, same head dims for q/k/v), so varlen/appendkv/paged/split/
        # packgqa/softcap/local and the non-64/128 head dims are all unused.
        # Cluster and the hdim 64/128 same-dim paths are kept so the FA3 numbers
        # reflect its best kernels.
        "FLASH_ATTENTION_DISABLE_SM80": "TRUE",
        "FLASH_ATTENTION_DISABLE_FP8": "TRUE",
        "FLASH_ATTENTION_DISABLE_HDIM96": "TRUE",
        "FLASH_ATTENTION_DISABLE_HDIM192": "TRUE",
        "FLASH_ATTENTION_DISABLE_HDIM256": "TRUE",
        "FLASH_ATTENTION_DISABLE_PAGEDKV": "TRUE",
        "FLASH_ATTENTION_DISABLE_APPENDKV": "TRUE",
        "FLASH_ATTENTION_DISABLE_SPLIT": "TRUE",
        "FLASH_ATTENTION_DISABLE_LOCAL": "TRUE",
        "FLASH_ATTENTION_DISABLE_SOFTCAP": "TRUE",
        "FLASH_ATTENTION_DISABLE_PACKGQA": "TRUE",
        "FLASH_ATTENTION_DISABLE_VARLEN": "TRUE",
    })
    .run_commands(
        "pip install packaging ninja",
        "pip install flash-attn --no-build-isolation || echo FA2_INSTALL_FAILED",
        "git clone --depth 1 https://github.com/Dao-AILab/flash-attention "
        "/tmp/flash-attention || echo FA_CLONE_FAILED",
        "cd /tmp/flash-attention/hopper && python setup.py install "
        "|| echo FA3_BUILD_FAILED",
    )
)

# Image for the FA4 comparison. FA4 (flash-attn-4) is the CuTeDSL/JIT build that
# runs natively on Hopper AND Blackwell, so it is the right exact-attention
# baseline on B200. It is a pure-python wheel (no nvcc build, no
# --no-build-isolation), JIT-compiled at first call. We also install FA2 (pip
# wheel) for a second exact baseline. We deliberately do NOT build FA3 here: it
# is sm_90a-only and does not run on B200's sm_100.
fa4_image = (
    image
    .run_commands(
        # flash-attn-4 ships only pre-releases (4.0.0bN), so --pre is required or
        # pip reports "no matching distribution".
        "pip install --pre 'flash-attn-4[cu12]' || pip install --pre flash-attn-4 "
        "|| echo FA4_INSTALL_FAILED",
        # The b19 wheel resolves an nvidia-cutlass-dsl whose `cute.core.ThrMma`
        # was removed (AttributeError at import). Pin the version that still
        # matches flash-attn-4 b19's API (upstream guidance: >= 4.5.2).
        "pip install 'nvidia-cutlass-dsl==4.5.2' || echo CUTLASS_DSL_PIN_FAILED",
        "pip install flash-attn --no-build-isolation || echo FA2_INSTALL_FAILED",
    )
)


def _run_tests():
    """Run the full test suite on whichever GPU the wrapper selected."""
    import subprocess
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    r = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-q", "--tb=short"],
        cwd=REMOTE,
    )
    if r.returncode != 0:
        raise RuntimeError(f"pytest failed with exit code {r.returncode}")
    print(f"all tests passed on {torch.cuda.get_device_name(0)}")


@app.function(gpu="A100-80GB", timeout=3600)
def test():
    """Run the full test suite on an A100-80GB."""
    _run_tests()


@app.function(gpu="H100", timeout=3600)
def test_h100():
    """Run the full test suite on an H100."""
    _run_tests()


def _run_bench():
    """FN vs cuBLAS (pure-PyTorch Nystrom reference). fwd, bwd, total.

    GPU-agnostic body; the decorated bench / bench_h100 wrappers pick the GPU.
    """
    import torch
    from flash_nystrom.flash_nystrom import FlashNystromFunction
    from flash_nystrom.reference import nystrom_attention_reference

    dtype = torch.float16
    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print("FN = custom kernels (this repo). cuBLAS = pure-PyTorch Nystrom "
          "(every matmul -> cuBLAS, torch softmax, autograd NS backward).\n")

    def reps_for(N):
        if N <= 8192:    return 10, 30
        if N <= 32768:   return 5, 15
        if N <= 131072:  return 3, 8
        return 2, 5

    def cuda_time(fn, warmup, reps):
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            evs = [(torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
            for s, e in evs:
                s.record(); fn(); e.record()
            torch.cuda.synchronize()
            return sorted(s.elapsed_time(e) for s, e in evs)[reps // 2]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return float("nan")

    def make(B, H, N, D):
        g = lambda: torch.randn(B, H, N, D, dtype=dtype, device=dev)
        return g(), g(), g(), g()

    def fn_fwd(q, k, v, m):
        with torch.no_grad():
            return FlashNystromFunction.apply(q, k, v, m, 6, True, 5.0, True)

    def ref_fwd(q, k, v, m):
        with torch.no_grad():
            return nystrom_attention_reference(q, k, v, m, 6, None, 0, 5.0)

    def fwdbwd(impl, q, k, v, dout, m):
        def run():
            qq = q.detach().requires_grad_(True)
            kk = k.detach().requires_grad_(True)
            vv = v.detach().requires_grad_(True)
            if impl == "fn":
                out = FlashNystromFunction.apply(qq, kk, vv, m, 6, True, 5.0, True)
            else:
                out = nystrom_attention_reference(qq, kk, vv, m, 6, None, 0, 5.0)
            out.backward(dout)
        return run

    hdr = (f"{'B,H':>6} {'N':>7} {'D':>4} {'m':>3} | "
           f"{'FN fwd':>8} {'cuB fwd':>8} {'f x':>5} | "
           f"{'FN tot':>8} {'cuB tot':>8} {'tot x':>6}")
    print(hdr); print("-" * len(hdr))

    for (B, H, D, m) in [(1, 4, 64, 32), (8, 16, 128, 64)]:
        for N in [4096, 16384, 65536, 131072]:
            # The kernels index with int32, so B*H*N*D must fit in int32. The
            # library raises on overflow by design; skip those shapes here.
            if B * H * N * D > 2**31 - 1:
                print(f"{B},{H:>3} {N:>7} {D:>4} {m:>3} | "
                      f"skipped (B*H*N*D={B*H*N*D} > int32 max)")
                continue
            w, r = reps_for(N)
            q, k, v, dout = make(B, H, N, D)

            fnf = cuda_time(lambda: fn_fwd(q, k, v, m), w, r)
            cbf = cuda_time(lambda: ref_fwd(q, k, v, m), w, r)
            fnt = cuda_time(fwdbwd("fn", q, k, v, dout, m), w, r)
            cbt = cuda_time(fwdbwd("ref", q, k, v, dout, m), w, r)

            def ratio(a, b):
                return f"{b/a:5.2f}x" if (a == a and b == b and a > 0) else "   - "
            print(f"{B},{H:>3} {N:>7} {D:>4} {m:>3} | "
                  f"{fnf:8.3f} {cbf:8.3f} {ratio(fnf, cbf):>5} | "
                  f"{fnt:8.3f} {cbt:8.3f} {ratio(fnt, cbt):>6}")
            del q, k, v, dout
            torch.cuda.empty_cache()

    print("\nf x / tot x = cuBLAS_time / FN_time. >1 means FN faster.")


@app.function(gpu="A100-80GB", timeout=3600)
def bench():
    """FN vs cuBLAS head-to-head on an A100-80GB."""
    _run_bench()


@app.function(gpu="H100", timeout=3600)
def bench_h100():
    """FN vs cuBLAS head-to-head on an H100."""
    _run_bench()


def _run_bench_gaps():
    """Fill the two gaps in the main bench, thoroughly.

    1. High-BH at large N: B=8,H=16 (BH=128) overflows int32 at N=131072. Use
       BH=64 (B=4,H=16, D=128) so N=131072 fits (2^30 elems) and we can confirm
       the forward f x keeps climbing with N at high batch*head.
    2. Low-BH at very large N: B=1,H=4 (BH=4, D=64) pushed to 2M tokens to see
       whether the end-to-end total stays at parity or tips further as cuBLAS
       fills the GPU. D=64/BH=4 keeps memory modest (N=2M is 537M elems).

    fwd, fwd+bwd total, vs the cuBLAS pure-PyTorch Nystrom reference only.
    """
    import torch
    from flash_nystrom.flash_nystrom import FlashNystromFunction
    from flash_nystrom.reference import nystrom_attention_reference

    dtype = torch.float16
    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"SMs: {torch.cuda.get_device_properties(0).multi_processor_count}")
    print("FN = custom kernels. cuBLAS = pure-PyTorch Nystrom reference.\n")

    def reps_for(N):
        if N <= 8192:    return 10, 30
        if N <= 32768:   return 5, 15
        if N <= 131072:  return 3, 8
        if N <= 524288:  return 2, 5
        return 1, 3

    def cuda_time(fn, warmup, reps):
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            evs = [(torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
            for s, e in evs:
                s.record(); fn(); e.record()
            torch.cuda.synchronize()
            return sorted(s.elapsed_time(e) for s, e in evs)[reps // 2]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return float("nan")

    def make(B, H, N, D):
        g = lambda: torch.randn(B, H, N, D, dtype=dtype, device=dev)
        return g(), g(), g(), g()

    def fn_fwd(q, k, v, m):
        with torch.no_grad():
            return FlashNystromFunction.apply(q, k, v, m, 6, True, 5.0, True)

    def ref_fwd(q, k, v, m):
        with torch.no_grad():
            return nystrom_attention_reference(q, k, v, m, 6, None, 0, 5.0)

    def fwdbwd(impl, q, k, v, dout, m):
        def run():
            qq = q.detach().requires_grad_(True)
            kk = k.detach().requires_grad_(True)
            vv = v.detach().requires_grad_(True)
            if impl == "fn":
                out = FlashNystromFunction.apply(qq, kk, vv, m, 6, True, 5.0, True)
            else:
                out = nystrom_attention_reference(qq, kk, vv, m, 6, None, 0, 5.0)
            out.backward(dout)
        return run

    def ratio(a, b):
        return f"{b/a:5.2f}x" if (a == a and b == b and a > 0) else "   - "

    groups = [
        ("HIGH BH  (B=4, H=16, D=128, m=64  -> BH=64)",
         4, 16, 128, 64, [4096, 16384, 65536, 131072]),
        ("LOW BH   (B=1, H=4,  D=64,  m=32  -> BH=4)",
         1, 4, 64, 32, [65536, 131072, 262144, 524288, 1048576, 2097152]),
    ]

    hdr = (f"{'B,H':>6} {'N':>8} {'D':>4} {'m':>3} | "
           f"{'FN fwd':>9} {'cuB fwd':>9} {'f x':>6} | "
           f"{'FN tot':>9} {'cuB tot':>9} {'tot x':>6}")

    for (label, B, H, D, m, Ns) in groups:
        print(f"\n### {label}")
        print(hdr); print("-" * len(hdr))
        for N in Ns:
            if B * H * N * D > 2**31 - 1:
                print(f"{B},{H:>3} {N:>8} {D:>4} {m:>3} | "
                      f"skipped (B*H*N*D > int32 max)")
                continue
            w, r = reps_for(N)
            q, k, v, dout = make(B, H, N, D)
            fnf = cuda_time(lambda: fn_fwd(q, k, v, m), w, r)
            cbf = cuda_time(lambda: ref_fwd(q, k, v, m), w, r)
            fnt = cuda_time(fwdbwd("fn", q, k, v, dout, m), w, r)
            cbt = cuda_time(fwdbwd("ref", q, k, v, dout, m), w, r)
            print(f"{B},{H:>3} {N:>8} {D:>4} {m:>3} | "
                  f"{fnf:9.3f} {cbf:9.3f} {ratio(fnf, cbf):>6} | "
                  f"{fnt:9.3f} {cbt:9.3f} {ratio(fnt, cbt):>6}")
            del q, k, v, dout
            torch.cuda.empty_cache()

    print("\nf x / tot x = cuBLAS_time / FN_time. >1 means FN faster. "
          "'-' = cuBLAS reference OOM'd at that N.")


@app.function(gpu="A100-80GB", timeout=5400)
def bench_gaps():
    """Extended high-BH + long-context sweep on an A100-80GB."""
    _run_bench_gaps()


@app.function(gpu="H100", timeout=5400)
def bench_gaps_h100():
    """Extended high-BH + long-context sweep on an H100."""
    _run_bench_gaps()


@app.function(gpu="H200", timeout=5400)
def bench_gaps_h200():
    """Extended high-BH + long-context sweep on an H200 (sm_90, 141 GB HBM3e)."""
    _run_bench_gaps()


@app.function(gpu="A100", timeout=5400)
def scaling_a100():
    """Training-throughput scaling sweep on A100 (profile_scaling.py).

    Source data for the paper's throughput-crossover figure: K tokens/s
    (fwd+bwd, largest batch that fits per backend) versus N for SDPA,
    FlashNystrom, and the cuBLAS Nystrom reference. Prints the JSON so the
    caller can regenerate the figure locally.
    """
    import subprocess
    r = subprocess.run(
        ["python", "benchmarks/profile_scaling.py",
         "--backends", "sdpa", "flash_nystrom", "nystrom_reference",
         "--Ns", "256", "512", "1024", "2048", "4096", "8192", "16384",
         "--json", "/tmp/scaling.json"],
        capture_output=True, text=True, cwd="/root/FlashNystrom")
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise RuntimeError("profile_scaling failed")
    print("===SCALING_JSON_BEGIN===")
    print(open("/tmp/scaling.json").read())
    print("===SCALING_JSON_END===")


@app.function(gpu="B200", timeout=3600)
def bench_bwd_profile_b200():
    """Full per-kernel backward profile at the gap-sweep shapes (sm80 path).

    Shows which backward kernels dominate the rows where FN loses to cuBLAS
    on B200, to direct the Blackwell-native porting order.
    """
    import os
    import subprocess
    script = r"""
import torch, sys
from flash_nystrom import flash_nystrom_attention
B, H, N, D, m = (int(x) for x in sys.argv[1:6])
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn_like(q).requires_grad_(True)
v = torch.randn_like(q).requires_grad_(True)
do = torch.randn_like(q)
for _ in range(8):
    out = flash_nystrom_attention(q, k, v, num_landmarks=m)
    out.backward(do)
    q.grad = k.grad = v.grad = None
torch.cuda.synchronize()
"""
    shapes = [
        (4, 16, 16384, 128, 64),    # high-BH row (tot 0.72x)
        (1, 4, 524288, 64, 32),     # long-context row (tot 0.55x)
        (1, 4, 2097152, 64, 32),    # longest row (tot 0.78x)
    ]
    env = dict(os.environ, FLASH_NYSTROM_PROFILE="1")
    for shape in shapes:
        r = subprocess.run(
            ["python", "-c", script] + [str(x) for x in shape],
            env=env, capture_output=True, text=True, cwd="/root/FlashNystrom")
        out = (r.stdout + r.stderr).splitlines()
        # print the LAST backward profile block (steady state)
        idxs = [i for i, l in enumerate(out) if "backward (FP16" in l]
        print(f"=== B,H,N,D,m = {shape} (rc={r.returncode}) ===")
        if idxs:
            i = idxs[-1]
            for l in out[i:i + 12]:
                print(l)
        else:
            for l in out[-8:]:
                print(l)


@app.function(gpu="B200", timeout=5400)
def bench_gaps_b200():
    """Extended high-BH + long-context sweep on a B200 (sm_100, Blackwell)."""
    _run_bench_gaps()


@app.function(gpu="H200", timeout=3600)
def test_h200():
    """Full test suite on an H200."""
    _run_tests()


@app.function(gpu="B200", timeout=3600)
def test_b200():
    """Full test suite on a B200 (sm_100)."""
    _run_tests()


@app.function(gpu="A100-80GB", timeout=3600)
def bench_sweep():
    """Forward-only: sweep FLASH_NYSTROM_KERNEL3_SPLITS to find the best path.

    The benchmark shapes default to the multi-CTA split path (kernel3_partial_tc,
    NOT pipelined). This sweep forces split counts -- including 1, which selects
    the pipelined single-CTA kernel3_fused_tc -- to measure which path is fastest
    on the A100 for each shape. fwd-only; cuBLAS reference shown for scale.
    """
    import os
    import torch
    from flash_nystrom.flash_nystrom import FlashNystromFunction
    from flash_nystrom.reference import nystrom_attention_reference

    dtype = torch.float16
    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"SMs: {torch.cuda.get_device_properties(0).multi_processor_count}\n")

    def cuda_time(fn, warmup, reps):
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            evs = [(torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
            for s, e in evs:
                s.record(); fn(); e.record()
            torch.cuda.synchronize()
            return sorted(s.elapsed_time(e) for s, e in evs)[reps // 2]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return float("nan")

    def make(B, H, N, D):
        g = lambda: torch.randn(B, H, N, D, dtype=dtype, device=dev)
        return g(), g(), g()

    def fn_fwd(q, k, v, m):
        with torch.no_grad():
            return FlashNystromFunction.apply(q, k, v, m, 6, True, 5.0, True)

    def ref_fwd(q, k, v, m):
        with torch.no_grad():
            return nystrom_attention_reference(q, k, v, m, 6, None, 0, 5.0)

    # "0" = auto (atoi("0") < 1 so the kernel falls through to its heuristic).
    # "1" = pipelined single-CTA kernel3_fused_tc. >=2 = split path.
    splits_list = ["0", "1", "2", "4", "8", "16"]
    configs = [(1, 4, 64, 32), (8, 16, 128, 64)]
    Ns = [16384, 65536]

    hdr = (f"{'B,H':>6} {'N':>7} {'D':>4} {'m':>3} | "
           + " ".join(f"{('auto' if s=='0' else 'sp'+s):>8}" for s in splits_list)
           + f" | {'cuBLAS':>8}")
    print(hdr); print("-" * len(hdr))

    for (B, H, D, m) in configs:
        for N in Ns:
            w, r = (5, 15) if N <= 16384 else (3, 8)
            q, k, v = make(B, H, N, D)
            row = []
            for s in splits_list:
                os.environ["FLASH_NYSTROM_KERNEL3_SPLITS"] = s
                row.append(cuda_time(lambda: fn_fwd(q, k, v, m), w, r))
            os.environ.pop("FLASH_NYSTROM_KERNEL3_SPLITS", None)
            cb = cuda_time(lambda: ref_fwd(q, k, v, m), w, r)
            print(f"{B},{H:>3} {N:>7} {D:>4} {m:>3} | "
                  + " ".join(f"{t:8.3f}" for t in row)
                  + f" | {cb:8.3f}")
            del q, k, v
            torch.cuda.empty_cache()

    print("\nAll times = FN forward (ms), median. Lower is better. sp1 = pipelined "
          "single-CTA path; sp>=2 / auto = split path. cuBLAS = reference forward.")


@app.function(gpu="A100-80GB", timeout=3600)
def bench_profile():
    """Per-kernel forward breakdown (FLASH_NYSTROM_PROFILE=1) on the A100.

    The split sweep showed kernel3's split count barely moves the high-BH (D=128)
    forward and cuBLAS still wins, so the bottleneck may not be kernel3 at all.
    This prints where the FN forward time actually goes -- landmarks, scale,
    kernel2_inv, kernel3_output_fused, kernel1_output_fused -- for the losing
    (high-BH) and winning (low-BH) configs, so we target the real cost.
    """
    import os
    import torch
    from flash_nystrom.flash_nystrom import FlashNystromFunction

    dtype = torch.float16
    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"SMs: {torch.cuda.get_device_properties(0).multi_processor_count}\n")

    def run_one(B, H, N, D, m, profiled_calls=2):
        q = torch.randn(B, H, N, D, dtype=dtype, device=dev)
        k = torch.randn(B, H, N, D, dtype=dtype, device=dev)
        v = torch.randn(B, H, N, D, dtype=dtype, device=dev)
        os.environ.pop("FLASH_NYSTROM_PROFILE", None)
        for _ in range(5):
            with torch.no_grad():
                FlashNystromFunction.apply(q, k, v, m, 6, True, 5.0, True)
        torch.cuda.synchronize()
        print(f"\n=== B={B} H={H} N={N} D={D} m={m}  (BH={B*H}) ===", flush=True)
        os.environ["FLASH_NYSTROM_PROFILE"] = "1"
        for _ in range(profiled_calls):
            with torch.no_grad():
                FlashNystromFunction.apply(q, k, v, m, 6, True, 5.0, True)
            torch.cuda.synchronize()
        os.environ.pop("FLASH_NYSTROM_PROFILE", None)
        del q, k, v
        torch.cuda.empty_cache()

    run_one(8, 16, 65536, 128, 64)   # high BH, the config FN loses on
    run_one(8, 16, 16384, 128, 64)   # high BH, smaller N
    run_one(1, 4, 65536, 64, 32)     # low BH, the config FN wins on

    print("\nPer-kernel ms is event-timed with a sync after each kernel (profiling "
          "serializes the pipeline, so the sum overstates the real wall time, but "
          "the per-kernel ATTRIBUTION is accurate).")


@app.function(gpu="H100", timeout=7200, image=fa_image)
def bench_fa_h100():
    """FlashNystrom vs FlashAttention-2 vs FlashAttention-3 on the H100.

    FA2/FA3 compute EXACT O(N^2) attention; FN is approximate O(m*N). So this
    is the "vs exact attention" comparison (like the 5060 SDPA table) but
    against the actual SOTA Hopper kernels. FA expects (B, S, H, D); our
    tensors are (B, H, N, D), so we transpose for the FA calls. FA is run only
    where it is compute-feasible (it is O(N^2)); past that FN keeps scaling.
    """
    import torch
    from flash_nystrom.flash_nystrom import flash_nystrom_attention

    fa2 = fa3 = None
    try:
        from flash_attn import flash_attn_func as fa2
    except Exception as e:  # noqa: BLE001
        print("FA2 unavailable:", repr(e))
    try:
        from flash_attn_interface import flash_attn_func as fa3
    except Exception as e:  # noqa: BLE001
        print("FA3 unavailable:", repr(e))

    dtype = torch.float16
    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"FA2: {'available' if fa2 else 'MISSING'}   "
          f"FA3: {'available' if fa3 else 'MISSING'}\n")

    def cuda_time(fn, warmup, reps):
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            evs = [(torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
            for s, e in evs:
                s.record(); fn(); e.record()
            torch.cuda.synchronize()
            return sorted(s.elapsed_time(e) for s, e in evs)[reps // 2]
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            torch.cuda.empty_cache()
            return float("nan")

    # FA takes (B, S, H, D); some versions return (out, lse). Normalize both.
    def fa_out(o):
        return o[0] if isinstance(o, tuple) else o

    def fn_fwdbwd(q, k, v, dout, m):
        def run():
            qq = q.detach().requires_grad_(True)
            kk = k.detach().requires_grad_(True)
            vv = v.detach().requires_grad_(True)
            flash_nystrom_attention(qq, kk, vv, num_landmarks=m,
                                    newton_iter=6, kappa_star=5.0).backward(dout)
        return run

    def fa_fwdbwd(fa, q, k, v, dout):
        def run():
            qq = q.transpose(1, 2).detach().requires_grad_(True)
            kk = k.transpose(1, 2).detach().requires_grad_(True)
            vv = v.transpose(1, 2).detach().requires_grad_(True)
            o = fa_out(fa(qq, kk, vv, causal=False))
            o.backward(dout.transpose(1, 2))
        return run

    def reps_for(N):
        if N <= 16384:   return 5, 15
        if N <= 65536:   return 3, 8
        if N <= 262144:  return 2, 5
        return 1, 2

    def rx(a, b):
        return f"{b/a:6.1f}x" if (a == a and b == b and a > 0) else "   -  "

    # (label, B, H, D, m, [N], FA feasibility cap)
    configs = [
        ("HIGH BH (B=4, H=16, D=128, m=64)",
         4, 16, 128, 64, [4096, 16384, 65536, 131072], 131072),
        ("LONG CONTEXT (B=1, H=4, D=64, m=32)",
         1, 4, 64, 32, [16384, 65536, 131072, 262144, 524288, 1048576, 2097152],
         1048576),
    ]
    hdr = (f"{'B,H':>6} {'N':>8} | {'FN tot':>8} {'FA2 tot':>9} {'FA3 tot':>9} | "
           f"{'FA2/FN':>7} {'FA3/FN':>7}")
    for (label, B, H, D, m, Ns, fa_cap) in configs:
        print(f"\n### {label}  (fwd+bwd ms; FA2/FA3 = exact O(N^2))")
        print(hdr); print("-" * len(hdr))
        for N in Ns:
            if B * H * N * D > 2**31 - 1:
                print(f"{B},{H:>3} {N:>8} | FN exceeds int32 element cap; skipped")
                continue
            g = lambda: torch.randn(B, H, N, D, dtype=dtype, device=dev)
            q, k, v, dout = g(), g(), g(), g()
            w, r = reps_for(N)
            fnt = cuda_time(fn_fwdbwd(q, k, v, dout, m), w, r)
            fa2t = (cuda_time(fa_fwdbwd(fa2, q, k, v, dout), w, r)
                    if (fa2 and N <= fa_cap) else float("nan"))
            fa3t = (cuda_time(fa_fwdbwd(fa3, q, k, v, dout), w, r)
                    if (fa3 and N <= fa_cap) else float("nan"))
            print(f"{B},{H:>3} {N:>8} | {fnt:8.2f} {fa2t:9.2f} {fa3t:9.2f} | "
                  f"{rx(fnt, fa2t):>7} {rx(fnt, fa3t):>7}")
            del q, k, v, dout
            torch.cuda.empty_cache()

    print("\nFA2/FA3 are exact O(N^2) attention; FN is approximate O(m*N). "
          "FA2/FN and FA3/FN = FA_total / FN_total; >1 means FN is faster. "
          "'-' = FA past its compute-feasible N (O(N^2) wall); FN keeps scaling.")


def _run_bench_fa4():
    """FlashNystrom vs FlashAttention-2 and FlashAttention-4 (CuTeDSL/JIT).

    FA4 (`pip flash-attn-4`) is the Blackwell/Hopper-native exact-attention
    build, so on a B200 it is the right exact baseline (FA3 is sm_90a-only and
    does not run here). FN is approximate O(mN). We measure BOTH forward-only and
    fwd+bwd, so FA4 is captured even if its CuTeDSL build is forward-only (the
    fwd column is still a fair comparison). FA tensors are (B,S,H,D); FN is
    (B,H,N,D), so we transpose for the FA calls.
    """
    import torch
    from flash_nystrom.flash_nystrom import flash_nystrom_attention
    fa2 = fa4 = None
    try:
        from flash_attn import flash_attn_func as fa2
    except Exception as e:  # noqa: BLE001
        print("FA2 unavailable:", repr(e))
    try:
        from flash_attn.cute import flash_attn_func as fa4
    except Exception as e:  # noqa: BLE001
        print("FA4 unavailable:", repr(e))

    dtype = torch.float16; dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"FA2: {'available' if fa2 else 'MISSING'}   "
          f"FA4: {'available' if fa4 else 'MISSING'}\n")

    def cuda_time(fn, w, r):
        try:
            for _ in range(w): fn()
            torch.cuda.synchronize()
            evs = [(torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True)) for _ in range(r)]
            for s, e in evs: s.record(); fn(); e.record()
            torch.cuda.synchronize()
            return sorted(s.elapsed_time(e) for s, e in evs)[r // 2]
        except Exception:  # noqa: BLE001  (OOM, or FA4 has no backward)
            torch.cuda.empty_cache(); return float("nan")

    def fa_out(o): return o[0] if isinstance(o, tuple) else o

    def fn_fwd(q, k, v, m):
        def run():
            with torch.no_grad():
                flash_nystrom_attention(q, k, v, num_landmarks=m, newton_iter=6,
                                        kappa_star=5.0)
        return run

    def fn_fb(q, k, v, dout, m):
        def run():
            qq = q.detach().requires_grad_(True)
            kk = k.detach().requires_grad_(True)
            vv = v.detach().requires_grad_(True)
            flash_nystrom_attention(qq, kk, vv, num_landmarks=m, newton_iter=6,
                                    kappa_star=5.0).backward(dout)
        return run

    def fa_fwd(fa, q, k, v):
        def run():
            with torch.no_grad():
                fa_out(fa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                          causal=False))
        return run

    def fa_fb(fa, q, k, v, dout):
        def run():
            qq = q.transpose(1, 2).detach().requires_grad_(True)
            kk = k.transpose(1, 2).detach().requires_grad_(True)
            vv = v.transpose(1, 2).detach().requires_grad_(True)
            fa_out(fa(qq, kk, vv, causal=False)).backward(dout.transpose(1, 2))
        return run

    def reps(N):
        if N <= 16384:  return 5, 15
        if N <= 65536:  return 3, 8
        if N <= 262144: return 2, 5
        return 1, 2

    def rx(a, b):
        return f"{b/a:6.1f}x" if (a == a and b == b and a > 0) else "   -  "

    configs = [
        ("HIGH BH (B=4, H=16, D=128, m=64)",
         4, 16, 128, 64, [4096, 16384, 65536, 131072], 131072),
        ("LONG CONTEXT (B=1, H=4, D=64, m=32)",
         1, 4, 64, 32, [16384, 65536, 131072, 262144, 524288, 1048576, 2097152],
         1048576),
    ]
    for (label, B, H, D, m, Ns, cap) in configs:
        print(f"\n### {label}  (ms; FA2/FA4 = exact O(N^2))")
        hdr = (f"{'N':>8} | {'FN fwd':>7} {'FN tot':>7} | {'FA2 fwd':>8} {'FA2 tot':>8} | "
               f"{'FA4 fwd':>8} {'FA4 tot':>8} | {'FA4f/FNf':>8} {'FA4t/FNt':>8}")
        print(hdr); print("-" * len(hdr))
        for N in Ns:
            if B * H * N * D > 2**31 - 1:
                print(f"{N:>8} | FN exceeds int32 element cap; skipped"); continue
            g = lambda: torch.randn(B, H, N, D, dtype=dtype, device=dev)
            q, k, v, dout = g(), g(), g(), g()
            w, r = reps(N)
            fnf = cuda_time(fn_fwd(q, k, v, m), w, r)
            fnt = cuda_time(fn_fb(q, k, v, dout, m), w, r)
            f2f = cuda_time(fa_fwd(fa2, q, k, v), w, r) if (fa2 and N <= cap) else float("nan")
            f2t = cuda_time(fa_fb(fa2, q, k, v, dout), w, r) if (fa2 and N <= cap) else float("nan")
            f4f = cuda_time(fa_fwd(fa4, q, k, v), w, r) if (fa4 and N <= cap) else float("nan")
            f4t = cuda_time(fa_fb(fa4, q, k, v, dout), w, r) if (fa4 and N <= cap) else float("nan")
            print(f"{N:>8} | {fnf:7.2f} {fnt:7.2f} | {f2f:8.2f} {f2t:8.2f} | "
                  f"{f4f:8.2f} {f4t:8.2f} | {rx(fnf, f4f):>8} {rx(fnt, f4t):>8}")
            del q, k, v, dout; torch.cuda.empty_cache()

    print("\nFA4 fwd / FA4 tot = forward-only / fwd+bwd (separate so FA4 is captured "
          "even if its CuTeDSL build is forward-only). FA4f/FNf and FA4t/FNt = "
          "FA4 / FN; >1 means FN is faster. nan = OOM, past compute-feasible N, or "
          "operation unsupported.")


@app.function(gpu="B200", timeout=7200, image=fa4_image)
def bench_fa4_b200():
    """FN vs FlashAttention-2/4 on a B200 (FA4's native Blackwell card)."""
    _run_bench_fa4()


@app.function(gpu="H200", timeout=7200, image=fa4_image)
def bench_fa4_h200():
    """FN vs FlashAttention-2/4 on an H200 (FA4 also runs on Hopper)."""
    _run_bench_fa4()


@app.local_entrypoint()
def main():
    print("=== tests ===")
    test.remote()
    print("=== benchmark ===")
    bench.remote()
