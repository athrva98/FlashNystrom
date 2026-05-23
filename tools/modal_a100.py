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
    # Build for A100 only (no nvidia-smi at image-build time, so pin the arch).
    .env({"FLASH_NYSTROM_CUDA_ARCH_LIST": "80"})
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


@app.function(gpu="A100-80GB", timeout=3600)
def test():
    """Run the full 84-test suite on the A100."""
    import subprocess
    r = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-q", "--tb=short"],
        cwd=REMOTE,
    )
    if r.returncode != 0:
        raise RuntimeError(f"pytest failed with exit code {r.returncode}")
    print("all tests passed on A100")


@app.function(gpu="A100-80GB", timeout=3600)
def bench():
    """FN vs cuBLAS (pure-PyTorch Nystrom reference) on the A100. fwd, bwd, total."""
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
            return FlashNystromFunction.apply(q, k, v, m, 6, True)

    def ref_fwd(q, k, v, m):
        with torch.no_grad():
            return nystrom_attention_reference(q, k, v, m, 6, None, 0)

    def fwdbwd(impl, q, k, v, dout, m):
        def run():
            qq = q.detach().requires_grad_(True)
            kk = k.detach().requires_grad_(True)
            vv = v.detach().requires_grad_(True)
            if impl == "fn":
                out = FlashNystromFunction.apply(qq, kk, vv, m, 6, True)
            else:
                out = nystrom_attention_reference(qq, kk, vv, m, 6, None, 0)
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


@app.function(gpu="A100-80GB", timeout=5400)
def bench_gaps():
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
            return FlashNystromFunction.apply(q, k, v, m, 6, True)

    def ref_fwd(q, k, v, m):
        with torch.no_grad():
            return nystrom_attention_reference(q, k, v, m, 6, None, 0)

    def fwdbwd(impl, q, k, v, dout, m):
        def run():
            qq = q.detach().requires_grad_(True)
            kk = k.detach().requires_grad_(True)
            vv = v.detach().requires_grad_(True)
            if impl == "fn":
                out = FlashNystromFunction.apply(qq, kk, vv, m, 6, True)
            else:
                out = nystrom_attention_reference(qq, kk, vv, m, 6, None, 0)
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
            return FlashNystromFunction.apply(q, k, v, m, 6, True)

    def ref_fwd(q, k, v, m):
        with torch.no_grad():
            return nystrom_attention_reference(q, k, v, m, 6, None, 0)

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
                FlashNystromFunction.apply(q, k, v, m, 6, True)
        torch.cuda.synchronize()
        print(f"\n=== B={B} H={H} N={N} D={D} m={m}  (BH={B*H}) ===", flush=True)
        os.environ["FLASH_NYSTROM_PROFILE"] = "1"
        for _ in range(profiled_calls):
            with torch.no_grad():
                FlashNystromFunction.apply(q, k, v, m, 6, True)
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


@app.local_entrypoint()
def main():
    print("=== tests ===")
    test.remote()
    print("=== benchmark ===")
    bench.remote()
