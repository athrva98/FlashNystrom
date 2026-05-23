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
    .apt_install("git", "build-essential")
    .pip_install(
        "torch",
        "pytest",
        "ninja",
        "numpy",
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


@app.local_entrypoint()
def main():
    print("=== tests ===")
    test.remote()
    print("=== benchmark ===")
    bench.remote()
