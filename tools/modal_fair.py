"""Fair-baseline check: FlashNystrom vs FUSED bidirectional linear attention.

The operator table compared our fused kernel against a pure-PyTorch linear
attention. fla-org/flash-bidirectional-linear-attention provides Triton kernels
for exactly this regime (non-causal linear attention, fwd+bwd), and its own
benchmark reports ~2.1x fwd / ~2.3x bwd over a torch baseline. Comparing a
fused kernel against an unfused baseline inflates our margin, so this measures
against the fused one.
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
REMOTE = "/root/FlashNystrom"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "clang")
    .pip_install("torch==2.7.1+cu128", "pytest", "ninja", "numpy", "einops",
                 "setuptools", "wheel", "triton",
                 extra_index_url="https://download.pytorch.org/whl/cu128")
    .run_commands(
        "git clone https://github.com/fla-org/flash-bidirectional-linear-attention.git /tmp/fbla",
        "pip install -e /tmp/fbla/. || echo FBLA_INSTALL_FAILED",
    )
    .env({"FLASH_NYSTROM_CUDA_ARCH_LIST": "80"})
    .add_local_dir(str(REPO), remote_path=REMOTE, copy=True,
                   ignore=["**/.git", "**/__pycache__", "**/*.so", "**/*.o",
                           "build/", "dist/", "**/*.egg-info", "runs/",
                           "third_party/cutlass/test", "third_party/cutlass/examples",
                           "third_party/cutlass/tools", "third_party/cutlass/docs",
                           "third_party/cutlass/media", "third_party/cutlass/python"])
    .run_commands(f"cd {REMOTE} && pip install -e . --no-build-isolation")
)
app = modal.App("fn-fair-baselines", image=image)


@app.function(gpu="A100-80GB", timeout=5400)
def probe_and_bench():
    import sys, torch
    sys.path.insert(0, REMOTE)

    # 1. What does flash_bla actually expose?
    print("=== flash_bla ops discovery ===")
    try:
        import flash_bla
        print("flash_bla imported from", flash_bla.__file__)
        import pkgutil, importlib
        for m in pkgutil.walk_packages(flash_bla.__path__, "flash_bla."):
            if "ops" in m.name or "layer" in m.name:
                try:
                    mod = importlib.import_module(m.name)
                    fns = [n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n))]
                    if fns:
                        print(f"  {m.name}: {fns[:10]}")
                except Exception as e:
                    print(f"  {m.name}: import err {str(e)[:50]}")
    except Exception as e:
        print("flash_bla import FAILED:", e)
        return

    # 2. Benchmark whatever bidirectional linear-attention op we found
    from flash_nystrom import flash_nystrom_attention as fn
    from benchmarks.baseline_ops import linear_attention_op

    def timed(f, q, k, v, warmup=5, iters=20):
        try:
            for _ in range(warmup):
                o = f(q, k, v); o.float().pow(2).sum().backward()
                q.grad = k.grad = v.grad = None
            torch.cuda.synchronize()
            ts = []
            for _ in range(iters):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record(); o = f(q, k, v); o.float().pow(2).sum().backward(); e.record()
                torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
                q.grad = k.grad = v.grad = None
            ts.sort(); return ts[len(ts)//2]
        except Exception as ex:
            return f"ERR:{str(ex)[:44]}"

    # Exact entry points, read from the repo tree:
    #   flash_bla/ops/simple_la/fused.py    -> simple_la(q, k, v, scale)
    #   flash_bla/ops/linear_attn/fused.py  -> linear_attention(...)
    # Both wrap a torch.autograd.Function, so fwd+bwd are both fused Triton.
    import inspect
    fused_ops = {}
    for label, path, name in (
        ("simple_la", "flash_bla.ops.simple_la.fused", "simple_la"),
        ("linear_attn", "flash_bla.ops.linear_attn.fused", "linear_attention"),
    ):
        try:
            import importlib
            mod = importlib.import_module(path)
            fn_ = getattr(mod, name)
            fused_ops[label] = fn_
            print(f"  {label}: {path}.{name}{inspect.signature(fn_)}")
        except Exception as e:
            print(f"  {label}: unavailable ({str(e)[:60]})")
    if not fused_ops:
        print("!! no fused entry point importable"); return

    B, H, D, m = 1, 8, 64, 64
    print(f"\n=== fwd+bwd ms, B={B} H={H} D={D} m={m} ===")
    hdr = [f"{'FlashNystrom':>11}", f"{'LA (torch)':>11}"] + [f"{k+' (fused)':>11}" for k in fused_ops]
    print(f"{'N':>9} | " + " | ".join(hdr))
    for N in (131072, 262144, 1048576):
        q, k, v = [torch.randn(B, H, N, D, device="cuda", dtype=torch.float16,
                               requires_grad=True) for _ in range(3)]
        t_fn = timed(lambda a,b,c: fn(a,b,c,num_landmarks=m,kappa_star=1e3,use_tc_pinv=True), q,k,v)
        t_t  = timed(linear_attention_op, q, k, v)
        fmt = lambda t: f"{t:11.2f}" if isinstance(t, float) else f"{str(t)[:11]:>11}"
        cells = [fmt(t_fn), fmt(t_t)]
        for label, f_ in fused_ops.items():
            cells.append(fmt(timed(lambda a,b,c: f_(a,b,c), q, k, v)))
        print(f"{N:>9} | " + " | ".join(cells))
        del q,k,v; torch.cuda.empty_cache()
