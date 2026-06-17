# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

# Long-context comparison: patch_size=2 -> N=257 tokens.
# At this N with m=64, m/N=1/4 — closer to Nyströmformer's working regime.
# Tests whether FN matches Nystrom-Ref accuracy when the approximation is meaningful.
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # .../benchmarks
from train_three_way import SDPAAttention, NystromRefAttention, TinyViT, train_one
from flash_nystrom import FlashNystromAttention, NystromConfig

print("="*70)
print("Long-context training (patch_size=2, N=257)")
print("FlashNystrom: m=64, newton_iter=6 (stable default)")
print("="*70)

results = []

print("\n--- 1. SDPA ---")
results.append(train_one("SDPA", lambda d, h: SDPAAttention(d, h),
                         epochs=20, patch_size=2))

print("\n--- 2. Nystrom-Ref (m=64, newton_iter=6) ---")
results.append(train_one("Nystrom-Ref",
    lambda d, h: NystromRefAttention(d, h, num_landmarks=64, newton_iter=6, conv_kernel_size=0),
    epochs=20, patch_size=2))

print("\n--- 3. FlashNystrom (m=64, newton_iter=6) ---")
cfg = NystromConfig(num_landmarks=64, newton_iter=6,
                    conv_kernel_size=0, use_conv_residual=False)
results.append(train_one("FlashNystrom",
    lambda d, h: FlashNystromAttention(d, h, cfg),
    epochs=20, patch_size=2))

print("\n" + "="*70)
print("Summary:")
for r in results:
    print(f"  {r['label']:>14}: test_acc={r['test_acc']:.1f}%  "
          f"avg_epoch={r['avg_epoch']:.1f}s  params={r['params_M']:.2f}M")

with open("long_context_results.json", "w") as f:
    json.dump(results, f, indent=2)
