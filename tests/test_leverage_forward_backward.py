"""End-to-end validation of landmark_mode=1 (leverage-Voronoi) through the fused
FlashNystrom forward AND its straight-through backward.

The kernel and the reference are made to use the SAME landmarks and the SAME
Voronoi membership:

  * forward  -- the fused output must equal nystrom_attention_reference() fed the
    leverage landmarks the kernel itself produced.
  * backward -- the kernel's straight-through gradient (membership held fixed,
    landmark = scale * cell mean) equals autograd through a torch landmark built
    as exactly that fixed scatter-mean. So we rebuild q_tilde/k_tilde
    differentiably from the forward's saved assignment and compare grads.
"""
import pytest
import torch

flash_nystrom = pytest.importorskip("flash_nystrom")
import flash_nystrom._C as _C
from flash_nystrom.flash_nystrom import flash_nystrom_attention
from flash_nystrom.reference import nystrom_attention_reference

DEV = "cuda"


def _scatter_mean_landmarks(x_s, assign, m):
    """Differentiable landmark = scale * mean over the FIXED Voronoi cell.
    x_s: (B,H,N,D) already scaled. assign: (B,H,N) int cell id. Counts are >=1
    (every seed lands in its own cell), so the clamp is only a safety net."""
    B, H, N, D = x_s.shape
    idx = assign.long().clamp(min=0)                       # -1 (unprocessed) -> 0; absent at subsample=1
    idxD = idx.unsqueeze(-1).expand(-1, -1, -1, D)
    acc = x_s.new_zeros(B, H, m, D).scatter_add_(2, idxD, x_s)
    cnt = x_s.new_zeros(B, H, m, 1).scatter_add_(
        2, idx.unsqueeze(-1), x_s.new_ones(B, H, N, 1))
    return acc / cnt.clamp(min=1.0)


def _fwd_with_assign(q, k, v, m, seed):
    # raw _C.forward to fetch the per-row assignment/counts the kernel used
    res = _C.forward(q.contiguous(), k.contiguous(), v.contiguous(),
                     m, 6, 0.0, False, 1, seed, 1)
    # [..., 13]=q_assign [14]=k_assign [15]=q_cnt [16]=k_cnt
    return res[13], res[14]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("D", [64])
def test_mode1_forward_matches_reference(D):
    B, H, N, m, seed = 2, 3, 512, 32, 7
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device=DEV)
    k = torch.randn(B, H, N, D, device=DEV)
    v = torch.randn(B, H, N, D, device=DEV)

    out = flash_nystrom_attention(q, k, v, num_landmarks=m, newton_iter=6,
                                  landmark_mode=1, landmark_seed=seed)

    scale = D ** -0.25
    qa, ka = _fwd_with_assign(q, k, v, m, seed)
    qt = _scatter_mean_landmarks((q * scale), qa.view(B, H, N), m)
    kt = _scatter_mean_landmarks((k * scale), ka.view(B, H, N), m)
    ref = nystrom_attention_reference(q, k, v, num_landmarks=m, newton_iter=6,
                                      kappa_star=0.0, q_tilde=qt, k_tilde=kt)

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mode1_backward_matches_straight_through():
    B, H, N, D, m, seed = 2, 2, 512, 64, 32, 11
    torch.manual_seed(1)
    q0 = torch.randn(B, H, N, D, device=DEV)
    k0 = torch.randn(B, H, N, D, device=DEV)
    v0 = torch.randn(B, H, N, D, device=DEV)
    go = torch.randn(B, H, N, D, device=DEV)

    # fixed membership from the kernel forward
    qa, ka = _fwd_with_assign(q0, k0, v0, m, seed)
    qa = qa.view(B, H, N); ka = ka.view(B, H, N)

    # --- kernel grads
    q1 = q0.clone().requires_grad_(True)
    k1 = k0.clone().requires_grad_(True)
    v1 = v0.clone().requires_grad_(True)
    out = flash_nystrom_attention(q1, k1, v1, num_landmarks=m, newton_iter=6,
                                  landmark_mode=1, landmark_seed=seed)
    (out * go).sum().backward()

    # --- reference grads: same fixed-membership landmarks, autograd through them
    q2 = q0.clone().requires_grad_(True)
    k2 = k0.clone().requires_grad_(True)
    v2 = v0.clone().requires_grad_(True)
    scale = D ** -0.25
    qt = _scatter_mean_landmarks((q2 * scale), qa, m)
    kt = _scatter_mean_landmarks((k2 * scale), ka, m)
    ref = nystrom_attention_reference(q2, k2, v2, num_landmarks=m, newton_iter=6,
                                      kappa_star=0.0, q_tilde=qt, k_tilde=kt)
    (ref * go).sum().backward()

    # dV is landmark-independent; dQ/dK include the straight-through landmark path
    torch.testing.assert_close(v1.grad, v2.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(q1.grad, q2.grad, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(k1.grad, k2.grad, rtol=5e-2, atol=5e-2)
