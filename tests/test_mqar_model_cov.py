"""Exhaustive coverage of paper.mqar.model (the MQAR model + backends).

Runs on CPU in fp32. The sdpa and nystrom_reference backends are pure torch;
the flash_nystrom backend routes to the pure-PyTorch reference on CPU tensors,
so its module builds and runs here too (kernels are covered by the GPU suite).
"""
import pytest
import torch
import torch.nn as nn

from paper.mqar.model import (
    SdpaAttention,
    NystromReferenceAttention,
    build_attention,
    ShortConvolution,
    BaseConv,
    ResidualSublayer,
    MQARModel,
)


def _x(B, N, dim):
    torch.manual_seed(B * 100 + N + dim)
    return torch.randn(B, N, dim)


# --------------------------------------------------------------------------- #
# SdpaAttention
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dim,heads", [(64, 1), (128, 2), (128, 4), (256, 8)])
@pytest.mark.parametrize("B,N", [(1, 32), (2, 64), (3, 16)])
def test_sdpa_forward_shape(dim, heads, B, N):
    attn = SdpaAttention(dim, heads)
    out = attn(_x(B, N, dim))
    assert out.shape == (B, N, dim) and torch.isfinite(out).all()


@pytest.mark.parametrize("dim,heads", [(64, 3), (100, 8), (128, 5)])
def test_sdpa_dim_not_divisible_raises(dim, heads):
    with pytest.raises(ValueError, match="not divisible"):
        SdpaAttention(dim, heads)


@pytest.mark.parametrize("causal", [True, False])
def test_sdpa_causal_flag(causal):
    attn = SdpaAttention(64, 2, causal=causal)
    assert attn.causal is causal
    out = attn(_x(2, 32, 64))
    assert out.shape == (2, 32, 64)


def test_sdpa_has_four_projections():
    attn = SdpaAttention(64, 2)
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        assert isinstance(getattr(attn, name), nn.Linear)
        assert getattr(attn, name).bias is None


# --------------------------------------------------------------------------- #
# NystromReferenceAttention
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dim,heads", [(64, 1), (128, 2), (128, 4)])
def test_nysref_forward_shape(dim, heads):
    attn = NystromReferenceAttention(dim, heads, num_landmarks=16, newton_iter=6)
    out = attn(_x(2, 64, dim))
    assert out.shape == (2, 64, dim) and torch.isfinite(out).all()


@pytest.mark.parametrize("dim,heads", [(64, 3), (100, 8)])
def test_nysref_dim_not_divisible_raises(dim, heads):
    with pytest.raises(ValueError, match="not divisible"):
        NystromReferenceAttention(dim, heads)


def test_nysref_conv_residual_creates_param():
    attn = NystromReferenceAttention(64, 2, num_landmarks=16,
                                     use_conv_residual=True, conv_kernel_size=3)
    assert isinstance(attn.conv_weight, nn.Parameter)
    assert attn.conv_weight.shape == (2, 3)
    assert attn.conv_kernel_size == 3
    out = attn(_x(2, 64, 64))
    assert out.shape == (2, 64, 64) and torch.isfinite(out).all()


@pytest.mark.parametrize("use_conv,ks", [(False, 3), (True, 0)])
def test_nysref_no_conv_when_disabled(use_conv, ks):
    attn = NystromReferenceAttention(64, 2, num_landmarks=16,
                                     use_conv_residual=use_conv, conv_kernel_size=ks)
    assert attn.conv_weight is None and attn.conv_kernel_size == 0


@pytest.mark.parametrize("kappa", [0.0, 1.0, 1e3])
def test_nysref_kappa(kappa):
    attn = NystromReferenceAttention(64, 2, num_landmarks=16, kappa_star=kappa)
    assert attn.kappa_star == kappa
    out = attn(_x(1, 64, 64))
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# build_attention
# --------------------------------------------------------------------------- #

def test_build_sdpa():
    assert isinstance(build_attention("sdpa", 64, 2), SdpaAttention)


def test_build_sdpa_causal_ok():
    m = build_attention("sdpa", 64, 2, causal=True)
    assert isinstance(m, SdpaAttention) and m.causal is True


def test_build_nystrom_reference():
    m = build_attention("nystrom_reference", 64, 2, num_landmarks=16)
    assert isinstance(m, NystromReferenceAttention)


def test_build_nystrom_reference_compile_wraps():
    # covers the torch.compile branch; do NOT call forward (avoids inductor on CI)
    m = build_attention("nystrom_reference_compile", 64, 2, num_landmarks=16)
    assert m is not None
    assert callable(m)


@pytest.mark.parametrize("backend", ["flash_nystrom", "flash_nystrom_tc"])
def test_build_flash_nystrom_backends(backend):
    from flash_nystrom import FlashNystromAttention
    m = build_attention(backend, 64, 2, num_landmarks=64)
    assert isinstance(m, FlashNystromAttention)


@pytest.mark.parametrize("backend", ["nystrom_reference", "flash_nystrom", "flash_nystrom_tc"])
def test_build_causal_rejected_for_nystrom(backend):
    with pytest.raises(ValueError, match="causal"):
        build_attention(backend, 64, 2, causal=True)


@pytest.mark.parametrize("backend", ["", "mha", "linear", "unknown", "Sdpa"])
def test_build_unknown_backend_raises(backend):
    with pytest.raises(ValueError, match="unknown backend"):
        build_attention(backend, 64, 2)


# --------------------------------------------------------------------------- #
# ShortConvolution / BaseConv
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dim", [16, 64, 128])
@pytest.mark.parametrize("N", [8, 32, 64])
@pytest.mark.parametrize("ks", [2, 3, 5])
def test_shortconv_preserves_length(dim, N, ks):
    conv = ShortConvolution(dim, ks)
    out = conv(_x(2, N, dim))
    assert out.shape == (2, N, dim)  # causal left-pad truncated back to N


def test_shortconv_kernel_size_stored():
    assert ShortConvolution(32, 4).kernel_size == 4


@pytest.mark.parametrize("dim,N,ks", [(64, 32, 3), (128, 16, 5), (32, 64, 2)])
def test_baseconv_shape(dim, N, ks):
    out = BaseConv(dim, ks)(_x(2, N, dim))
    assert out.shape == (2, N, dim) and torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# ResidualSublayer
# --------------------------------------------------------------------------- #

def test_residual_shape():
    out = ResidualSublayer(64, nn.Linear(64, 64))(_x(2, 16, 64))
    assert out.shape == (2, 16, 64)


def test_residual_adds_input():
    # a mixer that returns zeros -> output equals the residual input exactly
    class _Zero(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    x = _x(2, 8, 32)
    out = ResidualSublayer(32, _Zero())(x)
    torch.testing.assert_close(out, x)


# --------------------------------------------------------------------------- #
# MQARModel
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("backend", ["sdpa", "nystrom_reference"])
@pytest.mark.parametrize("dim,heads", [(64, 2), (128, 4)])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_model_forward_shape(backend, dim, heads, depth):
    vocab, N = 64, 64
    m = MQARModel(vocab, max_seq_len=128, dim=dim, depth=depth, heads=heads,
                  backend=backend, num_landmarks=64)
    idx = torch.randint(0, vocab, (2, N))
    out = m(idx)
    assert out.shape == (2, N, vocab) and torch.isfinite(out).all()


def test_model_encode_shape():
    m = MQARModel(64, 128, dim=64, depth=2, heads=2, backend="sdpa")
    h = m.encode(torch.randint(0, 64, (2, 64)))
    assert h.shape == (2, 64, 64)


@pytest.mark.parametrize("bad", ["xavier", "kaiming", "", "Normal"])
def test_model_bad_init_raises(bad):
    with pytest.raises(ValueError, match="init must be"):
        MQARModel(64, 128, backend="sdpa", init=bad)


def test_model_weight_tying():
    m = MQARModel(64, 128, dim=64, depth=2, backend="sdpa")
    assert m.head.weight is m.tok_emb.weight


def test_model_depth_alternation():
    m = MQARModel(64, 128, dim=64, depth=2, heads=2, backend="sdpa")
    assert isinstance(m.layers[0].mixer, BaseConv)       # even -> conv
    assert isinstance(m.layers[1].mixer, SdpaAttention)  # odd  -> attention


def test_model_depth1_is_conv_only():
    m = MQARModel(64, 128, dim=64, depth=1, backend="sdpa")
    assert len(m.layers) == 1 and isinstance(m.layers[0].mixer, BaseConv)


def test_model_hybrid_is_default_layout():
    m = MQARModel(64, 128, dim=64, depth=2, backend="sdpa")
    assert m.layer_layout == "hybrid"


def test_model_uniform_layout_all_mixer_no_conv():
    # figure-2 / DeltaNet-figure recipe: every layer is the mixer, no BaseConv
    m = MQARModel(64, 128, dim=64, depth=2, heads=2, backend="sdpa",
                  layer_layout="uniform")
    assert all(isinstance(l.mixer, SdpaAttention) for l in m.layers)
    assert not any(isinstance(l.mixer, BaseConv) for l in m.layers)


def test_model_uniform_depth1_is_mixer():
    m = MQARModel(64, 128, dim=64, depth=1, backend="sdpa", layer_layout="uniform")
    assert len(m.layers) == 1 and isinstance(m.layers[0].mixer, SdpaAttention)


@pytest.mark.parametrize("bad", ["alternating", "", "Uniform", "hybrid2"])
def test_model_bad_layer_layout_raises(bad):
    with pytest.raises(ValueError, match="layer_layout must be"):
        MQARModel(64, 128, backend="sdpa", layer_layout=bad)


@pytest.mark.parametrize("layout", ["hybrid", "uniform"])
def test_model_both_layouts_forward(layout):
    m = MQARModel(64, 128, dim=64, depth=2, heads=2, backend="sdpa", layer_layout=layout)
    out = m(torch.randint(0, 64, (2, 64)))
    assert out.shape == (2, 64, 64) and torch.isfinite(out).all()


def test_init_respects_no_reinit_mamba_dt_bias():
    # Mamba's dt_proj.bias is the inv_dt timescale init (marked _no_reinit); the
    # model's _init_weights sweep must NOT zero it. Zeroing it destroys the SSM
    # timescales and drops recall to chance (the Fig-4 Mamba bug).
    from paper.mqar.baselines import Mamba
    bare = Mamba(d_model=64)
    m = MQARModel(512, 256, dim=64, depth=2, heads=2, backend="mamba", layer_layout="uniform")
    dt = next(mod for name, mod in m.named_modules() if name.endswith("mamba.dt_proj"))
    assert not bool((dt.bias == 0).all()), "dt_proj.bias was zeroed by _init_weights"
    assert dt.weight.std().item() > 0.1, "dt_proj.weight was clobbered to normal(0.02)"
    # A_log / D structure preserved too
    mamba = next(mod for name, mod in m.named_modules() if name.endswith("mixer.mamba"))
    assert torch.equal(mamba.D.detach(), torch.ones_like(mamba.D)), "D was clobbered"


def test_init_no_reinit_does_not_affect_other_backends():
    # only Mamba marks _no_reinit; sdpa's projections must still get normal(0.02)
    m = MQARModel(512, 256, dim=64, depth=2, heads=2, backend="sdpa")
    q = next(mod for name, mod in m.named_modules() if name.endswith("q_proj"))
    assert 0.01 < q.weight.detach().std().item() < 0.04


# --------------------------------------------------------------------------- #
# position-embedding auto rule (figure-2: attention family only, uniform only)
# --------------------------------------------------------------------------- #

def test_pos_emb_auto_on_for_attention_uniform():
    # uniform layout removes BaseConv, so a permutation-equivariant mixer needs
    # position embeddings to see order at all (figure-2 configs.py:142)
    m = MQARModel(64, 128, dim=64, depth=2, heads=1, backend="sdpa",
                  layer_layout="uniform")
    assert m.use_pos_emb and isinstance(m.pos_emb, nn.Embedding)


@pytest.mark.parametrize("backend", ["hyena", "mamba"])
def test_pos_emb_auto_off_for_sequential_mixers_uniform(backend):
    # conv/recurrence mixers carry their own order signal -> no pos emb, which
    # keeps the validated DeltaNet-Fig4 runs bit-identical
    m = MQARModel(512, 128, dim=64, depth=2, heads=1, backend=backend,
                  layer_layout="uniform")
    assert not m.use_pos_emb and not hasattr(m, "pos_emb")


def test_pos_emb_auto_off_in_hybrid():
    # hybrid's BaseConv carries order; figure-2's models_repo sets
    # max_position_embeddings=0 for every mixer there
    m = MQARModel(64, 128, dim=64, depth=2, heads=1, backend="sdpa")
    assert not m.use_pos_emb


def test_pos_emb_explicit_override_beats_auto():
    m = MQARModel(64, 128, dim=64, depth=2, heads=1, backend="sdpa",
                  layer_layout="uniform", use_pos_emb=False)
    assert not m.use_pos_emb
    m2 = MQARModel(64, 128, dim=64, depth=2, heads=1, backend="mamba",
                   layer_layout="uniform", use_pos_emb=True)
    assert m2.use_pos_emb


def test_pos_emb_uniform_attention_forward_shape():
    m = MQARModel(64, 128, dim=64, depth=2, heads=1, backend="sdpa",
                  layer_layout="uniform")
    out = m(torch.randint(0, 64, (2, 64)))
    assert out.shape == (2, 64, 64) and torch.isfinite(out).all()


@pytest.mark.parametrize("init", ["normal", "orthogonal"])
def test_model_init_schemes(init):
    m = MQARModel(64, 128, dim=64, depth=2, backend="sdpa", init=init)
    assert m.init == init
    out = m(torch.randint(0, 64, (2, 64)))
    assert torch.isfinite(out).all()


def test_model_pos_emb_option():
    m = MQARModel(64, 128, dim=64, depth=2, backend="sdpa", use_pos_emb=True)
    assert hasattr(m, "pos_emb") and isinstance(m.pos_emb, nn.Embedding)
    out = m(torch.randint(0, 64, (2, 64)))
    assert out.shape == (2, 64, 64)


def test_model_no_pos_emb_by_default():
    m = MQARModel(64, 128, dim=64, depth=2, backend="sdpa")
    assert not hasattr(m, "pos_emb")


def test_model_seq_len_exceeds_max_with_pos_emb_raises():
    m = MQARModel(64, max_seq_len=32, dim=64, depth=2, backend="sdpa", use_pos_emb=True)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        m(torch.randint(0, 64, (2, 64)))  # N=64 > max_seq_len=32


def test_model_long_seq_ok_without_pos_emb():
    # no pos emb -> no max_seq_len constraint
    m = MQARModel(64, max_seq_len=32, dim=64, depth=2, backend="sdpa", use_pos_emb=False)
    out = m(torch.randint(0, 64, (2, 64)))
    assert out.shape == (2, 64, 64)


@pytest.mark.parametrize("drop", [0.0, 0.1, 0.5])
def test_model_embed_dropout(drop):
    m = MQARModel(64, 128, dim=64, depth=2, backend="sdpa", embed_dropout=drop)
    assert isinstance(m.embed_drop, nn.Dropout) and m.embed_drop.p == drop


def test_model_backward_flows():
    m = MQARModel(64, 128, dim=64, depth=2, heads=2, backend="sdpa")
    idx = torch.randint(0, 64, (2, 64))
    m(idx).sum().backward()
    assert m.tok_emb.weight.grad is not None
    assert torch.isfinite(m.tok_emb.weight.grad).all()
