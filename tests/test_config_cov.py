"""Exhaustive coverage of flash_nystrom.nystrom_config.NystromConfig.

Every field, every default, and every branch of __post_init__ (each assertion,
both the passing and the failing side), heavily parametrized.
"""
import math
import dataclasses
import pytest

from flash_nystrom.nystrom_config import NystromConfig

# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field,default", [
    ("num_landmarks", 64),
    ("newton_iter", 6),
    ("conv_kernel_size", 3),
    ("use_conv_residual", True),
    ("fast_dk2inv", True),
    ("kappa_star", 0.0),
    ("use_tc_pinv", False),
])
def test_field_default(field, default):
    assert getattr(NystromConfig(), field) == default


def test_all_defaults_construct():
    c = NystromConfig()
    assert isinstance(c, NystromConfig)


def test_is_dataclass():
    assert dataclasses.is_dataclass(NystromConfig)


def test_field_set_matches_source():
    names = {f.name for f in dataclasses.fields(NystromConfig)}
    assert names == {"num_landmarks", "newton_iter", "conv_kernel_size",
                     "use_conv_residual", "fast_dk2inv", "kappa_star", "use_tc_pinv"}

# --------------------------------------------------------------------------- #
# num_landmarks
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("m", list(range(1, 65)) + [100, 128, 256, 1024])
def test_num_landmarks_valid(m):
    assert NystromConfig(num_landmarks=m).num_landmarks == m


@pytest.mark.parametrize("m", [0, -1, -2, -64, -1000])
def test_num_landmarks_nonpositive_rejected(m):
    with pytest.raises(AssertionError, match="num_landmarks must be positive"):
        NystromConfig(num_landmarks=m)

# --------------------------------------------------------------------------- #
# newton_iter
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("j", list(range(1, 25)) + [30, 50, 100])
def test_newton_iter_valid(j):
    assert NystromConfig(newton_iter=j).newton_iter == j


@pytest.mark.parametrize("j", [0, -1, -3, -100])
def test_newton_iter_nonpositive_rejected(j):
    with pytest.raises(AssertionError, match="newton_iter must be positive"):
        NystromConfig(newton_iter=j)

# --------------------------------------------------------------------------- #
# conv_kernel_size
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("k", [0, 1, 3, 5, 7, 9, 11, 15, 31, 63])
def test_conv_kernel_size_valid(k):
    assert NystromConfig(conv_kernel_size=k).conv_kernel_size == k


@pytest.mark.parametrize("k", [-1, -2, -3, -10])
def test_conv_kernel_size_negative_rejected(k):
    with pytest.raises(AssertionError, match="conv_kernel_size must be non-negative"):
        NystromConfig(conv_kernel_size=k)


@pytest.mark.parametrize("k", [2, 4, 6, 8, 10, 12, 100])
def test_conv_kernel_size_even_positive_rejected(k):
    with pytest.raises(AssertionError, match="conv_kernel_size must be odd"):
        NystromConfig(conv_kernel_size=k)


def test_conv_kernel_size_zero_bypasses_odd_check():
    # 0 is allowed and must NOT trigger the odd assertion
    assert NystromConfig(conv_kernel_size=0).conv_kernel_size == 0

# --------------------------------------------------------------------------- #
# kappa_star
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("k", [0.0, 1e-6, 0.5, 1.0, 5.0, 1e3, 1e6, 1e12, 3.14])
def test_kappa_star_valid(k):
    assert NystromConfig(kappa_star=k).kappa_star == k


@pytest.mark.parametrize("k", [-1e-9, -0.5, -1.0, -1e3])
def test_kappa_star_negative_rejected(k):
    with pytest.raises(AssertionError, match="kappa_star must be finite"):
        NystromConfig(kappa_star=k)


@pytest.mark.parametrize("k", [math.inf, -math.inf, math.nan])
def test_kappa_star_nonfinite_rejected(k):
    with pytest.raises(AssertionError, match="kappa_star must be finite"):
        NystromConfig(kappa_star=k)


def test_kappa_star_int_zero_ok():
    assert NystromConfig(kappa_star=0).kappa_star == 0

# --------------------------------------------------------------------------- #
# boolean flags
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("flag", ["use_conv_residual", "fast_dk2inv", "use_tc_pinv"])
@pytest.mark.parametrize("val", [True, False])
def test_bool_flags(flag, val):
    c = NystromConfig(**{flag: val})
    assert getattr(c, flag) is val

# --------------------------------------------------------------------------- #
# combined / matrix construction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("m", [1, 16, 32, 64])
@pytest.mark.parametrize("j", [1, 6, 16])
@pytest.mark.parametrize("kappa", [0.0, 1e3])
@pytest.mark.parametrize("tc", [True, False])
def test_matrix_construct(m, j, kappa, tc):
    c = NystromConfig(num_landmarks=m, newton_iter=j, kappa_star=kappa, use_tc_pinv=tc)
    assert (c.num_landmarks, c.newton_iter, c.kappa_star, c.use_tc_pinv) == (m, j, kappa, tc)


@pytest.mark.parametrize("k", [0, 1, 3, 5])
@pytest.mark.parametrize("conv", [True, False])
def test_conv_combo(k, conv):
    c = NystromConfig(conv_kernel_size=k, use_conv_residual=conv)
    assert c.conv_kernel_size == k and c.use_conv_residual is conv


def test_equality():
    assert NystromConfig(num_landmarks=32) == NystromConfig(num_landmarks=32)
    assert NystromConfig(num_landmarks=32) != NystromConfig(num_landmarks=16)


def test_repr_contains_fields():
    r = repr(NystromConfig(num_landmarks=17))
    assert "num_landmarks=17" in r


@pytest.mark.parametrize("m", [1, 32, 64])
def test_replace(m):
    base = NystromConfig()
    c = dataclasses.replace(base, num_landmarks=m)
    assert c.num_landmarks == m and base.num_landmarks == 64
