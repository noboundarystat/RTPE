"""Smoke tests: shapes, NaN-safety under missingness, and mask semantics. Run: pytest -q"""
import pytest
import torch

from temporal_encoding.models import TORCH_MODELS

B, F, T, D = 4, 5, 7, 8


def make_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, F, T, generator=g)
    time = torch.sort(torch.rand(B, T, generator=g) * 100, dim=1).values
    mask = torch.ones(B, T)
    mask[0, 5:] = 0                                   # padded subject
    value_mask = (torch.rand(B, F, T, generator=g) > 0.4).float()
    value_mask[1, 2, :] = 0                           # one feature never observed for subject 1
    return x * mask.unsqueeze(1), time, mask, value_mask


@pytest.mark.parametrize("name", list(TORCH_MODELS))
@pytest.mark.parametrize("train", [True, False])
def test_forward_backward_finite(name, train):
    model = TORCH_MODELS[name](n_features=F, d_model=D, num_layers=2, max_len=T)
    model.train(train)
    x, time, mask, value_mask = make_inputs()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        out = model(x, time, mask, value_mask)
        assert out.shape == (B,)
        assert torch.isfinite(out).all()
        if train:
            out.sum().backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("name", list(TORCH_MODELS))
def test_complete_data_path(name):
    model = TORCH_MODELS[name](n_features=F, d_model=D, num_layers=1, max_len=T).eval()
    x, time, mask, _ = make_inputs()
    with torch.no_grad():
        assert torch.isfinite(model(x, time, mask, None)).all()


@pytest.mark.parametrize("name", ["Proposed", "Classic", "Shortformer", "Spacetimeformer"])
def test_missing_values_are_ignored(name):
    """Whatever sits in a missing cell must not change the prediction."""
    model = TORCH_MODELS[name](n_features=F, d_model=D, num_layers=2, max_len=T).eval()
    x, time, mask, value_mask = make_inputs()
    x_alt = torch.where(value_mask.bool() & mask.bool().unsqueeze(1), x, torch.full_like(x, 123.0))
    with torch.no_grad():
        assert torch.allclose(model(x, time, mask, value_mask), model(x_alt, time, mask, value_mask), atol=1e-5)


def test_proposed_uses_real_time_but_classic_does_not():
    x, time, mask, value_mask = make_inputs()
    time_shifted = time * 1.7 + 3.0
    with torch.no_grad():
        prop = TORCH_MODELS["Proposed"](n_features=F, d_model=D, num_layers=1, max_len=T).eval()
        cls = TORCH_MODELS["Classic"](n_features=F, d_model=D, num_layers=1, max_len=T).eval()
        assert not torch.allclose(prop(x, time, mask, value_mask), prop(x, time_shifted, mask, value_mask), atol=1e-4)
        assert torch.allclose(cls(x, time, mask, value_mask), cls(x, time_shifted, mask, value_mask))
