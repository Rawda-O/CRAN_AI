import torch

from cran.physics.channels import generate_gains
from cran.physics.rates_cf import CFParams, secondary_rate_cf


def test_cf_rate_nonnegative_and_finite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = generate_gains(256, "gaussian", device)
    p = CFParams()
    ps = torch.full((256,), 1.0, device=device)
    pr = torch.full((256,), 1.0, device=device)
    rs = secondary_rate_cf(g, ps, pr, p)
    assert torch.all(rs >= 0.0)
    assert torch.isfinite(rs).all()
