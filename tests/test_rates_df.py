import torch

from cran.physics.channels import generate_gains
from cran.physics.rates_df import DFParams, secondary_rate_df


def test_df_rate_nonnegative_and_finite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = generate_gains(256, "gaussian", device)
    p = DFParams()
    ps = torch.full((256,), 1.0, device=device)
    pr = torch.full((256,), 1.0, device=device)
    alpha = torch.full((256,), 0.5, device=device)
    rs = secondary_rate_df(g, ps, pr, alpha, p)
    assert torch.all(rs >= 0.0)
    assert torch.isfinite(rs).all()
