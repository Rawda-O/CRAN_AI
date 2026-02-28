import torch

from cran.physics.channels import generate_gains
from cran.physics.rates_df import DFParams, secondary_rate_df
from cran.physics.impairments import HardwareConfig


def test_residual_si_degrades_rate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = generate_gains(512, "gaussian", device)
    p = DFParams()
    ps = torch.full((512,), 5.0, device=device)
    pr = torch.full((512,), 5.0, device=device)
    alpha = torch.full((512,), 0.5, device=device)

    rs_ideal = secondary_rate_df(g, ps, pr, alpha, p, hw=HardwareConfig(enable=True, residual_si_db=float("inf")))
    rs_bad = secondary_rate_df(g, ps, pr, alpha, p, hw=HardwareConfig(enable=True, residual_si_db=40.0))
    assert torch.mean(rs_bad) <= torch.mean(rs_ideal) + 1e-6
