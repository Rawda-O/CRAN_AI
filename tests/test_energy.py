import torch

from cran.physics.energy import ee_secondary


def test_energy_efficiency_finite_positive():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rs = torch.full((1024,), 1.0, device=device)
    ps = torch.full((1024,), 2.0, device=device)
    pr = torch.full((1024,), 3.0, device=device)
    ee = ee_secondary(rs, ps, pr, pc=0.5)
    assert torch.isfinite(ee).all()
    assert torch.all(ee > 0.0)
