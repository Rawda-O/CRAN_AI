import torch

from cran.physics.outage import outage_hard, outage_smooth


def test_outage_bounds():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rs = torch.tensor([0.0, 0.1, 0.2, 1.0], device=device)
    th = 0.2
    hard = outage_hard(rs, th)
    smooth = outage_smooth(rs, th, k=10.0)
    assert hard.min().item() >= 0.0 and hard.max().item() <= 1.0
    assert smooth.min().item() >= 0.0 and smooth.max().item() <= 1.0
