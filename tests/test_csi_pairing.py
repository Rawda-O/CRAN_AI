import torch

from cran.physics.channels import generate_gains
from cran.physics.csi import ImperfectCSIConfig, make_imperfect_pair


def test_pairing_shapes_and_nonnegativity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = generate_gains(1024, "gaussian", device)
    cfg = ImperfectCSIConfig(sigma=0.2, apply_to_links=("PP","PR","RP","SP"), clamp_min=0.0)
    g_hat, g_ref = make_imperfect_pair(g, cfg)

    assert set(g_hat.keys()) == set(g_ref.keys())
    for k in g_ref:
        assert g_hat[k].shape == g_ref[k].shape
        assert torch.all(g_hat[k] >= 0.0)
