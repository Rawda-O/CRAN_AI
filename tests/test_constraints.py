import torch

from cran.physics.channels import generate_gains
from cran.physics.rates_df import DFParams, qos_A, qos_Q_df
from cran.physics.rates_cf import CFParams, qos_Q_cf, qos_A as qos_A_cf
from cran.physics.constraints import safety_scale_joint_df, safety_scale_joint_cf


def test_safety_layer_enforces_cf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = generate_gains(512, "gaussian", device)
    p = CFParams()
    A = qos_A_cf(g["PP"], p)
    # Intentionally violate QoS with large powers
    ps = torch.full((512,), 50.0, device=device)
    pr = torch.full((512,), 50.0, device=device)
    ps2, pr2 = safety_scale_joint_cf(g, ps, pr, A)
    Q2 = qos_Q_cf(g, ps2, pr2)
    assert torch.all(Q2 <= A + 1e-6)


def test_safety_layer_enforces_df():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = generate_gains(512, "gaussian", device)
    p = DFParams()
    A = qos_A(g["PP"], p)
    ps = torch.full((512,), 50.0, device=device)
    pr = torch.full((512,), 50.0, device=device)
    alpha = torch.full((512,), 0.9, device=device)
    ps2, pr2 = safety_scale_joint_df(g, ps, pr, alpha, A)
    Q2 = qos_Q_df(g, ps2, pr2, alpha)
    assert torch.all(Q2 <= A + 1e-6)
