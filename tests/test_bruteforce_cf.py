import torch

from cran.physics.channels import generate_gains
from cran.physics.rates_cf import CFParams, qos_A, qos_Q_cf
from cran.baselines.bruteforce_cf import solve_cf_bruteforce, CFGrid


def test_bruteforce_cf_feasible_and_nonnegative():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = generate_gains(128, "gaussian", device)
    p = CFParams()
    sol = solve_cf_bruteforce(g, ps_max=5.0, pr_max=5.0, params=p, grid=CFGrid(ps_steps=11, pr_steps=11))
    assert torch.isfinite(sol["rs"]).all()
    assert torch.all(sol["rs"] >= 0.0)

    A = qos_A(g["PP"], p)
    Q = qos_Q_cf(g, sol["ps"], sol["pr"])
    assert torch.all(Q <= A + 1e-6)
