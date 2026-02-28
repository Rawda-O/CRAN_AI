import torch

from cran.physics.channels import generate_gains
from cran.physics.rates_df import DFParams, qos_A, qos_Q_df
from cran.baselines.bruteforce_df import solve_df_bruteforce, DFGrid


def test_bruteforce_df_feasible_and_nonnegative():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = generate_gains(128, "gaussian", device)
    p = DFParams()
    sol = solve_df_bruteforce(g, ps_max=5.0, pr_max=5.0, params=p, grid=DFGrid(ps_steps=9, pr_steps=9, alpha_steps=5))
    assert torch.isfinite(sol["rs"]).all()
    assert torch.all(sol["rs"] >= 0.0)

    A = qos_A(g["PP"], p)
    Q = qos_Q_df(g, sol["ps"], sol["pr"], sol["alpha"])
    assert torch.all(Q <= A + 1e-6)  # baseline enforces feasibility
