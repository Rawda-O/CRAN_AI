# cran/baselines/__init__.py

"""Baselines (brute-force, heuristics) for CRAN-AI."""

from .bruteforce_cf import solve_cf_bruteforce, CFGrid
from .bruteforce_df import solve_df_bruteforce, DFGrid
