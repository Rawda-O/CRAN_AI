# cran/policies/generalized_policy.py

"""Generalized policy.

New contribution:
- The policy takes **system parameters** (tau, Ps_max, Pr_max, optionally SI level)
  as part of the input vector, so one network can generalize across operating points.

Important rule:
- CSI noise must NOT be applied to these system parameters (handled in csi.py).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from cran.policies.df_policy import DFPolicy
from cran.policies.cf_policy import CFPolicy


class GeneralizedDFPolicy(DFPolicy):
    """Same as DFPolicy but expects inputs that include system params."""
    pass


class GeneralizedCFPolicy(CFPolicy):
    """Same as CFPolicy but expects inputs that include system params."""
    pass
