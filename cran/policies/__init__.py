"""Policy networks.

Phase 7 introduces the AI-native layer:
- DF policy: outputs (Ps, Pr, alpha)
- CF policy: outputs (Ps, Pr)
- Generalized policy: includes system parameters (tau, power budgets) as inputs (new contribution)
- Robust wrapper: supports imperfect CSI inputs with perfect-CSI loss reference (new contribution)
"""
