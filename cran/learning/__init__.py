"""Learning engine (training, losses, metrics).

Phase 7 supports core contributions:
- Multi-objective training (rate + QoS + energy + outage)
- Robust imperfect-CSI training (paired input/target physics)
- Risk-controlled QoS (CVaR / chance margin hooks)
- GPU-first training with AMP (mixed precision) + device assertions
"""
