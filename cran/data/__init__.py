"""Data + feature engineering.

Provides:
- On-the-fly batch generation for channel gains (multiple channel models + mixed).
- Robust pairing (imperfect input, perfect loss reference).
- Feature builder (gains -> tensor features; + system params for generalization).
"""
