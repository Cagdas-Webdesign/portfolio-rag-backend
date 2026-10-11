"""An isolated comparison of grounding-check models — a measuring instrument.

Nothing in ``src`` imports this package, and nothing here changes how the
service runs. It feeds two models the *same* grounding-check request — the
production prompt (`rag.verification`), the same passages, the same answer —
and judges each reply with the production parser against a gold label written
down before either model was asked.
"""
