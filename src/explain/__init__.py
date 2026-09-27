"""src.explain — explainability module using GNNExplainer."""

from src.explain.explainer import (
    FraudExplainer,
    explain_wallet,
    evaluate_explainer,
    get_explainer,
)

__all__ = [
    "FraudExplainer",
    "explain_wallet",
    "evaluate_explainer",
    "get_explainer",
]
