"""White-box risk rules."""

from .paper import (
    BrokerAwareOrderRiskPolicy,
    CLIENT_ORDER_ID_PREFIX,
    OrderValidationIssue,
    OrderValidationResult,
    build_client_order_id,
    extract_client_order_id,
    validate_client_order_id,
)
from .regime import (
    FORWARD_REGIME_LABELS,
    REGIME_LABELS,
    BenchmarkForwardRegimeLabeler,
    BenchmarkTrendDrawdownRegimeDetector,
    BenchmarkTrendDrawdownVolRegimeDetector,
    BenchmarkTrendDrawdownVolCrossAssetRegimeDetector,
    RegimeDetector,
    RegimeGatePolicy,
    apply_regime_gate,
    build_regime_confusion_matrix,
)

__all__ = [
    "BrokerAwareOrderRiskPolicy",
    "CLIENT_ORDER_ID_PREFIX",
    "OrderValidationIssue",
    "OrderValidationResult",
    "build_client_order_id",
    "extract_client_order_id",
    "validate_client_order_id",
    "REGIME_LABELS",
    "FORWARD_REGIME_LABELS",
    "BenchmarkForwardRegimeLabeler",
    "BenchmarkTrendDrawdownRegimeDetector",
    "BenchmarkTrendDrawdownVolRegimeDetector",
    "BenchmarkTrendDrawdownVolCrossAssetRegimeDetector",
    "RegimeDetector",
    "RegimeGatePolicy",
    "apply_regime_gate",
    "build_regime_confusion_matrix",
]
