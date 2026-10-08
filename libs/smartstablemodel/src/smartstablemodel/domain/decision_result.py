from typing import Dict, Any, Optional, List

class DecisionResult:
    """Result object for rule evaluation.

    primary_alert: final chosen alert label or None
    confidence: 0..1 aggregated confidence
    contributing_factors: human-readable factor strings aiding explainability
    secondary_alerts: optional list of other noteworthy conditions
    window_size: number of recent events considered (context-aware version)
    """

    def __init__(self, warning_type: Optional[str],
                 label: str,
                 confidence: float,
                 contributing_factors: List[str],
                 severity: float = 0.0,
                 secondary_alerts: Optional[List[str]] = None,
                 window_size: Optional[int] = None,
                 debug: Optional[Dict[str, Any]] = None):
        self.warning_type = warning_type
        self.severity = severity
        self.label = label
        self.confidence = confidence
        self.contributing_factors = contributing_factors
        self.secondary_alerts = secondary_alerts or []
        self.debug = debug  # optional deep trace

    def as_dict(self) -> Dict[str, Any]:
        base = {
            "warning_type": self.warning_type,
            "label": self.label,
            "confidence": self.confidence,
            "contributing_factors": self.contributing_factors,
            "secondary_alerts": self.secondary_alerts,
        }
        if self.debug:
            base["debug"] = self.debug
        return base