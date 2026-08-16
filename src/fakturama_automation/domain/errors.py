class AutomationError(RuntimeError):
    """Base error for workflow and gateway failures."""


class TransientUiError(AutomationError):
    """A UI condition that may succeed when retried."""


class ManualReviewRequired(AutomationError):
    """The workflow found ambiguity and must not choose automatically."""


class VerificationError(AutomationError):
    """Persisted or displayed values differ from expected values."""


class UnsupportedAutomation(AutomationError):
    """A safe implementation for a requested UI action is not available."""
