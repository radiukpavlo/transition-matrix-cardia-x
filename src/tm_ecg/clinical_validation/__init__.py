"""Audited physician-in-the-loop validation for CARDIA-X.

The package deliberately separates physician-response coding, benchmark coding,
scenario construction, and metric calculation.  The modules may share typed
records, but the physician coder never receives benchmark or assisted fields.
"""

from tm_ecg.clinical_validation.metrics import compute_cohen_kappa
from tm_ecg.clinical_validation.models import KappaResult

__all__ = ["KappaResult", "compute_cohen_kappa"]

