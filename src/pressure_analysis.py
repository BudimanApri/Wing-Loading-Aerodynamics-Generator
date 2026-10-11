import numpy as np
from .force_integration import InsufficientData


def pressure_coefficient(data, reference, source="recompute"):
    if source == "exported":
        if not data.has("cp"):
            raise InsufficientData("No exported pressure coefficient")
        return np.asarray(data.columns["cp"])
    if not data.has("p"):
        raise InsufficientData("Static pressure is needed to recompute Cp")
    return (data.columns["p"] - reference.p_inf) / reference.q


def cp_reference_fit(data, max_points=50_000):
    """Diagnostic only: p = q_fit Cp + p0_fit; never set references from this."""
    if not data.has("p", "cp"):
        return None
    stride = max(1, len(data) // max_points)
    p, cp = data.columns["p"][::stride], data.columns["cp"][::stride]
    if np.ptp(cp) <= 1e-12:
        return None
    q, p0 = np.linalg.lstsq(np.column_stack([cp, np.ones(len(cp))]), p, rcond=None)[0]
    residual = p - (q * cp + p0)
    return {"q_fit_Pa": float(q), "p0_fit_Pa": float(p0),
            "rms_residual_Pa": float(np.sqrt(np.mean(residual**2))),
            "note": "Diagnostic fit only; exported Cp references require confirmation."}
