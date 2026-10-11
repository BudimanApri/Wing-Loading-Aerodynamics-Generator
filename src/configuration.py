"""Default reference template; physical export assumptions remain unverified."""
import numpy as np
import re


def suggested_aoa(label, fallback=2.):
    match=re.search(r"\baoa\s*[:=_]?\s*([+-]?\d+(?:\.\d+)?)",str(label),flags=re.IGNORECASE)
    return float(match.group(1)) if match else float(fallback)


def default_case(path, aoa_deg=2.):
    if not np.isfinite(aoa_deg):
        raise ValueError("AoA must be finite")
    alpha = np.deg2rad(aoa_deg)
    return {"paths": [str(path)], "sides": ["combined"], "rho": 1.225, "speed": 10.,
        "aoa_deg": float(aoa_deg), "operating_pressure_Pa": 101325., "p_inf": 0., "p_force_ref": 0.,
        "pressure_basis": "unconfirmed", "area_orientation": "unverified", "shear_direction": "unverified",
        "face_centered_verified": False, "pressure_basis_verified": False,
        "si_units_verified": False, "unique_wall_faces_verified": False,
        "freestream": [float(np.cos(alpha)), 0., float(-np.sin(alpha))],
        "lift": [float(-np.sin(alpha)), 0., float(-np.cos(alpha))],
        "geometry": None, "zones": [], "moment_reference_point": [0., 0., 0.], "S_ref_m2": 0., "c_ref_m": 0.,
        "coordinate_note": "+x aft, +z down, +y=z cross x; AoA nose-up positive; lift upward positive"}
