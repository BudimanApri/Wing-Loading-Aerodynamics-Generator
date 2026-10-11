import numpy as np
import pandas as pd
from .force_integration import unit
from .spanwise_analysis import Distribution, SpanIndex


def chordwise_distribution(data, force, geometry, reference, station, width=.05, bins=100, index=None):
    if width <= 0 or not np.isfinite(width) or bins < 1:
        raise ValueError("Positive strip width and bin count required")
    index = index or SpanIndex(data, geometry)
    lower = max(station - width / 2, geometry.stations[0])
    upper = min(station + width / 2, geometry.stations[-1])
    if lower >= upper or station < geometry.stations[0] or station > geometry.stations[-1]:
        raise ValueError("Station lies outside geometry")
    selected = index.select(lower, upper, include_upper=upper == geometry.stations[-1])
    if not len(selected):
        raise ValueError("No CFD faces in the selected strip; increase width")
    effective_width = upper - lower
    points, f = data.vector()[selected], force[selected]
    s, xi, _ = geometry.coordinates(points)
    # End/tip caps and small overhangs still contribute to strip totals.
    outside = (xi < 0) | (xi > 1)
    ix = np.clip(np.floor(xi * bins).astype(int), 0, bins - 1)
    edges = np.linspace(0, 1, bins + 1)
    centers = (edges[1:] + edges[:-1]) / 2
    _, c, ec, _ = geometry.frame(float(station))
    count = np.bincount(ix, minlength=bins)
    sums = lambda v: np.bincount(ix, weights=v, minlength=bins)
    L, D = sums(f @ unit(reference.lift)), sums(f @ unit(reference.drag))
    moment = np.cross(points - geometry.axis_points(s, .25), f) @ geometry.span_axis
    M = sums(moment)
    table = pd.DataFrame({"station_m": station, "xi": centers,
        "xi_lower": edges[:-1], "xi_upper": edges[1:], "dx_m": np.diff(edges) * c,
        "strip_width_m": effective_width, "face_count": count,
        "load_N_per_m2": np.where(count > 0, L / effective_width / (np.diff(edges) * c), np.nan),
        "bin_lift_N_per_m": L / effective_width, "bin_drag_N_per_m": D / effective_width,
        "bin_pitch_Nm_per_m": M / effective_width,
        "cumulative_lift_N_per_m": np.cumsum(L) / effective_width})
    lift, drag, pitch = L.sum()/effective_width, D.sum()/effective_width, M.sum()/effective_width
    force_per_span = f.sum(axis=0) / effective_width
    cp_denominator = float(np.cross(ec, force_per_span) @ geometry.span_axis) * c
    cp = .25 + pitch / cp_denominator if abs(cp_denominator) > 1e-10 * max(1., np.linalg.norm(force_per_span) * c) else None
    metadata = {"station_m": float(station), "strip_lower_m": float(lower), "strip_upper_m": float(upper),
        "requested_width_m": width, "effective_width_m": float(effective_width), "chord_m": float(c),
        "lift_N_per_m": float(lift), "drag_N_per_m": float(drag), "pitch_Nm_per_m": float(pitch),
        "cl": float(lift/reference.q/c), "cd": float(drag/reference.q/c), "cm": float(pitch/reference.q/c**2),
        "center_of_pressure_xi": None if cp is None else float(cp),
        "center_of_pressure_note": "Effective chord-line estimate at station; includes all selected forces. Undefined for vanishing moment-shift denominator; can lie outside chord.",
        "moment_reference": "Local quarter chord at each face span; right-hand positive span axis.",
        "force_conservation_error_N": float(abs(L.sum() - (f @ unit(reference.lift)).sum())),
        "outside_chord_faces_assigned_to_edge_bins": int(outside.sum()),
        "empty_bins": int((count == 0).sum()),
        "method": "Finite strip force integration, face-center assignment; no geometric clipping.",
        "load_note": "Projected force per strip width per local chord bin length; swept/twisted geometry remains a finite-strip average, not interpolated pressure."}
    return Distribution(table, metadata)
