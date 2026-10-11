"""Face-center strip assignment conserves selected forces exactly."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .force_integration import unit


class SpanIndex:
    def __init__(self, data, geometry):
        self.span = geometry.span(data.vector())
        self.order = np.argsort(self.span, kind="stable")
        self.sorted_span = self.span[self.order]

    def select(self, lower, upper, include_upper=False):
        tolerance = 16 * np.finfo(float).eps * max(1., abs(lower), abs(upper))
        lo = np.searchsorted(self.sorted_span, lower - tolerance, side="left")
        hi = np.searchsorted(self.sorted_span, upper + tolerance if include_upper else upper - tolerance,
                             side="right" if include_upper else "left")
        return self.order[lo:hi]


@dataclass
class Distribution:
    table: pd.DataFrame
    metadata: dict


def bin_indices(values, edges):
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Bin edges must be finite and strictly increasing")
    valid = (values >= edges[0]) & (values <= edges[-1])
    indices = np.searchsorted(edges, values, side="right") - 1
    indices = np.minimum(indices, len(edges) - 2)
    return indices, valid


def spanwise_distribution(data, force, geometry, reference, edges, axis_fraction=.25, index=None):
    edges = np.asarray(edges, dtype=float)
    positions = data.vector()
    span = geometry.span(positions) if index is None else index.span
    bins, valid = bin_indices(span, edges)
    if edges[0] < 0 and edges[-1] == 0:
        # Root-center faces belong to the positive-span panel, never both.
        valid &= span < 0
    if not valid.any():
        raise ValueError("No surface faces within selected span range")
    n = len(edges) - 1
    s = span[valid]
    points, f, ix = positions[valid], force[valid], bins[valid]
    widths, mid = np.diff(edges), (edges[1:] + edges[:-1]) / 2
    le, chord, ec, _ = geometry.frame(mid)
    axis = geometry.axis_points(s, axis_fraction)
    moments = np.cross(points - axis, f) @ geometry.span_axis
    lift, drag = f @ unit(reference.lift), f @ unit(reference.drag)
    sums = lambda v: np.bincount(ix, weights=v, minlength=n)
    L, D, M = sums(lift), sums(drag), sums(moments)
    counts = np.bincount(ix, minlength=n)
    denominator = reference.q * chord
    panel_tip = np.where(mid < 0, abs(geometry.stations[0]), abs(geometry.stations[-1]))
    normalized = np.divide(mid, panel_tip, out=np.full(n, np.nan), where=panel_tip > 0)
    # With no records, sectional state is unresolved, not a measured zero load.
    def resolved(value):
        return np.where(counts > 0, value, np.nan)
    table = pd.DataFrame({
        "span_m": mid, "span_lower_m": edges[:-1], "span_upper_m": edges[1:],
        "strip_width_m": widths, "normalized_semispan": normalized,
        "panel": np.where(mid < 0, "right" if geometry.positive_panel == "left" else "left", geometry.positive_panel), "chord_m": chord, "face_count": counts,
        "lift_N_per_m": resolved(L / widths), "drag_N_per_m": resolved(D / widths),
        "cl": resolved(L / widths / denominator), "cd": resolved(D / widths / denominator),
        "pitch_Nm_per_m": resolved(M / widths), "cm": resolved(M / widths / denominator / chord),
        "strip_lift_N": L, "strip_drag_N": D, "strip_pitch_Nm": M,
        "cumulative_lift_N": np.cumsum(L)})
    summed = np.array([sums(f[:, j]).sum() for j in range(3)])
    direct = f.sum(axis=0)
    total_direct = force.sum(axis=0)
    metadata = {"assignment": "Face centers; no face clipping or reconstruction at strip boundaries.",
                "moment_axis_fraction": axis_fraction,
                "moment_sign": "Right-hand about configured positive span axis on both panels.",
                "reference_point": "Local chord axis at EACH face span coordinate.",
                "selected_force_N": direct.tolist(), "all_surface_force_N": total_direct.tolist(),
                "strip_sum_force_N": summed.tolist(),
                "conservation_absolute_error_N": float(np.linalg.norm(summed - direct)),
                "outside_span_faces": int((~valid).sum()),
                "empty_strips": int((counts == 0).sum()),
                "coefficient_definition": "cl=L'/(q*c_mid), cd=D'/(q*c_mid), cm=M'/(q*c_mid^2).",
                "cumulative_direction": "Increasing signed span, negative-span tip towards positive-span tip.",
                "drag_note": "Force projection; induced/profile drag is not decomposed.",
                "scaling": "Exported faces only; no symmetry multiplier."}
    return Distribution(table, metadata)


def make_span_edges(geometry, count=80, panel="both", width=None):
    lower, upper = geometry.stations[[0, -1]]
    if panel in ("left", "right"):
        if panel == geometry.positive_panel:
            lower = max(0., lower)
        else:
            upper = min(0., upper)
    elif panel != "both":
        raise ValueError("Unknown panel")
    if lower >= upper or count < 1:
        raise ValueError("Selected panel has no span extent")
    crosses_root = lower < 0 < upper
    if width is not None:
        if width <= 0 or not np.isfinite(width):
            raise ValueError("Strip width must be finite and positive")
        def distances(length):
            result = np.arange(0.,length,width)
            tolerance = 32*np.finfo(float).eps*max(1.,length)
            if len(result)>1 and abs(result[-1]-length)<=tolerance:
                result[-1]=length
                return result
            return np.r_[result,length]
        if crosses_root:
            edges = np.r_[-distances(-lower)[::-1][:-1],distances(upper)]
        elif upper == 0:
            edges = -distances(-lower)[::-1]
        elif lower == 0:
            edges = distances(upper)
        else:
            edges = lower+distances(upper-lower)
    else:
        if crosses_root:
            # Build each panel from EXACT zero, rather than adding zero to an
            # almost symmetric full-span linspace. The latter creates an empty
            # root sliver when the measured tips differ by a few micrometres.
            total = max(2,int(count))
            negative_count = min(total-1,max(1,int(round(total*(-lower)/(upper-lower)))))
            positive_count = total-negative_count
            edges = np.r_[np.linspace(lower,0.,negative_count+1)[:-1],
                          np.linspace(0.,upper,positive_count+1)]
        else:
            edges = np.linspace(lower,upper,int(count)+1)
    return edges


def elliptical_reference(stations, geometry, integrated_lift):
    """Full symmetric elliptical wing only; not inferred for incomplete panels."""
    lower, upper = geometry.stations[[0, -1]]
    if not np.isclose(-lower, upper) or lower >= 0:
        raise ValueError("Elliptical comparison requires a symmetric full-span geometry")
    semi = upper
    s = np.asarray(stations)
    return 2 * integrated_lift / (np.pi * semi) * np.sqrt(np.clip(1 - (s / semi)**2, 0, 1))
