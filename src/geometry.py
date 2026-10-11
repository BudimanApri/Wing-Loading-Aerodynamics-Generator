"""User-defined 3D section geometry; span coordinates are signed from root."""
from dataclasses import dataclass
import numpy as np
from .force_integration import unit


@dataclass
class WingGeometry:
    stations: np.ndarray
    leading_edges: np.ndarray
    trailing_edges: np.ndarray
    twists_deg: np.ndarray
    span_axis: tuple = (0., 1., 0.)
    origin: tuple = (0., 0., 0.)
    upper_normal_sign: int = 1
    positive_panel: str = "right"

    def __post_init__(self):
        self.stations = np.asarray(self.stations, dtype=float)
        self.leading_edges = np.asarray(self.leading_edges, dtype=float)
        self.trailing_edges = np.asarray(self.trailing_edges, dtype=float)
        self.twists_deg = np.asarray(self.twists_deg, dtype=float)
        n = len(self.stations)
        if n < 2 or self.leading_edges.shape != (n, 3) or self.trailing_edges.shape != (n, 3) or self.twists_deg.shape != (n,):
            raise ValueError("Geometry needs at least two stations with LE, TE and twist")
        if not all(np.isfinite(v).all() for v in (self.stations, self.leading_edges, self.trailing_edges, self.twists_deg)):
            raise ValueError("Geometry must be finite")
        if np.any(np.diff(self.stations) <= 0):
            raise ValueError("Geometry stations must increase strictly")
        self.span_axis = unit(self.span_axis)
        if self.upper_normal_sign not in (-1, 1):
            raise ValueError("Upper normal sign must be +1 or -1 relative to chord cross span")
        if self.positive_panel not in ("left", "right"):
            raise ValueError("Positive span panel must be left or right")
        self.origin = np.asarray(self.origin, dtype=float)
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all():
            raise ValueError("Geometry origin must be finite")
        if not np.allclose((self.leading_edges - self.origin) @ self.span_axis, self.stations, atol=1e-6):
            raise ValueError("LE projections must match signed span stations")
        chord = self.trailing_edges - self.leading_edges
        if np.any(np.linalg.norm(chord, axis=1) <= 0) or np.any(np.abs(chord @ self.span_axis) > 1e-6):
            raise ValueError("Section chord must be nonzero and perpendicular to span axis")
        # Validate intermediate sections too (opposing endpoint chords can collapse).
        self.frame(np.linspace(self.stations[0], self.stations[-1], max(50, n * 5)))

    def span(self, positions):
        return (np.asarray(positions) - self.origin) @ self.span_axis

    def frame(self, stations):
        s = np.asarray(stations, dtype=float)
        if np.any(~np.isfinite(s)) or np.any(s < self.stations[0] - 1e-10) or np.any(s > self.stations[-1] + 1e-10):
            raise ValueError("Station outside supplied geometry; extrapolation is disabled")
        le = np.stack([np.interp(s, self.stations, self.leading_edges[:, i]) for i in range(3)], axis=-1)
        te = np.stack([np.interp(s, self.stations, self.trailing_edges[:, i]) for i in range(3)], axis=-1)
        cvec = te - le
        chord = np.linalg.norm(cvec, axis=-1)
        if np.any(chord <= 1e-12):
            raise ValueError("Local section chord collapses")
        ec = cvec / chord[..., None]
        angle = np.deg2rad(np.interp(s, self.stations, self.twists_deg))
        # Right-hand rotation around configured positive span axis.
        ec = ec * np.cos(angle)[..., None] + np.cross(self.span_axis, ec) * np.sin(angle)[..., None]
        normal = self.upper_normal_sign * np.cross(ec, self.span_axis)
        return le, chord, ec, normal

    def coordinates(self, positions):
        s = self.span(positions)
        le, chord, ec, normal = self.frame(s)
        xi = np.einsum("ij,ij->i", positions - le, ec) / chord
        height = np.einsum("ij,ij->i", positions - le, normal)
        return s, xi, height

    def axis_points(self, stations, fraction=.25):
        if not 0 <= fraction <= 1:
            raise ValueError("Axis fraction must lie in [0,1]")
        le, chord, ec, _ = self.frame(stations)
        return le + fraction * chord[..., None] * ec

    @property
    def area(self):
        # Trapezoid rule is exact for piecewise-linear aligned tapered sections.
        # General section chords use dense quadrature, documented in metadata.
        s = np.linspace(self.stations[0], self.stations[-1], 2001)
        return float(np.trapezoid(self.frame(s)[1], s))
