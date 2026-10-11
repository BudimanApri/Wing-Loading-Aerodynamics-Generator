"""Geometry measured from exported surface coordinates, with uncertainty notes.

This is a sampled surface envelope, not a CAD/topology reconstruction. No
pressure, face areas or forces are inferred by this module.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .geometry import WingGeometry
from .force_integration import InsufficientData, unit


@dataclass
class GeometryEstimate:
    geometry: WingGeometry
    table: pd.DataFrame
    metadata: dict


def detect_geometry(data, strips=96, span_axis=(0,1,0), chord_axis=(1,0,0),
                    origin=(0,0,0), upper_normal_sign=-1, positive_panel="left"):
    es, ex = unit(span_axis), unit(chord_axis)
    if abs(es @ ex) > 1e-6:
        raise ValueError("Detection chord/span axes must be perpendicular")
    if strips < 2:
        raise ValueError("At least two geometry strips required")
    points = data.vector()
    span = (points-origin) @ es
    x = points @ ex
    lo, hi = float(span.min()), float(span.max())
    if hi-lo <= 1e-12:
        raise InsufficientData("Geometry detection needs more than one span station")
    edges = np.linspace(lo,hi,strips+1)
    bins = np.clip(np.searchsorted(edges,span,side="right")-1,0,strips-1)
    order = np.argsort(bins,kind="stable")
    cuts = np.searchsorted(bins[order],np.arange(strips+1))
    records, sparse = [], 0
    for i in range(strips):
        ids = order[cuts[i]:cuts[i+1]]
        if len(ids)<4:
            sparse += 1
            continue
        local_x = x[ids]
        a,b = float(local_x.min()),float(local_x.max())
        if b-a <= 1e-10:
            sparse += 1
            continue
        station = float(np.median(span[ids]))
        # Use actual extrema, averaging only points at essentially identical x.
        tolerance = max(1e-10,(b-a)*1e-7)
        le = points[ids[local_x <= a+tolerance]].mean(axis=0)
        te = points[ids[local_x >= b-tolerance]].mean(axis=0)
        le += (station-(le-origin) @ es)*es
        te += (station-(te-origin) @ es)*es
        distinct = np.unique(local_x)
        max_gap = np.max(np.diff(distinct))/(b-a) if len(distinct)>1 else 1.
        records.append((station,le,te,len(ids),float(max_gap)))
    if len(records)<2:
        raise InsufficientData("Too few populated span sections to detect chord geometry")
    stations = np.array([r[0] for r in records])
    le = np.array([r[1] for r in records]); te = np.array([r[2] for r in records])
    # End intervals reuse their nearest measured section, not an invented taper.
    knots = np.unique(np.r_[lo,stations,hi,0. if lo<0<hi else stations[0]])
    interp = lambda array: np.column_stack([np.interp(knots,stations,array[:,j]) for j in range(3)])
    final_le,final_te = interp(le),interp(te)
    final_le += (knots-(final_le-origin) @ es)[:,None]*es
    final_te += (knots-(final_te-origin) @ es)[:,None]*es
    geometry = WingGeometry(knots,final_le,final_te,np.zeros(len(knots)),tuple(es),tuple(origin),upper_normal_sign,positive_panel)
    table = pd.DataFrame({"span_m":stations,"LE_x_m":le[:,0],"LE_y_m":le[:,1],"LE_z_m":le[:,2],
        "TE_x_m":te[:,0],"TE_y_m":te[:,1],"TE_z_m":te[:,2],"chord_m":np.linalg.norm(te-le,axis=1),
        "record_count": [r[3] for r in records],"largest_chord_sample_gap_fraction":[r[4] for r in records]})
    warnings = ["LE/TE and chord are estimated from sampled coordinate extrema; sparse mesh points and span averaging limit accuracy.",
                "Detected end sections reuse the nearest sampled envelope; the exact CAD tip/root edge is not reconstructed.",
                "Chord direction includes the measured LE–TE height difference; no separate twist is guessed.",
                "Coefficients and quarter-chord/spar moments using this geometry have an estimated reference geometry."]
    if sparse:
        warnings.append(f"{sparse} geometry strips lack sufficient chord sampling; their geometry is interpolated from populated sections")
    if table.largest_chord_sample_gap_fraction.max() > .2:
        warnings.append("Large gaps within chord sampling: inspect completeness or separate disconnected lifting surfaces")
    metadata = {"source":"automatic sampled-coordinate envelope","method":"Local min/max along configured chord axis, measured LE/TE heights, piecewise-linear span interpolation",
        "span_bounds_m":[lo,hi],"nominal_geometry_strip_width_m":(hi-lo)/strips,
        "populated_geometry_strips":len(records),"sparse_geometry_strips":sparse,
        "median_detected_chord_m":float(table.chord_m.median()),"reference_area_m2":geometry.area,
        "exact_CAD_geometry":False,"warnings":warnings}
    return GeometryEstimate(geometry,table,metadata)


def infer_pressure_sides(data, geometry, span_bins=96, chord_bins=60):
    """Infer normal orientation from coordinate heights for PRESSURE plots only.

    The inferred sign is never applied to forces or written into verified
    export conventions. Require two opposing normal populations and strong
    agreement with local height envelopes. Caps/ambiguous normals are excluded.
    """
    if not data.has("ax","ay","az"):
        raise InsufficientData("Automatic pressure-side identification needs area-vector directions or separate upper/lower files")
    s,xi,h = geometry.coordinates(data.vector())
    sbin = np.clip(((s-geometry.stations[0])/(geometry.stations[-1]-geometry.stations[0])*span_bins).astype(int),0,span_bins-1)
    xbin = np.clip((xi*chord_bins).astype(int),0,chord_bins-1)
    cell = sbin*chord_bins+xbin
    size = span_bins*chord_bins
    min_h,max_h = np.full(size,np.inf),np.full(size,-np.inf)
    np.minimum.at(min_h,cell,h); np.maximum.at(max_h,cell,h)
    residual = h-(min_h[cell]+max_h[cell])/2
    thickness = max_h[cell]-min_h[cell]
    _,_,_,normal = geometry.frame(s)
    area = data.vector(("ax","ay","az"))
    magnitude = np.linalg.norm(area,axis=1)
    projection = np.divide(np.einsum("ij,ij->i",area,normal),magnitude,
                           out=np.zeros(len(s)),where=magnitude>0)
    chord_scale = float(np.median(geometry.frame(geometry.stations)[1]))
    evidence = (np.abs(projection)>.35)&(np.abs(residual)>.2*thickness)&(thickness>chord_scale*1e-5)&(xi>.05)&(xi<.95)
    if evidence.sum()<20 or min((projection[evidence]>0).sum(),(projection[evidence]<0).sum()) < .05*evidence.sum():
        raise InsufficientData("Cannot establish both upper and lower surfaces automatically; supply explicit side labels or verify normals")
    agreement = float(np.mean(projection[evidence]*residual[evidence]>0))
    confidence = max(agreement,1-agreement)
    if confidence < .85:
        raise InsufficientData(f"Automatic side identification is ambiguous ({confidence:.1%} normal/height agreement); inspect geometry or provide side labels")
    sign = 1 if agreement>.5 else -1
    dot = projection*sign
    labels = np.where(dot>.1,"upper",np.where(dot<-.1,"lower","other"))
    return labels,{"source":"inferred normal/coordinate-height agreement for pressure interpolation only",
        "agreement_fraction":confidence,"evidence_records":int(evidence.sum()),
        "unclassified_records":int((labels=="other").sum()),
        "suggested_area_direction_diagnostic_only":"body_to_fluid" if sign==1 else "fluid_to_body",
        "force_convention_verified":False}
