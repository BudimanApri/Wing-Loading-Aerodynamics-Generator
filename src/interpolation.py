"""Pressure interpolation is deliberately separate from force integration."""
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError
from .force_integration import InsufficientData
from .pressure_analysis import pressure_coefficient
from .spanwise_analysis import Distribution, SpanIndex


def surface_labels(data, geometry, convention, explicit=None, normal_threshold=.1):
    if explicit is not None:
        labels = np.asarray(explicit)
        if labels.shape != (len(data),):
            raise ValueError("Surface labels must match records")
        return labels
    if data.has("surface"):
        raw = np.char.lower(data.columns["surface"].astype(str))
        if np.isin(raw, ["upper", "lower", "other"]).all():
            return raw
    if convention.area_orientation not in ("body_to_fluid", "fluid_to_body") or not convention.face_centered_verified:
        raise InsufficientData("Upper/lower identification requires explicit labels or verified face normals")
    if not data.has("ax", "ay", "az"):
        raise InsufficientData("No normals or explicit upper/lower labels")
    positions = data.vector()
    _, _, _, normal = geometry.frame(geometry.span(positions))
    area = data.vector(("ax", "ay", "az"))
    area *= 1 if convention.area_orientation == "body_to_fluid" else -1
    norm = np.linalg.norm(area, axis=1)
    dot = np.divide(np.einsum("ij,ij->i", area, normal), norm,
                    out=np.zeros(len(norm)), where=norm > 0)
    return np.where(dot > normal_threshold, "upper", np.where(dot < -normal_threshold, "lower", "other"))


def pressure_section(data, geometry, reference, station, neighborhood=.1, bins=100,
                     cp_source="recompute", labels=None, index=None):
    if labels is None:
        raise InsufficientData("Verified upper/lower labels required")
    if cp_source == "pressure":
        from .pressure_sampling import sample_pressure
        geometry.frame(float(station))
        if neighborhood <= 0 or bins < 2:
            raise ValueError("Positive interpolation neighborhood and at least two bins required")
        index = index or SpanIndex(data,geometry)
        ids = index.select(station-neighborhood/2,station+neighborhood/2,True)
        grid = (np.arange(bins)+.5)/bins
        result = sample_pressure(data,geometry,reference,labels,ids,grid,np.full(bins,station),"pressure")
        result.table["station_m"] = station
        result.metadata["neighborhood_m"] = neighborhood
        return result
    if neighborhood <= 0 or bins < 2:
        raise ValueError("Positive interpolation neighborhood and at least two bins required")
    geometry.frame(float(station))
    index = index or SpanIndex(data, geometry)
    ids = index.select(station-neighborhood/2, station+neighborhood/2, True)
    if len(ids) < 3:
        raise InsufficientData("Insufficient pressure points near requested station")
    s, xi, _ = geometry.coordinates(data.vector()[ids])
    cp = pressure_coefficient(data, reference, cp_source)[ids]
    grid = (np.arange(bins) + .5) / bins
    result, details = {}, {}
    for side in ("upper", "lower"):
        mask = (np.asarray(labels)[ids] == side) & (xi >= 0) & (xi <= 1)
        points = np.column_stack([xi[mask], s[mask]])
        values = cp[mask]
        if len(values) < 3:
            result[side] = np.full(bins, np.nan)
            details[side] = "Too few points"
            continue
        # Average duplicates solely for PRESSURE interpolation, never for forces.
        unique, inverse = np.unique(points, axis=0, return_inverse=True)
        averages = np.bincount(inverse, weights=values) / np.bincount(inverse)
        try:
            if np.ptp(unique[:, 1]) < 1e-12:
                if abs(unique[0, 1] - station) > 1e-10:
                    raise ValueError("Only one different span station available")
                order = np.argsort(unique[:, 0])
                result[side] = np.interp(grid, unique[order, 0], averages[order], left=np.nan, right=np.nan)
            else:
                interp = LinearNDInterpolator(unique, averages, fill_value=np.nan)
                result[side] = interp(grid, np.full(bins, station))
            details[side] = {"points": len(points), "unresolved_bins": int(np.isnan(result[side]).sum())}
        except (QhullError, ValueError) as exc:
            result[side] = np.full(bins, np.nan)
            details[side] = str(exc).splitlines()[0]
    delta = result["lower"] - result["upper"]
    table = pd.DataFrame({"station_m": station, "xi": grid, "cp_upper": result["upper"],
                          "cp_lower": result["lower"], "delta_cp": delta})
    # Only recomputed Cp has the confirmed pressure reference for dimensional use.
    if cp_source == "recompute":
        table["pressure_difference_Pa"] = delta * reference.q
        table["approx_vertical_pressure_load_N_per_m2"] = delta * reference.q
    return Distribution(table, {"method": "Linear pressure interpolation in (xi, signed span); no extrapolation outside convex hull.",
        "cp_source": cp_source, "neighborhood_m": neighborhood, "coverage": details,
        "force_status": "Interpolated pressure only; not conservative CFD force integration.",
        "approximation": "Pressure difference is conventional sectional normal loading; swept/twisted lift requires face-force projection."})
