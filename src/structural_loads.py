"""Exact beam integrals for piecewise-constant strip densities."""
import numpy as np
import pandas as pd
from .spanwise_analysis import Distribution


def structural_loads(span_result):
    table = span_result.table
    output = []
    for panel in ("left", "right"):
        rows = table[table.panel == panel].copy()
        if rows.empty:
            continue
        sign = -1 if rows.span_m.mean() < 0 else 1
        lower = np.minimum(abs(rows.span_lower_m), abs(rows.span_upper_m)).to_numpy()
        upper = np.maximum(abs(rows.span_lower_m), abs(rows.span_upper_m)).to_numpy()
        order = np.argsort(lower)
        lower, upper = lower[order], upper[order]
        rows = rows.iloc[order]
        if not np.isclose(lower[0], 0.) or not np.allclose(lower[1:], upper[:-1], atol=1e-10):
            raise ValueError("Structural panels must cover a contiguous root-to-tip span")
        if (rows.face_count == 0).any() or rows[["lift_N_per_m", "pitch_Nm_per_m"]].isna().any().any():
            raise ValueError("Empty/unresolved strips prevent structural integration; reduce strip count")
        width = upper - lower
        load = rows.lift_N_per_m.to_numpy()
        torsion = rows.pitch_Nm_per_m.to_numpy()
        panel_force = load * width
        panel_first_moment = panel_force * (lower + upper) / 2
        shear = np.r_[np.cumsum(panel_force[::-1])[::-1], 0.]
        first = np.r_[np.cumsum(panel_first_moment[::-1])[::-1], 0.]
        edge = np.r_[lower, upper[-1]]
        bending = first - edge * shear
        torque = np.r_[np.cumsum((torsion*width)[::-1])[::-1], 0.]
        output.append(pd.DataFrame({"panel": panel, "distance_from_root_m": edge,
            "span_m": edge * sign, "shear_N": shear, "bending_Nm": bending, "internal_torque_Nm": torque}))
    if not output:
        raise ValueError("No valid panel loads")
    result = pd.concat(output, ignore_index=True)
    summaries = {}
    for panel, rows in result.groupby("panel"):
        summaries[panel] = {}
        for key in ("shear_N", "bending_Nm", "internal_torque_Nm"):
            peak = rows.loc[rows[key].abs().idxmax()]
            summaries[panel][key] = {"maximum_absolute": float(abs(peak[key])), "signed_value": float(peak[key]),
                                     "span_m": float(peak.span_m), "root_value": float(rows.iloc[0][key])}
    return Distribution(result, {"model": "Cantilever panels; piecewise-constant strip load; exact analytic bin integrals at edges.",
        "sign": "Positive shear and bending from positive lift; torsion right-hand about positive configured span axis on both panels.",
        "loads": "Aerodynamic forces only. Gravity, inertia, propulsion and concentrated masses are excluded.",
        "moment_axis_fraction": span_result.metadata["moment_axis_fraction"],
        "summary": summaries, "limitation": "Preliminary beam loads; not structural FEA or stress/failure assessment."})


def attach_structural_columns(span_result, beam_result):
    """Evaluate shear/bending/torque at strip midpoints from the exact bin model."""
    table = span_result.table.copy()
    for column in ("shear_N", "bending_Nm", "internal_torque_Nm"):
        table[column] = np.nan
    for panel, rows in table.groupby("panel"):
        beam = beam_result.table[beam_result.table.panel == panel].sort_values("distance_from_root_m")
        for idx, row in rows.iterrows():
            r = abs(row.span_m)
            edge_index = np.searchsorted(beam.distance_from_root_m.to_numpy(), r)
            tip_edge = beam.iloc[edge_index]
            h = tip_edge.distance_from_root_m - r
            table.loc[idx, "shear_N"] = tip_edge.shear_N + row.lift_N_per_m*h
            table.loc[idx, "bending_Nm"] = tip_edge.bending_Nm + tip_edge.shear_N*h + row.lift_N_per_m*h*h/2
            table.loc[idx, "internal_torque_Nm"] = tip_edge.internal_torque_Nm + row.pitch_Nm_per_m*h
    return Distribution(table, {**span_result.metadata, "structural": beam_result.metadata})
