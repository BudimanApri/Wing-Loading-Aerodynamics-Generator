"""Data diagnostics and explicit comparisons, never automatic convention changes."""
import numpy as np
import pandas as pd
from .pressure_analysis import cp_reference_fit
from .spanwise_analysis import make_span_edges, spanwise_distribution

FORCE_INPUT_FIELDS = ("x", "y", "z", "p", "ax", "ay", "az", "tx", "ty", "tz")


def duplicate_records(data):
    """Review equal force-input rows; matching values do not establish topology.

    IDs, Cp and zone/source labels do not affect the physical force inputs.
    Verify numeric equality, rather than treating a row-hash match as proof.
    Return all members of each repeated group, including the first occurrence.
    """
    physical = [k for k in FORCE_INPUT_FIELDS if data.has(k)]
    frame = pd.DataFrame({k: data.columns[k] for k in physical})
    extra = frame.duplicated(keep="first")
    affected = frame.duplicated(keep=False)
    ids = np.flatnonzero(affected)
    result = pd.DataFrame({k:v[ids] for k,v in data.columns.items()})
    groups = pd.factorize(pd.MultiIndex.from_frame(frame.loc[affected]),sort=False)[0]
    result.insert(0,"duplicate_group",groups+1)
    result.insert(1,"export_record_number",ids+1)
    result.insert(2,"repeated_copy",extra.to_numpy()[ids])
    return result


def dataset_diagnostics(data, reference=None):
    warnings = []
    required = ("p", "ax", "ay", "az")
    missing = [k for k in required if not data.has(k)]
    if missing:
        warnings.append(f"Face force integration unavailable; missing {missing}")
    repeated = duplicate_records(data)
    duplicate_count = int(repeated.repeated_copy.sum())
    if duplicate_count:
        warnings.append(f"{duplicate_count} extra records have identical positions, pressure, area and shear. Matching values do not prove identical mesh faces; review the repeated-record table before force integration")
    ranges = {k: [float(np.min(data.columns[k])), float(np.max(data.columns[k]))]
              for k in ("x", "y", "z", "p", "cp") if data.has(k)}
    result = {"records": len(data), "available_variables": list(data.columns), "ranges": ranges,
              "exact_duplicate_records": duplicate_count,
              "duplicate_group_count": int(repeated.duplicate_group.nunique()),
              "affected_duplicate_records": len(repeated),
              "extra_duplicate_record_percent": 100*duplicate_count/len(data),
              "duplicate_examples": repeated.head(8).to_dict(orient="records"),
              "duplicate_identity_note": "Record IDs/Cp/zone labels are excluded from the force-input comparison. Without connectivity, equal values do not establish whether faces are physically identical.",
              "warnings": warnings}
    if data.has("id"):
        keys = {"id": data.columns["id"]}
        for key in ("zone", "source"):
            if data.has(key):
                keys[key] = data.columns[key]
        result["repeated_ids_within_source_zone"] = int(pd.DataFrame(keys).duplicated().sum())
        if result["repeated_ids_within_source_zone"]:
            warnings.append("Repeated record IDs within a source/zone; verify unique face identities")
    if data.has("zone"):
        zones = np.unique(data.columns["zone"].astype(str)).tolist()
        result["zones"] = zones
        if any("shadow" in z.lower() for z in zones):
            warnings.append("Wall-shadow zones detected; exclude duplicate physical walls before integration")
    else:
        warnings.append("No zone identifiers: wall/shadow provenance and surface completeness require export-side verification")
    if data.has("ax", "ay", "az"):
        area = data.vector(("ax", "ay", "az"))
        norm = np.linalg.norm(area, axis=1)
        result["zero_area_records"] = int((norm <= 0).sum())
        result["area_magnitude_quantiles_m2_unverified"] = np.quantile(norm, [0, .1, .5, .9, 1]).tolist()
        result["area_vector_sum_m2_unverified"] = area.sum(axis=0).tolist()
        result["closure_ratio_unverified"] = float(np.linalg.norm(area.sum(axis=0)) / max(norm.sum(), 1e-30))
        warnings.append("Area closure is diagnostic; open tips/root cuts and nodal averaging can affect the residual")
    fit = cp_reference_fit(data)
    result["cp_reference_fit"] = fit
    if reference is not None and data.has("p", "cp"):
        stride = max(1, len(data) // 50000)
        residual = data.columns["cp"][::stride] - (data.columns["p"][::stride] - reference.p_inf) / reference.q
        result["exported_vs_recomputed_cp_rms"] = float(np.sqrt(np.mean(residual**2)))
        if fit and abs(fit["q_fit_Pa"] / reference.q - 1) > .05:
            warnings.append(f"Exported Cp implies q near {fit['q_fit_Pa']:.6g} Pa; configured q is {reference.q:.6g} Pa. Check Fluent Reference Values")
    data.metadata["exact_duplicate_records"] = duplicate_count
    return result


def compare_reports(computed, reported):
    """Match report keys, force components, directions, selected walls and units."""
    rows = []
    for key, expected in reported.items():
        if key not in computed or expected is None or not np.isfinite(expected):
            continue
        actual = float(computed[key])
        rows.append({"quantity": key, "computed": actual, "Fluent": expected,
                     "difference": actual - expected,
                     "absolute_error": abs(actual - expected),
                     "relative_error_percent": 100*abs(actual - expected)/abs(expected) if abs(expected) > 1e-12 else np.nan})
    return pd.DataFrame(rows)


def convergence_diagnostics(data, force, geometry, reference, counts=(20, 40, 80, 160), axis_fraction=.25):
    from .structural_loads import structural_loads
    rows = []
    previous = {}
    for count in counts:
        result = spanwise_distribution(data, force, geometry, reference,
                                       make_span_edges(geometry, count), axis_fraction)
        row = {"requested_strips": count, "actual_strips": len(result.table),
               "empty_strips": result.metadata["empty_strips"],
               "lift_N": result.table.strip_lift_N.sum(),
               "conservation_error_N": result.metadata["conservation_absolute_error_N"]}
        try:
            beam = structural_loads(result)
            for panel, summary in beam.metadata["summary"].items():
                key = f"{panel}_root_bending_Nm"
                value = summary["bending_Nm"]["root_value"]
                row[key] = value
                row[f"{panel}_root_bending_change_percent"] = (100*abs(value-previous[key])/abs(previous[key])
                    if key in previous and abs(previous[key]) > 1e-12 else np.nan)
                previous[key] = value
        except ValueError:
            row["structural_status"] = "Unresolved/empty strips"
        rows.append(row)
    return pd.DataFrame(rows)


def chordwise_convergence(data, force, geometry, reference, station, width, counts=(25,50,100,200)):
    from .chordwise_analysis import chordwise_distribution
    from .spanwise_analysis import SpanIndex
    index = SpanIndex(data, geometry)
    rows = []
    for count in counts:
        result = chordwise_distribution(data, force, geometry, reference, station, width, count, index)
        populated = result.table.load_N_per_m2.dropna().to_numpy()
        rows.append({"chord_bins": count, "empty_bins": result.metadata["empty_bins"],
            "lift_N_per_m": result.metadata["lift_N_per_m"],
            "pitch_Nm_per_m": result.metadata["pitch_Nm_per_m"],
            "conservation_error_N": result.metadata["force_conservation_error_N"],
            "maximum_absolute_populated_load_N_per_m2": float(np.max(np.abs(populated))) if len(populated) else np.nan})
    return pd.DataFrame(rows)
