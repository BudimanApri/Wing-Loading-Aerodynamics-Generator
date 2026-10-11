"""Coordinate-backed pressure sampling, independent of face-force integration."""
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError
from .force_integration import InsufficientData
from .spanwise_analysis import Distribution


def sample_pressure(data, geometry, reference, labels, ids, query_xi, query_span, cp_source="pressure"):
    if not data.has("p") and cp_source != "exported":
        raise InsufficientData("Static pressure is required")
    if cp_source == "exported" and not data.has("cp"):
        raise InsufficientData("Exported Cp is missing")
    labels = np.asarray(labels)
    if labels.shape != (len(data),):
        raise ValueError("Pressure-side labels must match records")
    ids = np.asarray(ids,dtype=int)
    if len(ids)<3:
        raise InsufficientData("Too few pressure samples near the requested location; increase the interpolation neighborhood")
    positions = np.column_stack([data.columns[k][ids] for k in ("x","y","z")])
    s,xi,_ = geometry.coordinates(positions)
    quantities = []
    keys = []
    if data.has("p"):
        quantities.append(data.columns["p"][ids]); keys.append("p")
    if cp_source == "exported":
        quantities.append(data.columns["cp"][ids]); keys.append("cp")
    elif cp_source == "recompute":
        quantities.append((data.columns["p"][ids]-reference.p_inf)/reference.q); keys.append("cp")
    values = np.column_stack(quantities)
    qx,qs = np.broadcast_arrays(np.asarray(query_xi,dtype=float),np.asarray(query_span,dtype=float))
    table = pd.DataFrame({"span_m":qs.ravel(),"xi":qx.ravel()})
    details = {}
    for side in ("upper","lower"):
        keep = (labels[ids]==side)&(xi>=0)&(xi<=1)
        points = np.column_stack([xi[keep],s[keep]])
        actual = values[keep]
        output = np.full((qx.size,len(keys)),np.nan)
        if len(points)>=3:
            unique,inverse = np.unique(points,axis=0,return_inverse=True)
            average = np.column_stack([np.bincount(inverse,weights=actual[:,j])/np.bincount(inverse) for j in range(len(keys))])
            try:
                if np.ptp(unique[:,1])<1e-12:
                    order = np.argsort(unique[:,0])
                    at_station = np.isclose(qs.ravel(),unique[0,1],atol=1e-10,rtol=0)
                    for j in range(len(keys)):
                        output[at_station,j] = np.interp(qx.ravel()[at_station],unique[order,0],average[order,j],left=np.nan,right=np.nan)
                else:
                    output = LinearNDInterpolator(unique,average,fill_value=np.nan)(qx.ravel(),qs.ravel())
                details[side] = {"points":len(points),"unresolved_queries":int(np.isnan(output[:,0]).sum())}
            except QhullError:
                details[side] = "Degenerate local point cloud; no extrapolation performed"
        else:
            details[side] = "Too few surface samples"
        for j,key in enumerate(keys):
            table[f'{key}_{side}'] = output[:,j]
    # Stable export schema; unavailable Cp never gets manufactured.
    for side in ("upper","lower"):
        if f'cp_{side}' not in table:
            table[f'cp_{side}'] = np.nan
    table["delta_cp"] = table.cp_lower-table.cp_upper
    if "p_upper" in table:
        table.rename(columns={"p_upper":"pressure_upper_Pa","p_lower":"pressure_lower_Pa"},inplace=True)
        table["pressure_difference_Pa"] = table.pressure_lower_Pa-table.pressure_upper_Pa
        table["approx_vertical_pressure_load_N_per_m2"] = table.pressure_difference_Pa
    return Distribution(table,{"method":"Linear interpolation of measured surface pressure in (xi, signed span), no convex-hull extrapolation",
        "cp_source":cp_source,"coverage":details,"force_status":"Pressure sampling only; no face force summation",
        "pressure_difference_note":"Lower minus upper pressure; constant pressure reference cancels. This is approximate conventional sectional pressure loading, not a complete 3D force density."})


def spanwise_pressure_trace(data,geometry,reference,labels,fraction=.25,neighborhood=.08,count=80,cp_source="pressure",panel="both"):
    if not 0<=fraction<=1 or neighborhood<=0 or count<2:
        raise ValueError("Chord fraction [0,1], positive neighborhood and at least two span samples required")
    span,xi,_ = geometry.coordinates(data.vector())
    ids = np.flatnonzero(np.abs(xi-fraction)<=neighborhood/2)
    if len(ids)<3:
        raise InsufficientData("Too few samples near selected chord location; increase the chord neighborhood")
    lo,hi = geometry.stations[[0,-1]]
    if panel in ("left","right"):
        if panel==geometry.positive_panel:
            lo=max(lo,0.)
        else:
            hi=min(hi,0.)
    if lo>=hi:
        raise ValueError("Selected panel has no span extent")
    stations = np.linspace(lo,hi,count)
    result = sample_pressure(data,geometry,reference,labels,ids,np.full(count,fraction),stations,cp_source)
    result.table["panel"] = np.where(stations<0,"right" if geometry.positive_panel=="left" else "left",geometry.positive_panel)
    tip = np.where(stations<0,abs(geometry.stations[0]),abs(geometry.stations[-1]))
    result.table["normalized_semispan"] = np.divide(stations,tip,out=np.full(count,np.nan),where=tip>0)
    result.metadata.update(chord_location_xi=fraction,chord_interpolation_neighborhood_xi=neighborhood,
        selection="Pressure/loading trace at a fixed local chord fraction; not whole-chord spanwise lift in N/m")
    return result
