"""Local wing CFD engineering workbench. Launch: streamlit run app.py."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import streamlit as st

from src.fluent_parser import read_fluent, combine, inspect_file
from src.file_selection import require_file, split_local_inputs, folder_exports, folder_candidates
from src.configuration import default_case, suggested_aoa
from src.auto_geometry import detect_geometry, infer_pressure_sides
from src.pressure_sampling import spanwise_pressure_trace
from src.force_integration import Reference, ForceConvention, face_forces, integrate, wing_force_summary, unit, InsufficientData
from src.geometry import WingGeometry
from src.pressure_analysis import pressure_coefficient
from src.chordwise_analysis import chordwise_distribution
from src.spanwise_analysis import SpanIndex, make_span_edges, spanwise_distribution, elliptical_reference
from src.interpolation import surface_labels, pressure_section
from src.structural_loads import structural_loads, attach_structural_columns
from src.validation import dataset_diagnostics, compare_reports, convergence_diagnostics, chordwise_convergence, duplicate_records
from src.visualization import line_plot, compare_distributions
from src.export import csv_bytes, metadata_text, excel_bytes, figure_bytes, to_parquet, bundle_results

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "data/configuration.json"
PAGES = ["CFD Data Manager", "Chordwise Analysis", "Spanwise Analysis", "Structural Loads", "Validation and Reporting"]
GEOMETRY_COLUMNS = ["span_m", "LE_x_m", "LE_y_m", "LE_z_m", "TE_x_m", "TE_y_m", "TE_z_m", "twist_deg"]
st.set_page_config(page_title="Wing Aerodynamic Analysis", page_icon="✈", layout="wide")


if "cases" not in st.session_state:
    if CONFIG.exists():
        st.session_state.cases = json.loads(CONFIG.read_text())
    else:
        st.session_state.cases = {name: default_case(ROOT.parent / name)
                                 for name in ("Front wing", "Rear wing") if (ROOT.parent / name).exists()}


def save_configuration():
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(metadata_text(st.session_state.cases))


def signature(paths):
    result=[]
    for raw_path in paths:
        path=require_file(raw_path)
        stamp=path.stat()
        result.append((str(path),stamp.st_size,stamp.st_mtime_ns))
    return tuple(result)


@st.cache_data(show_spinner=False,max_entries=32)
def discover_folder(path, stamps):
    return folder_exports(path)


@st.cache_resource(show_spinner="Opening cached CFD columns…", max_entries=12)
def open_case(paths, stamps, sides, zones=()):
    datasets = [read_fluent(p, ROOT / "data/processed") for p in paths]
    data = datasets[0] if len(datasets) == 1 else combine(datasets)
    explicit = None
    if all(side in ("upper", "lower", "other") for side in sides):
        explicit = np.concatenate([np.full(len(d), side) for d, side in zip(datasets, sides)])
    if zones:
        if not data.has("zone"):
            raise ValueError("Zone filtering requires exported zone identifiers")
        mask = np.isin(data.columns["zone"].astype(str), zones)
        if not mask.any():
            raise ValueError("No records match selected zones")
        data = data.subset(mask)
        data.metadata["selected_zones"] = list(zones)
        if explicit is not None:
            explicit = explicit[mask]
    return data, explicit, dataset_diagnostics(data)


def configured_data(case):
    return open_case(tuple(case["paths"]), signature(case["paths"]), tuple(case["sides"]), tuple(case.get("zones", [])))


def ref_from(case):
    return Reference(case["rho"], case["speed"], case["p_inf"], case["p_force_ref"],
                     tuple(case["lift"]), tuple(case["freestream"]))


def convention_from(case):
    values = {k: case[k] for k in (
        "face_centered_verified", "area_orientation", "shear_direction", "pressure_basis_verified",
        "si_units_verified", "unique_wall_faces_verified")}
    values["export_location"] = case.get("export_location", "unconfirmed")
    if values["export_location"] == "node":
        values["face_centered_verified"] = False
    return ForceConvention(**values)


def geometry_from(case):
    if case.get("geometry") and case.get("geometry_mode", "manual") == "manual":
        return WingGeometry(**case["geometry"])
    if case.get("geometry_mode") == "manual":
        raise InsufficientData("Save manual geometry or select Automatic from CFD coordinates")
    return detected_geometry(tuple(case["paths"]),signature(case["paths"]),tuple(case["sides"]),tuple(case.get("zones",[]))).geometry


@st.cache_resource(show_spinner="Detecting geometry from surface coordinates…", max_entries=12)
def detected_geometry(paths, stamps, sides, zones):
    data,_,_ = open_case(paths,stamps,sides,zones)
    return detect_geometry(data)


def geometry_metadata(case):
    if case.get("geometry") and case.get("geometry_mode", "manual") == "manual":
        return {"source":"manual user-supplied section geometry"}
    return detected_geometry(tuple(case["paths"]),signature(case["paths"]),tuple(case["sides"]),tuple(case.get("zones",[]))).metadata


@st.cache_resource(show_spinner="Identifying pressure surfaces…", max_entries=12)
def pressure_labels(paths, stamps, sides, settings):
    case = json.loads(settings)
    data,explicit,_ = open_case(paths,stamps,sides,tuple(case.get("zones",[])))
    geo,conv = geometry_from(case),convention_from(case)
    try:
        labels = surface_labels(data,geo,conv,explicit)
        return labels,{"source":"explicit labels or verified normals"}
    except InsufficientData:
        return infer_pressure_sides(data,geo)


def labels_from(case):
    return pressure_labels(tuple(case["paths"]),signature(case["paths"]),tuple(case["sides"]),metadata_text(case))


def show_geometry_source(case):
    info = geometry_metadata(case)
    if info["source"].startswith("automatic"):
        st.caption("Geometry detected from exported coordinates. Chord, coefficients and chord-axis moments use a sampled geometry estimate; manual CAD geometry remains an optional override.")


@st.cache_resource(show_spinner="Preparing surface forces…", max_entries=8)
def prepare_forces(paths, stamps, sides, settings):
    case = json.loads(settings)
    data, labels, diagnostics = open_case(paths, stamps, sides, tuple(case.get("zones", [])))
    ref, conv = ref_from(case), convention_from(case)
    if case["pressure_basis"] == "absolute" and data.has("p") and np.min(data.columns["p"]) < 0:
        raise InsufficientData("Negative exported pressure is inconsistent with an absolute-pressure declaration")
    forces = face_forces(data, ref, conv)
    return data, labels, diagnostics, ref, conv, forces


@st.cache_resource(show_spinner="Preparing span index…", max_entries=8)
def prepare(paths, stamps, sides, settings):
    case = json.loads(settings)
    data,labels,diagnostics,ref,conv,forces = prepare_forces(paths,stamps,sides,settings)
    geo=geometry_from(case)
    index=SpanIndex(data,geo)
    return data, labels, diagnostics, ref, conv, geo, index, forces


def prepared(case):
    return prepare(tuple(case["paths"]), signature(case["paths"]), tuple(case["sides"]), metadata_text(case))


def provenance(case, data, extra=None):
    return {"units": "SI; unit-bearing column names; coefficients dimensionless", "case_configuration": case,
            "q_inf_Pa": ref_from(case).q, "input_provenance": data.metadata,
            "validation_status": "Analytical tests passed; Fluent agreement requires matching external reports",
            "geometry_reference": geometry_metadata(case),
            **(extra or {})}


def downloads(result, name, case, data):
    meta = {**provenance(case, data), **result.metadata}
    cols = st.columns(3)
    cols[0].download_button("Download CSV", csv_bytes(result.table, meta), name + ".csv", "text/csv", key=name+"csv")
    cols[1].download_button("Download metadata", metadata_text(meta), name + ".json", "application/json", key=name+"meta")
    if cols[2].button("Prepare Excel", key=name+"xlsx_btn"):
        try:
            cols[2].download_button("Download Excel", excel_bytes(result.table, meta), name+".xlsx", key=name+"xlsx")
        except ImportError:
            st.warning("Install openpyxl for Excel export")


def show_plot(fig, key):
    st.plotly_chart(fig, width="stretch", key=key)
    with st.expander("Export this plot"):
        st.download_button("Interactive HTML", figure_bytes(fig), key+".html", "text/html", key=key+"html")
        format = st.selectbox("Static format", ["png", "svg"], key=key+"format")
        if st.button("Prepare static image", key=key+"static"):
            try:
                st.download_button("Download image", figure_bytes(fig, format), key+"."+format, key=key+"image")
            except Exception as exc:
                st.warning(f"Static export needs Kaleido and Chrome/Edge: {exc}")


def calculation_active(key, parameters, label):
    """Keep results visible for plot controls and download-triggered reruns."""
    token = hashlib.sha256(metadata_text(parameters).encode()).hexdigest()
    if st.button(label, type="primary", key=key+"run"):
        st.session_state[key] = token
    return st.session_state.get(key) == token


def show_wing_forces(data,forces,reference,case,key):
    result=wing_force_summary(forces,reference)
    available=result.metadata["total_available"]
    chosen=result.table[result.table.component==("total" if available else "pressure")].iloc[0]
    st.subheader("Total wing force" if available else "Integrated wing pressure force")
    if available:
        st.caption("Pressure + viscous forces over the entire selected wing/case. Panel and span-strip selections do not change this whole-wing summary.")
    else:
        st.caption("Pressure-only contribution over the entire selected wing/case; total requires verified viscous traction.")
    cards=st.columns(3)
    cards[0].metric("Lift [N]",f"{chosen.lift_N:.3f}")
    cards[1].metric("Drag [N]",f"{chosen.drag_N:.3f}")
    cards[2].metric("Resultant force [N]",f"{chosen.resultant_N:.3f}")
    components=st.columns(3)
    for i,axis in enumerate("xyz"):
        components[i].metric(f"F{axis} [N]",f"{chosen['F'+axis+'_N']:.3f}")
    with st.expander("Pressure / viscous breakdown and force export"):
        st.dataframe(result.table,width="stretch",hide_index=True)
        force_metadata={"case_configuration":case,"input_provenance":data.metadata,"q_inf_Pa":reference.q,
            "validation_status":"Fluent agreement requires matching external reports",**result.metadata}
        st.download_button("Download integrated wing forces",csv_bytes(result.table,force_metadata),
            "wing_force_summary.csv","text/csv",key=key+"wing_force_csv")
    return result


st.title("Wing Aerodynamic Analysis")
st.caption("ANSYS Fluent surface exports · pressure and face forces · preliminary wing beam loads")
with st.sidebar:
    page = st.radio("Analysis page", PAGES)
    choices = list(st.session_state.cases)
    selected = st.selectbox("Wing / CFD case", choices) if choices else None
    st.caption("+x aft · +z down · +y = z × x\nAoA: nose-up positive")
    st.caption("Pressure interpolation and integrated face forces are reported separately.")


def manager(selected):
    st.subheader("Sources and surface assignments")
    with st.expander("Add a wing or CFD case", expanded=not selected):
        paths = st.text_area("Local file or folder paths, one per line (best for large files)",
            placeholder="C:\\CFD\\exports\\wing.csv\nor browse a folder: C:\\CFD\\exports",key="local_source_inputs")
        items=[]
        source_error=None
        try:
            explicit,folders=split_local_inputs(paths)
            items.extend(explicit)
            candidates=[]
            for folder in folders:
                stamps=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in folder_candidates(folder))
                candidates.extend(discover_folder(folder,stamps))
            candidates=list(dict.fromkeys(candidates))
            if folders:
                if candidates:
                    picked=st.multiselect("Select CFD exports from the folder(s)",candidates,
                        format_func=lambda path: f"{Path(path).name} — {Path(path).parent}",key="folder_export_selection")
                    items.extend(picked)
                    st.caption("Select files belonging to one wing / CFD case. Import front and rear wings as separate cases.")
                else:
                    st.info("No tabular CFD exports with x/y/z headers found directly inside this folder. Choose the folder containing the exported files.")
        except (ValueError,OSError) as exc:
            source_error=str(exc)
            st.error(source_error)
        with st.form("add_case"):
            name = st.text_input("Case name (e.g. Front wing · AoA 2)")
            hint=inspect_file(items[0])["record_hint"] if items and source_error is None else "unknown"
            import_aoa=st.number_input("Case AoA [deg], nose-up positive",value=suggested_aoa(Path(items[0]).name) if items else 2.)
            import_location=st.selectbox("Export Location for this case",["unconfirmed","node","face_center"],
                index=1 if hint=="node" else 2 if hint in ("cell_center","face") else 0,
                format_func=lambda value:{"unconfirmed":"Unconfirmed","node":"Node","face_center":"Cell Center — boundary face values"}[value])
            uploads = st.file_uploader("Or upload tabular ASCII / CSV exports", accept_multiple_files=True)
            side_text = st.text_input("Surface labels, one per file, comma-separated", "combined")
            submitted = st.form_submit_button("Import case")
        if submitted:
            try:
                if source_error:
                    raise ValueError(source_error)
                for upload in uploads or []:
                    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(upload.name).name)
                    content = upload.getbuffer()
                    digest = hashlib.sha256(content).hexdigest()[:16]
                    dest = ROOT / "data/raw" / (digest+"_"+safe)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        dest.write_bytes(content)
                    items.append(str(dest))
                items=list(dict.fromkeys(items))
                sides = [v.strip().lower() for v in side_text.split(",")]
                if len(sides) == 1:
                    sides *= len(items)
                if not name.strip() or not items or name in st.session_state.cases:
                    raise ValueError("Provide a new case name and select at least one CFD file. A folder path needs a file selection above.")
                if len(sides) != len(items) or not set(sides) <= {"combined", "upper", "lower", "other"}:
                    raise ValueError("Use combined, upper, lower or other, one label per file")
                open_case(tuple(items), signature(items), tuple(sides))
                case = default_case(items[0],import_aoa)
                case.update(paths=items, sides=sides,export_location=import_location)
                st.session_state.cases[name] = case
                save_configuration()
                st.success("Case imported; select it in the sidebar")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    if not selected:
        return
    case = st.session_state.cases[selected]
    try:
        data, _, diag = configured_data(case)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    st.write(f"{len(data):,} records · {', '.join(data.columns)}")
    with st.expander("Edit source assignments / filter zones"):
        with st.form("sources_"+selected):
            sources = st.data_editor(pd.DataFrame({"path": case["paths"], "side": case["sides"]}),
                num_rows="dynamic", column_config={"side": st.column_config.SelectboxColumn(options=["combined", "upper", "lower", "other"])}, key="sources_table_"+selected)
            unfiltered, _, _ = open_case(tuple(case["paths"]), signature(case["paths"]), tuple(case["sides"]))
            all_zones = np.unique(unfiltered.columns["zone"].astype(str)).tolist() if unfiltered.has("zone") else []
            zones = st.multiselect("Zones to include (empty = all exported zones)", all_zones, default=case.get("zones", []))
            if st.form_submit_button("Save sources and zones"):
                try:
                    new_paths, new_sides = sources.path.tolist(), sources.side.tolist()
                    if not new_paths or not set(new_sides) <= {"combined", "upper", "lower", "other"}:
                        raise ValueError("At least one valid source and side assignment required")
                    new_paths=[str(require_file(p)) for p in new_paths]
                    open_case(tuple(new_paths), signature(new_paths), tuple(new_sides), tuple(zones))
                    case.update(paths=new_paths, sides=new_sides, zones=zones)
                    save_configuration()
                    st.rerun()
                except (ValueError, OSError, TypeError) as exc:
                    st.error(str(exc))
    st.dataframe(pd.DataFrame({k: v[:12] for k, v in data.columns.items()}), width="stretch")
    with st.expander("Input diagnostics and source metadata"):
        st.json({"diagnostics": diag, "source": data.metadata})
    if diag["exact_duplicate_records"]:
        with st.expander("Repeated physical records — review before force integration"):
            repeated = duplicate_records(data)
            st.write(f"{diag['duplicate_group_count']} matching groups; {diag['exact_duplicate_records']} extra copies ({diag['extra_duplicate_record_percent']:.5f}% of records).")
            st.caption("These rows match in coordinates, pressure, area-vector and shear components despite their different record IDs. This is separate from repeated coordinate columns. No records have been removed. Shared/coincident nodes or overlapping walls are possible; mesh connectivity/export settings are needed to establish the cause.")
            st.dataframe(repeated,width="stretch")
            st.download_button("Download repeated-record evidence",csv_bytes(repeated,
                {"input_provenance":data.metadata,"comparison_fields":"x,y,z,p,ax,ay,az,tx,ty,tz; record IDs and Cp excluded",
                 "finding":"Equal force-input values; mesh face identity is unresolved","original_data_modified":False}),
                "duplicate_records.csv", "text/csv", key="duplicate_evidence_"+selected)
    for warning in diag["warnings"]:
        st.warning(warning)
    if st.button("Convert cached columns to Parquet"):
        path = ROOT / "data/processed" / (hashlib.sha256(selected.encode()).hexdigest()[:16]+".parquet")
        try:
            to_parquet(data, path)
            st.success(f"Saved {path}")
        except ImportError:
            st.warning("Install pyarrow for Parquet export")
    st.subheader("Reference conditions and export conventions")
    st.caption("Set the CFD reference conditions and export conventions for this case. Both pressure references use the same gauge/absolute basis as the exported pressure column.")
    with st.form("references_"+selected):
        a, b, c = st.columns(3)
        rho = a.number_input("Density [kg/m³]", min_value=1e-8, value=float(case["rho"]), format="%.6f")
        speed = b.number_input("Speed [m/s]", min_value=1e-8, value=float(case["speed"]))
        aoa = c.number_input("AoA [deg], nose-up positive", value=float(case["aoa_deg"]))
        operating = a.number_input("Operating pressure [Pa]", value=float(case["operating_pressure_Pa"]))
        p_inf = b.number_input("Freestream pressure in export basis [Pa]", value=float(case["p_inf"]))
        p_ref = c.number_input("Force reference pressure in export basis [Pa]", value=float(case["p_force_ref"]))
        basis = st.selectbox("Exported pressure basis", ["unconfirmed", "gauge", "absolute"], index=["unconfirmed", "gauge", "absolute"].index(case["pressure_basis"]))
        use_aoa = st.checkbox("Derive directions from AoA in the confirmed +x-aft, +z-down frame", value=True)
        drag_input = st.text_input("Freestream/drag direction x,y,z (used when derive is unchecked)", ",".join(map(str, case["freestream"])))
        lift_input = st.text_input("Lift direction x,y,z (used when derive is unchecked)", ",".join(map(str, case["lift"])))
        st.caption("Derived drag=(cos α,0,−sin α), lift=(−sin α,0,−cos α). Vectors are normalized; lift and drag must be orthogonal. Operating pressure converts gauge to absolute only; it is not automatically subtracted from gauge pressure.")
        area_options = ["unverified", "body_to_fluid", "fluid_to_body"]
        shear_options = ["unverified", "on_body", "on_fluid"]
        orientation = a.selectbox("Exported area-vector direction", area_options, index=area_options.index(case["area_orientation"]))
        shear = b.selectbox("Exported wall shear traction acts", shear_options, index=shear_options.index(case["shear_direction"]))
        locations = ["unconfirmed", "node", "face_center"]
        location = st.selectbox("Fluent ASCII export Location", locations,
            index=locations.index(case.get("export_location", "face_center" if case["face_centered_verified"] else "unconfirmed")),
            format_func=lambda value: {"unconfirmed":"Unconfirmed", "node":"Node", "face_center":"Cell Center — boundary face values"}[value])
        face_ok = st.checkbox("Verified Cell Center boundary export: one record per face", case["face_centered_verified"], disabled=location!="face_center")
        pressure_ok = st.checkbox("Verified pressure basis and BOTH pressure references", case["pressure_basis_verified"])
        units_ok = st.checkbox("Verified units: m, m², Pa; shear is stress in Pa", case["si_units_verified"])
        unique_ok = st.checkbox("Verified unique selected walls, no wall-shadow duplicates", case["unique_wall_faces_verified"])
        S_ref = a.number_input("Reference area [m²], 0 = geometry integral ∫c ds", min_value=0., value=float(case["S_ref_m2"]))
        c_ref = b.number_input("Global moment reference chord [m], 0 = omit CM", min_value=0., value=float(case["c_ref_m"]))
        moment_origin = st.text_input("Global moment reference point x,y,z [m]", ",".join(map(str, case["moment_reference_point"])))
        if st.form_submit_button("Save reference conditions"):
            try:
                angle = np.deg2rad(aoa)
                drag = [np.cos(angle), 0, -np.sin(angle)] if use_aoa else list(map(float, drag_input.split(",")))
                lift = [-np.sin(angle), 0, -np.cos(angle)] if use_aoa else list(map(float, lift_input.split(",")))
                Reference(rho, speed, p_inf, p_ref, tuple(lift), tuple(drag))
                point = list(map(float, moment_origin.split(",")))
                if len(point) != 3 or not np.isfinite(point).all():
                    raise ValueError("Moment reference point needs three finite coordinates")
                case.update(rho=rho, speed=speed, aoa_deg=aoa, operating_pressure_Pa=operating,
                    p_inf=p_inf, p_force_ref=p_ref, pressure_basis=basis, freestream=drag, lift=lift,
                    area_orientation=orientation, shear_direction=shear, face_centered_verified=face_ok and location=="face_center", export_location=location,
                    pressure_basis_verified=pressure_ok and basis != "unconfirmed", si_units_verified=units_ok,
                    unique_wall_faces_verified=unique_ok, S_ref_m2=S_ref, c_ref_m=c_ref, moment_reference_point=point)
                save_configuration()
                st.success("References saved")
            except ValueError as exc:
                st.error(str(exc))
    updated_diag = dataset_diagnostics(data, ref_from(case))
    if updated_diag.get("cp_reference_fit"):
        fit = updated_diag["cp_reference_fit"]
        st.write(f"Configured q∞ = {ref_from(case).q:.6g} Pa · diagnostic Cp fit q = {fit['q_fit_Pa']:.6g} Pa")
    for warning in updated_diag["warnings"]:
        if "Cp implies" in warning:
            st.warning(warning)
    with st.expander("Integrated wing forces"):
        if calculation_active("manager_wing_force",case,"Calculate total wing force"):
            try:
                _,_,_,ref,_,forces=prepare_forces(tuple(case["paths"]),signature(case["paths"]),tuple(case["sides"]),metadata_text(case))
                show_wing_forces(data,forces,ref,case,"manager_")
            except InsufficientData as exc:
                st.error(str(exc))
    st.subheader("Wing geometry")
    mode = st.radio("Geometry source",["Automatic from CFD coordinates","Manual override"],
        index=1 if case.get("geometry") and case.get("geometry_mode","manual")=="manual" else 0, horizontal=True, key="geo_mode_"+selected)
    chosen_mode = "automatic" if mode.startswith("Automatic") else "manual"
    if chosen_mode != case.get("geometry_mode", "manual" if case.get("geometry") else "automatic"):
        case["geometry_mode"] = chosen_mode
        save_configuration()
    if chosen_mode == "automatic":
        estimate = detected_geometry(tuple(case["paths"]),signature(case["paths"]),tuple(case["sides"]),tuple(case.get("zones",[])))
        st.success(f"Using coordinate-derived geometry: signed span {estimate.geometry.stations[0]:.4f} to {estimate.geometry.stations[-1]:.4f} m; median detected chord {estimate.metadata['median_detected_chord_m']:.4f} m. No manual entry required.")
        with st.expander("Detected geometry and sampling limits"):
            st.dataframe(estimate.table, width="stretch")
            st.json(estimate.metadata)
        try:
            _,side_info = labels_from(case)
            st.caption(f"Pressure surface identification: {side_info['source']}")
        except InsufficientData as exc:
            st.warning(str(exc))
    st.caption("Manual override is optional. Enter actual LE/TE and twist when exact reference geometry is needed. Twist is right-hand about positive span, separate from nose-up-positive AoA.")
    geo = case.get("geometry")
    if geo:
        rows = [[s, *le, *te, t] for s, le, te, t in zip(geo["stations"], geo["leading_edges"], geo["trailing_edges"], geo["twists_deg"])]
        frame = pd.DataFrame(rows, columns=GEOMETRY_COLUMNS)
    else:
        frame = pd.DataFrame({k: pd.Series(dtype=float) for k in GEOMETRY_COLUMNS})
    with st.expander("Optional manual geometry override", expanded=chosen_mode=="manual"), st.form("geometry_"+selected):
        edited = st.data_editor(frame, num_rows="dynamic", width="stretch", key="geometry_table_"+selected)
        span_axis = st.text_input("Positive span axis x,y,z", ",".join(map(str, geo["span_axis"] if geo else [0,1,0])))
        origin = st.text_input("Span origin x,y,z [m]", ",".join(map(str, geo["origin"] if geo else [0,0,0])))
        positive_panel = st.selectbox("Positive span corresponds to physical panel", ["left", "right"], index=0 if not geo or geo.get("positive_panel") == "left" else 1)
        st.caption("With +x aft and +z down, +y points to aircraft left. Panel assignment is explicit.")
        upper_sign = st.selectbox("Upper-side normal relative to chord × positive span", [-1, 1], index=0 if not geo or geo.get("upper_normal_sign") == -1 else 1)
        if st.form_submit_button("Validate and save geometry"):
            try:
                values = edited.to_numpy(dtype=float)
                obj = WingGeometry(values[:,0], values[:,1:4], values[:,4:7], values[:,7],
                                   tuple(map(float, span_axis.split(","))), tuple(map(float, origin.split(","))), upper_sign, positive_panel)
                case["geometry"] = asdict(obj)
                case["geometry_mode"] = "manual"
                save_configuration()
                st.success(f"Geometry saved; section reference area ∫c ds = {obj.area:.6g} m²")
            except (ValueError, IndexError) as exc:
                st.error(str(exc))
    config_upload = st.file_uploader("Import complete case configuration JSON", type=["json"], key="config_import")
    if config_upload and st.button("Apply configuration to selected case"):
        try:
            replacement = json.loads(config_upload.getvalue())
            required = set(default_case(""))
            if not required <= set(replacement):
                raise ValueError(f"Missing configuration fields: {required-set(replacement)}")
            ref_from(replacement)
            if replacement["geometry"]:
                geometry_from(replacement)
            open_case(tuple(replacement["paths"]), signature(replacement["paths"]), tuple(replacement["sides"]))
            st.session_state.cases[selected] = replacement
            save_configuration()
            st.rerun()
        except (ValueError, OSError, KeyError, TypeError) as exc:
            st.error(str(exc))
    st.download_button("Export case configuration", metadata_text(case), "case_configuration.json", "application/json")


def chordwise_page(case):
    data, explicit, _ = configured_data(case)
    geo, ref, conv = geometry_from(case), ref_from(case), convention_from(case)
    show_geometry_source(case)
    index = SpanIndex(data, geo)
    method = st.radio("Calculation method", ["Pressure interpolation", "Finite strip force integration"], horizontal=True)
    station = st.number_input("Spanwise station y [m]",min_value=float(geo.stations[0]),max_value=float(geo.stations[-1]),value=float(np.clip(0.,geo.stations[0],geo.stations[-1])),format="%.4f")
    with st.expander("Compare additional span stations"):
        additional = st.text_input("Additional signed span stations [m], comma-separated", "")
    stations_text = str(station)+((','+additional) if additional.strip() else '')
    a, b, c = st.columns(3)
    width = a.number_input("Strip / interpolation neighborhood width [m]", min_value=.0001, value=.05, format="%.4f")
    bins = b.number_input("Chordwise bins", min_value=2, max_value=2000, value=100)
    cp_source = c.selectbox("Pressure quantity", ["pressure", "recompute", "exported"],
        format_func=lambda value: {"pressure":"Static pressure / Δp (no Cp reference needed)","recompute":"Recomputed Cp","exported":"Exported Cp (check reference values)"}[value])
    invert = st.checkbox("Invert Cp axis", True)
    component = st.selectbox("Force component", ["pressure", "viscous", "total"])
    if not calculation_active("chord_request", [case, stations_text, width, bins, cp_source, method, component], "Calculate chordwise results"):
        return
    stations = list(map(float, stations_text.split(",")))
    results, force_results = {}, {}
    labels = None
    try:
        labels,side_info = labels_from(case)
    except InsufficientData as exc:
        st.warning(str(exc))
    for i, station in enumerate(stations):
        pressure = None
        if labels is not None and (cp_source in ("pressure","exported") or case["pressure_basis_verified"]):
            pressure = pressure_section(data, geo, ref, station, width, int(bins), cp_source, labels, index)
            pressure.metadata["surface_identification"] = side_info
        elif cp_source == "recompute":
            st.warning("Recomputed Cp requires confirmed freestream pressure in the export basis")
        if method == "Pressure interpolation":
            if pressure is None:
                raise InsufficientData("Pressure interpolation needs verified side labels and pressure references, or exported Cp")
            result = pressure
        else:
            *_, forces = prepared(case)
            result = chordwise_distribution(data, forces.component(component), geo, ref, station, width, int(bins), index)
            force_results[f"station_{i}"] = result
            if pressure is not None:
                result.table = result.table.merge(pressure.table.drop(columns="station_m"), on="xi", how="left")
                result.metadata["pressure_interpolation"] = pressure.metadata
            else:
                for column in ("cp_upper", "cp_lower", "delta_cp", "pressure_difference_Pa"):
                    result.table[column] = np.nan
        result.metadata["force_component"] = component if method.startswith("Finite") else "none"
        name = f"station_{i}_{station:.6g}m"
        results[name] = result
        st.write(f"Station {station:.6g} m")
        st.json(result.metadata, expanded=False)
        st.dataframe(result.table, width="stretch")
        downloads(result, name, case, data)
        if pressure is not None:
            if cp_source == "pressure":
                show_plot(line_plot(pressure.table,"xi",["pressure_upper_Pa","pressure_lower_Pa"],title=f"Measured surface pressure [Pa] · {station:g} m"),name+"pressure")
                show_plot(line_plot(pressure.table,"xi",["approx_vertical_pressure_load_N_per_m2"],title="Pressure difference / approximate sectional loading [N/m²]"),name+"load")
            else:
                show_plot(line_plot(pressure.table, "xi", ["cp_upper", "cp_lower"], invert_y=invert, title=f"Cp · {station:g} m"), name+"cp")
                show_plot(line_plot(pressure.table, "xi", ["delta_cp"], title="Lower − upper Cp"), name+"delta")
    if force_results:
        show_plot(compare_distributions(force_results, "xi", "load_N_per_m2", "Chordwise projected load [N/m²]"), "chord_load")
        show_plot(compare_distributions(force_results, "xi", "cumulative_lift_N_per_m", "Cumulative sectional lift [N/m]"), "chord_cumulative")
    elif len(results) > 1:
        quantity = "approx_vertical_pressure_load_N_per_m2" if cp_source=="pressure" else "delta_cp"
        show_plot(compare_distributions(results,"xi",quantity,"Pressure loading comparison"),"chord_compare")
    st.download_button("Download all stations with metadata", bundle_results(results, provenance(case, data)), "chordwise_results.zip")


def discretization(case):
    a, b, c = st.columns(3)
    panel = a.selectbox("Panel", ["both", "left", "right"])
    count = b.number_input("Spanwise strip count", min_value=1, max_value=2000, value=80)
    mode = c.selectbox("Discretization", ["count", "width"])
    width = st.number_input("Strip width [m]", min_value=.0001, value=.1) if mode == "width" else None
    component = st.selectbox("Aerodynamic force component", ["pressure", "viscous", "total"])
    return panel, int(count), width, component


def spanwise_chord_location_page(case):
    data,_,_ = configured_data(case)
    geo,ref = geometry_from(case),ref_from(case)
    a,b,c = st.columns(3)
    fraction = a.number_input("Chord location x/c",min_value=0.,max_value=1.,value=.25,step=.01)
    panel = b.selectbox("Panel",["both","left","right"])
    count = c.number_input("Spanwise sample count",min_value=2,max_value=2000,value=80)
    with st.expander("Pressure sampling settings"):
        neighborhood = st.number_input("Chord interpolation neighborhood Δ(x/c)",min_value=.001,max_value=1.,value=.08,step=.01)
        quantity = st.selectbox("Pressure quantity",["pressure","recompute","exported"],
            format_func=lambda value: {"pressure":"Static pressure / Δp","recompute":"Recomputed Cp","exported":"Exported Cp (reference mismatch flagged)"}[value])
        normalized = st.checkbox("Use signed normalized semispan",False)
    st.caption("This samples upper/lower pressure and their difference along a chosen chord fraction. It gives approximate sectional pressure loading in N/m². Whole-chord lift/drag in N/m is available using the other spanwise mode.")
    if not calculation_active("span_pressure_request",[case,fraction,panel,count,neighborhood,quantity],"Calculate spanwise pressure loading"):
        return
    if quantity=="recompute" and not case["pressure_basis_verified"]:
        raise InsufficientData("Recomputed Cp requires a confirmed freestream pressure; use static pressure/Δp meanwhile")
    labels,side_info = labels_from(case)
    result = spanwise_pressure_trace(data,geo,ref,labels,fraction,neighborhood,int(count),quantity,panel)
    result.metadata["surface_identification"] = side_info
    st.json(result.metadata,expanded=False)
    st.dataframe(result.table,width="stretch")
    x = "normalized_semispan" if normalized else "span_m"
    if "pressure_upper_Pa" in result.table:
        show_plot(line_plot(result.table,x,["pressure_upper_Pa","pressure_lower_Pa"],title=f"Surface pressure at x/c={fraction:g} [Pa]"),"span_chord_pressure")
        show_plot(line_plot(result.table,x,["approx_vertical_pressure_load_N_per_m2"],title=f"Pressure loading at x/c={fraction:g} [N/m²]"),"span_chord_load")
    if quantity!="pressure":
        show_plot(line_plot(result.table,x,["cp_upper","cp_lower"],invert_y=True,title="Cp along selected chord location"),"span_chord_cp")
        show_plot(line_plot(result.table,x,["delta_cp"],title="Lower − upper Cp along selected chord location"),"span_chord_delta")
    downloads(result,"spanwise_at_chord",case,data)


def spanwise_page(case):
    show_geometry_source(case)
    scope = st.radio("Spanwise quantity",["At a selected chord location","Whole-chord integrated lift / drag"],horizontal=True)
    if scope.startswith("At"):
        spanwise_chord_location_page(case)
        return
    panel, count, width, component = discretization(case)
    normalized = st.checkbox("Use signed normalized semispan", False)
    names = st.multiselect("Compare other wing / CFD cases", [n for n in st.session_state.cases if n != selected])
    elliptical = st.checkbox("Overlay ideal full-wing elliptical lift", False)
    leading_edge = st.checkbox("Also calculate leading-edge pitching moment", False)
    elastic_axis = st.number_input("Structural preview reference axis [% chord], user-defined assumption", min_value=0., max_value=100., value=35.) / 100
    if not calculation_active("span_request", [case, {n: st.session_state.cases[n] for n in names}, panel, count, width, component, leading_edge, elastic_axis], "Calculate spanwise distributions"):
        return
    results = {}
    base_geo, base_ref = None, None
    for name in [selected, *names]:
        candidate = st.session_state.cases[name]
        data, _, _, ref, _, geo, index, forces = prepared(candidate)
        if base_geo is None:
            base_geo, base_ref = geo, ref
        else:
            if not (np.allclose(geo.span_axis, base_geo.span_axis) and np.allclose(geo.origin, base_geo.origin)):
                st.warning(f"Skipped {name}: span reference differs. Configure a common span comparison frame.")
                continue
            if not (np.allclose(unit(ref.lift), unit(base_ref.lift)) and np.allclose(unit(ref.drag), unit(base_ref.drag))):
                st.info(f"{name}: lift/drag use this case's wind axes at AoA {candidate['aoa_deg']:g}°. This compares wind-frame scalar loads; vector components use different directions.")
        result = spanwise_distribution(data, forces.component(component), geo, ref,
                                       make_span_edges(geo, count, panel, width), index=index)
        result.metadata["force_component"] = component
        if leading_edge:
            le_result = spanwise_distribution(data, forces.component(component), geo, ref,
                make_span_edges(geo, count, panel, width), 0., index)
            result.table["leading_edge_pitch_Nm_per_m"] = le_result.table.pitch_Nm_per_m
        # Contributions use identical bins, even when plotted forces are pressure-only.
        for label, array in (("pressure", forces.pressure), ("viscous", forces.viscous)):
            if array is not None:
                part = spanwise_distribution(data, array, geo, ref, make_span_edges(geo, count, panel, width), index=index)
                result.table[label+"_lift_N_per_m"] = part.table.lift_N_per_m
                result.table[label+"_drag_N_per_m"] = part.table.drag_N_per_m
        try:
            torsion = spanwise_distribution(data, forces.component(component), geo, ref,
                make_span_edges(geo, count, panel, width), elastic_axis, index)
            beam = structural_loads(torsion)
            # Midpoint structural columns refer to the explicitly chosen chord axis.
            attached = attach_structural_columns(torsion, beam)
            for column in ("shear_N", "bending_Nm", "internal_torque_Nm"):
                result.table[column] = attached.table[column]
            result.table["torsion_Nm_per_m"] = torsion.table.pitch_Nm_per_m
            result.metadata["structural_axis_fraction"] = elastic_axis
            result.metadata["structural_note"] = "User-defined reference axis; actual spar/elastic-axis location must come from structural geometry."
        except ValueError as exc:
            result.metadata["structural_status"] = str(exc)
        results[name] = result
        st.write(name)
        show_wing_forces(data,forces,ref,candidate,"span_"+re.sub(r"\W","_",name))
        st.json(result.metadata, expanded=False)
        downloads(result, "span_"+re.sub(r"\W", "_", name), candidate, data)
        st.dataframe(result.table, width="stretch")
    x = "normalized_semispan" if normalized else "span_m"
    for y, title in [("lift_N_per_m", "Lift [N/m]"), ("drag_N_per_m", "Drag [N/m]"), ("cl", "Sectional lift coefficient"),
                     ("cd", "Sectional drag coefficient"), ("pitch_Nm_per_m", "Quarter-chord pitch [Nm/m]"),
                     ("cm", "Quarter-chord moment coefficient"), ("cumulative_lift_N", "Cumulative lift, increasing signed span [N]")]:
        fig = compare_distributions(results, x, y, title)
        if y == "lift_N_per_m" and elliptical:
            for name, result in results.items():
                try:
                    geo = geometry_from(st.session_state.cases[name])
                    if panel != "both" or result.metadata["outside_span_faces"]:
                        raise ValueError("Overlay requires complete full-span selection")
                    curve = elliptical_reference(result.table.span_m, geo, result.table.strip_lift_N.sum())
                    fig.add_scatter(x=result.table[x], y=curve, mode="lines", name=name+" ideal ellipse")
                except ValueError as exc:
                    st.warning(str(exc))
        show_plot(fig, "span_"+y)
    if selected in results and "viscous_drag_N_per_m" in results[selected].table:
        show_plot(line_plot(results[selected].table, x, ["pressure_drag_N_per_m", "viscous_drag_N_per_m"], title="Pressure and viscous drag projections"), "drag_parts")
    if leading_edge:
        show_plot(compare_distributions(results, x, "leading_edge_pitch_Nm_per_m", "Leading-edge pitch [Nm/m]"), "leading_edge_pitch")
    st.caption("Sectional force projections do not separate induced and profile drag. Normalized semispan uses each wing's supplied geometry; coefficients use each case's q and local chord.")


def structural_page(case):
    show_geometry_source(case)
    panel, count, width, component = discretization(case)
    axis = st.number_input("Structural axis [% local chord]", min_value=0., max_value=100., value=35.) / 100
    st.info("Axis location is a user-defined assumption. Aerodynamic loads only; no gravity, inertia, propulsion or concentrated masses. This is preliminary beam analysis.")
    if not calculation_active("beam_request", [case, panel, count, width, component, axis], "Calculate beam loads"):
        return
    data, _, _, ref, _, geo, index, forces = prepared(case)
    show_wing_forces(data,forces,ref,case,"structural_")
    st.caption(f"The beam diagrams below use the selected {component} force component.")
    span = spanwise_distribution(data, forces.component(component), geo, ref,
                                make_span_edges(geo, count, panel, width), axis, index)
    beam = structural_loads(span)
    span.table["torsion_Nm_per_m"] = span.table.pitch_Nm_per_m
    beam.metadata["force_component"] = component
    root_rows = [{"panel":panel,"root_shear_N":summary["shear_N"]["root_value"],
                  "root_bending_Nm":summary["bending_Nm"]["root_value"],
                  "root_torque_Nm":summary["internal_torque_Nm"]["root_value"]}
                 for panel,summary in beam.metadata["summary"].items()]
    st.subheader("Root loads")
    st.dataframe(pd.DataFrame(root_rows),width="stretch",hide_index=True)
    with st.expander("Beam model and maximum-load details"):
        st.json(beam.metadata)
    show_plot(line_plot(beam.table, "distance_from_root_m", ["shear_N"], title="Shear [N]"), "beam_shear")
    show_plot(line_plot(beam.table, "distance_from_root_m", ["bending_Nm"], title="Bending moment [Nm]"), "beam_bending")
    show_plot(line_plot(span.table, "span_m", ["torsion_Nm_per_m"], title="Distributed torsion about structural axis [Nm/m]"), "beam_torsion")
    show_plot(line_plot(beam.table, "distance_from_root_m", ["internal_torque_Nm"], title="Internal torque [Nm]"), "beam_torque")
    downloads(beam, "beam_edges", case, data)
    downloads(attach_structural_columns(span, beam), "span_with_structural_loads", case, data)


def validation_page(case):
    data, _, _ = configured_data(case)
    ref = ref_from(case)
    diagnostics = dataset_diagnostics(data, ref)
    st.json(diagnostics)
    for warning in diagnostics["warnings"]:
        st.warning(warning)
    st.download_button("Download validation metadata", metadata_text(provenance(case, data, {"diagnostics": diagnostics})), "validation_metadata.json")
    try:
        forces = face_forces(data, ref, convention_from(case))
    except InsufficientData as exc:
        st.warning(f"Engineering force report unavailable: {exc}")
        return
    show_wing_forces(data,forces,ref,case,"validation_")
    computed = {}
    for name, array in (("pressure", forces.pressure), ("viscous", forces.viscous), ("total", forces.total if forces.viscous is not None else None)):
        if array is None:
            continue
        summary = integrate(array, data.vector(), case["moment_reference_point"])
        force, moment = summary["force_N"], summary["moment_Nm"]
        for i, axis in enumerate("xyz"):
            computed[f"{name}_F{axis}_N"] = force[i]
            computed[f"{name}_M{axis}_Nm"] = moment[i]
        computed[name+"_L_N"] = force @ unit(ref.lift)
        computed[name+"_D_N"] = force @ unit(ref.drag)
    st.dataframe(pd.DataFrame({"quantity": list(computed), "computed": list(computed.values())}), width="stretch")
    area = case["S_ref_m2"]
    try:
        report_geometry = geometry_from(case)
    except InsufficientData:
        report_geometry = None
    if area == 0 and report_geometry is not None:
        area = report_geometry.area
    if area > 0:
        coeffs = {}
        for name in ("pressure", "viscous", "total"):
            if name+"_L_N" in computed:
                coeffs[name+"_CL"] = computed[name+"_L_N"] / ref.q / area
                coeffs[name+"_CD"] = computed[name+"_D_N"] / ref.q / area
                if case["c_ref_m"] > 0 and report_geometry is not None:
                    pitch = np.array([computed[f"{name}_M{a}_Nm"] for a in "xyz"]) @ report_geometry.span_axis
                    coeffs[name+"_CM"] = pitch / ref.q / area / case["c_ref_m"]
        st.json({"coefficients": coeffs, "reference_area_m2": area, "reference_chord_m": case["c_ref_m"], "reference_point_m": case["moment_reference_point"]})
    st.subheader("Comparison with Fluent force reports")
    st.caption("Use the same walls, pressure reference, units, directions and moment point. Enter only quantities actually reported. Blank values remain unvalidated. Full/half-wing scaling is never automatic.")
    reported_frame = pd.DataFrame({"quantity": list(computed), "Fluent_value": [np.nan]*len(computed)})
    report_upload = st.file_uploader("Optional Fluent report CSV: quantity,Fluent_value", type=["csv"])
    if report_upload:
        reported_frame = pd.read_csv(report_upload)
    edited = st.data_editor(reported_frame, disabled=["quantity"], width="stretch", key="fluent_reports_"+selected)
    comparison = compare_reports(computed, dict(zip(edited.quantity, edited.Fluent_value)))
    if comparison.empty:
        st.warning("No Fluent report supplied: agreement with the CFD solver is unvalidated")
    else:
        st.dataframe(comparison, width="stretch")
        st.download_button("Download force comparison", csv_bytes(comparison, provenance(case, data)), "fluent_comparison.csv")
    if report_geometry is not None:
        count = st.number_input("Base strip count for convergence study", min_value=2, max_value=500, value=20)
        if calculation_active("convergence_request", [case, count], "Run strip-resolution study"):
            geo = geometry_from(case)
            component = "total" if forces.viscous is not None else "pressure"
            result = convergence_diagnostics(data, forces.component(component), geo, ref,
                                             (int(count), int(count)*2, int(count)*4, int(count)*8))
            st.dataframe(result, width="stretch")
            st.download_button("Download convergence study", csv_bytes(result, provenance(case, data)), "convergence.csv")
            st.caption("Conservation alone does not establish local accuracy. Compare root bending changes, strip populations and chordwise distributions as resolution varies.")
        station = st.number_input("Chordwise convergence station [m]", value=0.)
        strip_width = st.number_input("Chordwise convergence strip width [m]", min_value=.0001, value=.1)
        if calculation_active("chord_convergence_request", [case, station, strip_width], "Run chord-bin resolution study"):
            result = chordwise_convergence(data, forces.total, geometry_from(case), ref, station, strip_width)
            st.dataframe(result, width="stretch")
            st.download_button("Download chord-bin study", csv_bytes(result, provenance(case, data,
                {"force_component": "total" if forces.viscous is not None else "pressure"})), "chord_bin_convergence.csv")


try:
    if page == PAGES[0]:
        manager(selected)
    elif not selected:
        st.info("Import a CFD case on the Data Manager page")
    else:
        case = st.session_state.cases[selected]
        if page == PAGES[1]:
            chordwise_page(case)
        elif page == PAGES[2]:
            spanwise_page(case)
        elif page == PAGES[3]:
            structural_page(case)
        else:
            validation_page(case)
except (ValueError, OSError, KeyError, TypeError) as exc:
    st.error(str(exc))
    st.caption("No engineering result was produced for this failed calculation. Review Data Manager settings and input completeness.")
