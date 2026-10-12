# Wing Aerodynamic Analysis

A local Python application for ANSYS Fluent aerodynamic surface exports, built with Streamlit and Plotly.

Features include CFD file/folder selection, automatic sampled geometry, chordwise and spanwise pressure/force distributions, integrated wing forces, preliminary beam loads, validation diagnostics and exports.

## Install and run

Open PowerShell in the application directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`. Press Ctrl+C to stop the server. If the port is occupied, append `--server.port 8502` and open `http://localhost:8502`.

Python 3.11 or newer is supported. Use `requirements.txt` for a compatible fresh installation; `requirements-lock.txt` records the development environment. PNG/SVG export needs Kaleido and a Chrome-compatible browser. Self-contained HTML plots work offline. Restart Streamlit after editing numerical modules under `src/`.

## Workflow

1. In **CFD Data Manager**, import a named wing/case. Enter a full file path such as `C:\CFD\exports\wing.csv`, or a folder and select exports from its list. Choose files belonging to one lifting surface/case. Separate upper/lower files can have explicit side labels.
2. Enter actual density, speed, AoA, pressure references, export Location and vector conventions. Defaults and filename suggestions are not verified simulation conditions.
3. Review automatic sampled geometry. It uses coordinate extrema and measured section inclination, not CAD reconstruction. Manual LE/TE geometry remains an optional override for exact reference geometry.
4. Use **Chordwise Analysis** at a signed span station. Use **Spanwise Analysis** for a fixed-chord pressure trace or whole-chord integrated lift/drag.
5. Expand **Integrated wing forces** on Data Manager and click **Calculate total wing force**. The summary also appears after spanwise/structural calculations and in **Validation and Reporting**. It integrates all selected wall faces, independently of panel/strip selections. Full total is available only with verified viscous traction; pressure-only results are labeled separately.
6. In **Structural Loads**, select panel, discretization and force component, specify the structural reference axis as a chord percentage, and calculate beam loads.
7. Compare integrated forces against Fluent reports using identical walls, directions, pressure references and units. Check resolution studies before using local distributions.

## Data and numerical conventions

Comma/semicolon/tab/whitespace tabular ASCII exports are supported. Fields include x/y/z coordinates, static pressure or Cp, face-area-vector components, optional wall-shear components and zone/side labels. Unsupported mesh/profile formats are rejected.

Native Fluent files can contain leading location coordinates and later requested coordinate quantities. Location coordinates drive geometry, filtering and moment arms; later quantities are retained as `scalar_x/y/z`. Ambiguous generic repeated coordinates must agree.

For boundary-wall force integration, export **Location = Cell Center**. Nodal exports support sampled geometry and pressure interpolation; averaged nodal area/shear/pressure values cannot be assumed to represent independent faces. [Fluent export documentation](https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/flu_ug/flu_ug_sec_file_export_steps.html)

Calculations use SI units. Pressure references must match the export's gauge/absolute basis. If area vectors point from body into fluid, body pressure force is `-(p-p_ref) A`; reverse vectors use the opposite sign. Viscous force is verified wall traction multiplied by area magnitude. Lift/drag directions are configurable unit vectors.

The default frame is +x aft, +z down, +y = z cross x. For nose-up-positive incidence alpha, `e_D=(cos alpha,0,-sin alpha)` and `e_L=(-sin alpha,0,-cos alpha)`. Directions must match the actual model frame. Moments follow the right-hand rule about positive configured span.

Face-center strip/bin assignment conserves selected forces but does not geometrically clip faces crossing boundaries. Local accuracy depends on mesh spacing and bin resolution. Panel grids start at exact root zero, avoiding artificial slivers from slightly asymmetric tips. Genuine unresolved strips block beam integration.

Pressure interpolation does not extrapolate beyond its local convex hull; missing coverage remains NaN. Pressure difference is approximate sectional loading in N/m², distinct from complete 3D projected face forces and whole-chord lift in N/m. Cp references are checked separately. Induced/profile drag and aerodynamic-center claims are not inferred from force projections.

Resultant wing force is `|sum F|`, not `sum |F|`. No symmetry multiplier is applied automatically.

Beam diagrams use exact integrals of piecewise-constant strip densities for equivalent straight cantilever panels. Structural-axis moments use the full force cross product about the chosen chord axis. Loads are aerodynamic only; weight, inertia, propulsion, masses, stress and failure assessment require additional modeling.

## Performance and exports

Large ASCII files are parsed in chunks with float64 precision and cached as memory-mapped NPY columns. Cache keys include file path, size, mtime and parser version. Original files are not reparsed for each station query. Optional Parquet conversion streams chunks.

CSV exports include units and `#` JSON metadata comments; read with `pandas.read_csv(path, comment='#')`. Metadata records source signatures, references, conventions and diagnostics. Excel uses Results and References sheets. Plots export as HTML, PNG or SVG.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests generate synthetic analytic fixtures and do not require private CFD datasets. They cover force/moment signs, vector totals, references, conservative binning, geometry, interpolation, parser/cache behavior, asymmetric-span root handling and beam integrals.
