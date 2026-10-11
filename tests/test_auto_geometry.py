import numpy as np
import pytest
from src.fluent_parser import Dataset
from src.auto_geometry import detect_geometry,infer_pressure_sides
from src.pressure_sampling import spanwise_pressure_trace
from src.interpolation import pressure_section
from src.force_integration import InsufficientData,ForceConvention,face_forces


def envelope_cloud():
    s=np.linspace(-2,2,81)
    le=.15*np.abs(s)
    c=1-.2*np.abs(s)
    xi=np.linspace(0,1,101)
    x=(le[:,None]+c[:,None]*xi).ravel()
    y=np.repeat(s,len(xi))
    mid=.03*x+.02*y
    h=.04*np.sin(np.pi*np.tile(xi,len(s)))+.001
    n=len(x)
    return Dataset({"x":np.tile(x,2),"y":np.tile(y,2),"z":np.r_[mid-h,mid+h],
        "p":np.r_[-10+2*x+3*y,20+2*x+3*y],
        "ax":np.zeros(2*n),"ay":np.zeros(2*n),"az":np.repeat([-1.,1.],n)}, {})


def test_detects_measured_sweep_taper_and_height_without_guessing():
    data=envelope_cloud()
    estimate=detect_geometry(data,strips=40)
    assert estimate.metadata["span_bounds_m"]==[-2,2]
    _,c,ec,_=estimate.geometry.frame(np.array([0.,1.,2.]))
    # Envelope has a finite-strip error; it must stay within sampled-strip scale.
    np.testing.assert_allclose(c,[1.,.8,.6],atol=.03)
    assert (ec[:,2]>0).all()
    assert not estimate.metadata["exact_CAD_geometry"]


def test_auto_sides_support_pressure_but_do_not_verify_forces(reference):
    data=envelope_cloud(); geo=detect_geometry(data,40).geometry
    labels,info=infer_pressure_sides(data,geo,span_bins=40,chord_bins=40)
    n=len(data)//2
    assert (labels[:n]=="upper").all()
    assert (labels[n:]=="lower").all()
    assert info["agreement_fraction"]> .99
    assert not info["force_convention_verified"]
    with pytest.raises(InsufficientData,match="face-centered"):
        face_forces(data,reference,ForceConvention())
    for key in ("ax","ay","az"):
        data.columns[key]*=-1
    reversed_labels,reversed_info=infer_pressure_sides(data,geo,40,40)
    np.testing.assert_array_equal(labels,reversed_labels)
    assert reversed_info["suggested_area_direction_diagnostic_only"]=="fluid_to_body"


def test_pressure_difference_has_no_freestream_reference_dependency(reference):
    data=envelope_cloud(); geo=detect_geometry(data,40).geometry
    # Compare matched local sections with span-dependent pressure on both
    # surfaces. A gradient in global x would sample different physical x at
    # matching xi when the measured chord is tilted.
    n=len(data)//2
    data.columns["p"]=np.r_[-10+3*data.columns["y"][:n],20+3*data.columns["y"][n:]]
    labels,_=infer_pressure_sides(data,geo,40,40)
    result=pressure_section(data,geo,reference,.5,.2,50,"pressure",labels)
    valid=result.table.pressure_difference_Pa.dropna()
    assert len(valid)>35
    np.testing.assert_allclose(valid,30,atol=1e-8)
    assert result.table.cp_upper.isna().all()
    assert "lift_N_per_m" not in result.metadata
    from src.force_integration import Reference
    shifted=pressure_section(data,geo,Reference(2,20,100000,100000),.5,.2,50,"pressure",labels)
    np.testing.assert_allclose(result.table.pressure_difference_Pa,shifted.table.pressure_difference_Pa)


def test_selected_chord_span_trace_is_pressure_not_whole_chord_force(reference):
    data=envelope_cloud();geo=detect_geometry(data,40).geometry
    n=len(data)//2
    data.columns["p"]=np.r_[-10+3*data.columns["y"][:n],20+3*data.columns["y"][n:]]
    labels,_=infer_pressure_sides(data,geo,40,40)
    result=spanwise_pressure_trace(data,geo,reference,labels,.4,.1,80,"pressure","left")
    assert (result.table.span_m>=0).all()
    np.testing.assert_allclose(result.table.pressure_difference_Pa.dropna(),30,atol=1e-8)
    assert "lift_N_per_m" not in result.table
    assert result.metadata["chord_location_xi"]==.4


def test_single_surface_does_not_invent_other_side():
    data=envelope_cloud().subset(np.arange(len(envelope_cloud()))<len(envelope_cloud())//2)
    geo=detect_geometry(data,40).geometry
    with pytest.raises(InsufficientData,match="both"):
        infer_pressure_sides(data,geo,40,40)
