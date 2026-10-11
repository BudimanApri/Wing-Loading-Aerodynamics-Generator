import numpy as np
import pytest
from src.geometry import WingGeometry
from src.validation import dataset_diagnostics, compare_reports
from src.force_integration import face_forces, Reference, InsufficientData
from src.spanwise_analysis import spanwise_distribution, make_span_edges
from src.structural_loads import structural_loads


def test_swept_tapered_twisted_local_frame():
    geo=WingGeometry([0,2],[[0,0,0],[1,2,0]],[[2,0,0],[2,2,0]],[0,30])
    s=np.array([0,1,2])
    le,c,ec,normal=geo.frame(s)
    points=le+.4*c[:,None]*ec+.02*normal
    actual_s,xi,height=geo.coordinates(points)
    np.testing.assert_allclose(actual_s,s)
    np.testing.assert_allclose(xi,.4,atol=1e-12)
    np.testing.assert_allclose(height,.02,atol=1e-12)
    assert geo.area==pytest.approx(3)


def test_rotated_coordinate_frame():
    geo=WingGeometry([0,1],[[0,0,0],[1,0,0]],[[0,1,0],[1,1,0]],[0,0],span_axis=(1,0,0))
    s,xi,h=geo.coordinates(np.array([[.5,.2,.03]]))
    np.testing.assert_allclose([s[0],xi[0],h[0]],[.5,.2,-.03])


def test_user_downward_z_frame_and_panel_mapping(uniform_wing,reference,convention):
    geo=WingGeometry([-2,0,2],[[0,-2,0],[0,0,0],[0,2,0]],[[1,-2,0],[1,0,0],[1,2,0]],
                     [0,0,0],upper_normal_sign=-1,positive_panel="left")
    uniform_wing.columns["z"]*=-1
    uniform_wing.columns["az"]*=-1
    alpha=np.deg2rad(6)
    ref=Reference(1.225,10,0,0,(-np.sin(alpha),0,-np.cos(alpha)),(np.cos(alpha),0,-np.sin(alpha)))
    f=face_forces(uniform_wing,ref,convention).pressure
    result=spanwise_distribution(uniform_wing,f,geo,ref,make_span_edges(geo,10,"left"))
    assert (result.table.span_m>0).all()
    assert set(result.table.panel)=={"left"}
    assert result.table.strip_lift_N.sum()==pytest.approx(60*np.cos(alpha))
    assert structural_loads(result).table.span_m.max()==pytest.approx(2)
    from src.interpolation import surface_labels
    labels=surface_labels(uniform_wing,geo,convention)
    assert (labels[uniform_wing.columns["z"]<0]=="upper").all()


def test_geometry_extrapolation_is_disabled(geometry):
    with pytest.raises(ValueError,match="outside"):
        geometry.frame(3)


def test_duplicate_records_detected_and_blocked(uniform_wing,reference,convention):
    data=uniform_wing
    for key in data.columns:
        data.columns[key]=np.r_[data.columns[key],data.columns[key][:1]]
    diag=dataset_diagnostics(data,reference)
    assert diag["exact_duplicate_records"]==1
    with pytest.raises(InsufficientData,match="duplicated"):
        face_forces(data,reference,convention)


def test_comparison_zero_report_and_missing():
    result=compare_reports({"L_N":1,"D_N":0},{"L_N":2,"D_N":0,"missing":100})
    assert len(result)==2
    assert result.iloc[0].relative_error_percent==50
    assert np.isnan(result.iloc[1].relative_error_percent)


def test_duplicate_evidence_keeps_unique_ids_visible(uniform_wing):
    from src.validation import duplicate_records
    data=uniform_wing
    data.columns["id"]=np.arange(1,len(data)+1,dtype=float)
    for key in data.columns:
        value=data.columns[key][:1] if key!="id" else [len(data)+1]
        data.columns[key]=np.r_[data.columns[key],value]
    records=duplicate_records(data)
    assert len(records)==2
    assert records.id.nunique()==2
    assert records.repeated_copy.sum()==1
    assert records.duplicate_group.nunique()==1


def test_coincident_coordinates_with_distinct_area_are_not_duplicates(uniform_wing):
    from src.validation import duplicate_records
    data=uniform_wing
    for key in data.columns:
        data.columns[key]=np.r_[data.columns[key],data.columns[key][:1]]
    data.columns["az"][-1]*=2
    assert duplicate_records(data).empty
