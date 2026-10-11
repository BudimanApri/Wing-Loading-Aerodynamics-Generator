import numpy as np
import pytest
from src.force_integration import face_forces
from src.spanwise_analysis import spanwise_distribution, make_span_edges, elliptical_reference
from src.structural_loads import structural_loads, attach_structural_columns


def test_full_span_conservation_coefficients_moments(uniform_wing, geometry, reference, convention):
    force = face_forces(uniform_wing,reference,convention).pressure
    result = spanwise_distribution(uniform_wing,force,geometry,reference,make_span_edges(geometry,20))
    np.testing.assert_allclose(result.table.lift_N_per_m,30,atol=1e-11)
    np.testing.assert_allclose(result.table.cl,15,atol=1e-11)
    np.testing.assert_allclose(result.table.pitch_Nm_per_m,-7.5,atol=1e-11)
    np.testing.assert_allclose(result.table.cm,-3.75,atol=1e-11)
    assert result.table.strip_lift_N.sum() == pytest.approx(120)
    assert result.metadata["conservation_absolute_error_N"] < 1e-10
    assert result.metadata["outside_span_faces"] == 0


def test_uniform_cantilever_and_torque_exact(uniform_wing,geometry,reference,convention):
    force=face_forces(uniform_wing,reference,convention).pressure
    result=spanwise_distribution(uniform_wing,force,geometry,reference,make_span_edges(geometry,20),.35)
    beam=structural_loads(result)
    for panel, rows in beam.table.groupby("panel"):
        r=rows.distance_from_root_m.to_numpy()
        np.testing.assert_allclose(rows.shear_N,30*(2-r),atol=1e-11)
        np.testing.assert_allclose(rows.bending_Nm,15*(2-r)**2,atol=1e-11)
        np.testing.assert_allclose(rows.internal_torque_Nm,-4.5*(2-r),atol=1e-11)
    mids=attach_structural_columns(result,beam).table
    np.testing.assert_allclose(mids.bending_Nm,15*(2-np.abs(mids.span_m))**2,atol=1e-11)


def test_nonuniform_bins_conserve_all_faces(uniform_wing,geometry,reference,convention):
    f=face_forces(uniform_wing,reference,convention).total
    result=spanwise_distribution(uniform_wing,f,geometry,reference,[-2,-1.1,-.15,0,.3,1.4,2])
    np.testing.assert_allclose(result.metadata["strip_sum_force_N"],f.sum(axis=0),atol=1e-11)


def test_empty_strips_block_structural(uniform_wing,geometry,reference,convention):
    f=face_forces(uniform_wing,reference,convention).pressure
    result=spanwise_distribution(uniform_wing,f,geometry,reference,make_span_edges(geometry,1000))
    assert result.metadata["empty_strips"] > 0
    assert result.table.lift_N_per_m.isna().any()
    with pytest.raises(ValueError,match="Empty"):
        structural_loads(result)


def test_panel_selection_and_ellipse(uniform_wing,geometry,reference,convention):
    f=face_forces(uniform_wing,reference,convention).pressure
    for panel in ("left","right"):
        result=spanwise_distribution(uniform_wing,f,geometry,reference,make_span_edges(geometry,10,panel))
        assert set(result.table.panel)=={panel}
        assert result.table.strip_lift_N.sum()==pytest.approx(60)
    s=np.linspace(-2,2,10001)
    curve=elliptical_reference(s,geometry,120)
    assert np.trapezoid(curve,s)==pytest.approx(120,rel=2e-6)


@pytest.mark.parametrize("count",[1,20,40,80,81,160])
def test_asymmetric_tip_rounding_does_not_create_root_sliver(geometry,count):
    from src.geometry import WingGeometry
    bounds=np.array([-2.,0.,2.000005245])
    le=np.column_stack([np.zeros(3),bounds,np.zeros(3)])
    te=le+np.array([1.,0,0])
    geo=WingGeometry(bounds,le,te,[0,0,0])
    edges=make_span_edges(geo,count)
    assert len(edges)-1==max(2,count)
    assert np.count_nonzero(edges==0)==1
    assert np.diff(edges).min()>.01
    assert edges[0]==-2 and edges[-1]==bounds[-1]


def test_asymmetric_span_conserves_force_and_allows_structural(uniform_wing,reference,convention):
    from src.geometry import WingGeometry
    bounds=np.array([-2.,0.,2.000005245])
    le=np.column_stack([np.zeros(3),bounds,np.zeros(3)])
    geo=WingGeometry(bounds,le,le+np.array([1.,0,0]),[0,0,0])
    force=face_forces(uniform_wing,reference,convention).pressure
    result=spanwise_distribution(uniform_wing,force,geo,reference,make_span_edges(geo,20))
    assert result.metadata["empty_strips"]==0
    np.testing.assert_allclose(result.metadata["strip_sum_force_N"],force.sum(axis=0),atol=1e-11)
    beam=structural_loads(result)
    for panel,rows in beam.table.groupby("panel"):
        assert rows.iloc[0].shear_N==pytest.approx(60)
        assert rows.iloc[-1].shear_N==pytest.approx(0)
        assert rows.iloc[-1].bending_Nm==pytest.approx(0)


def test_width_strips_are_root_anchored_for_both_panels(geometry):
    edges=make_span_edges(geometry,80,width=.3)
    assert np.count_nonzero(edges==0)==1
    assert .3 in edges and -.3 in edges
    assert edges[[0,-1]].tolist()==[-2,2]
    assert np.diff(edges).min()>.19
