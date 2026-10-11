import numpy as np
import pytest
from src.force_integration import face_forces, ForceConvention
from src.chordwise_analysis import chordwise_distribution
from src.interpolation import pressure_section, surface_labels


def test_chord_force_conservation_and_center_of_pressure(uniform_wing,geometry,reference,convention):
    f=face_forces(uniform_wing,reference,convention).pressure
    result=chordwise_distribution(uniform_wing,f,geometry,reference,.45,.4,20)
    np.testing.assert_allclose(result.table.load_N_per_m2,30,atol=1e-10)
    assert result.metadata["lift_N_per_m"]==pytest.approx(30)
    assert result.metadata["center_of_pressure_xi"]==pytest.approx(.5)
    assert result.metadata["force_conservation_error_N"]<1e-11
    assert result.table.cumulative_lift_N_per_m.iloc[-1]==pytest.approx(30)


def test_tip_strip_uses_effective_width(uniform_wing,geometry,reference,convention):
    f=face_forces(uniform_wing,reference,convention).pressure
    result=chordwise_distribution(uniform_wing,f,geometry,reference,2,.4,20)
    assert result.metadata["effective_width_m"]==pytest.approx(.2)
    assert result.metadata["lift_N_per_m"]==pytest.approx(30)


def test_linear_pressure_interpolation_not_extrapolated(uniform_wing,geometry,reference,convention):
    data=uniform_wing
    data.columns["p"]=np.where(data.columns["z"]>0, -10+2*data.columns["x"]+3*data.columns["y"],20+2*data.columns["x"]+3*data.columns["y"])
    labels=surface_labels(data,geometry,convention)
    result=pressure_section(data,geometry,reference,.43,.5,100,"recompute",labels)
    valid=result.table.delta_cp.notna()
    assert valid.any() and (~valid).any()
    np.testing.assert_allclose(result.table.delta_cp[valid],15,atol=1e-11)
    np.testing.assert_allclose(result.table.pressure_difference_Pa[valid],30,atol=1e-11)
    assert "lift_N_per_m" not in result.metadata


def test_no_fabricated_pressure_force_or_zero_cp(uniform_wing,geometry,reference,convention):
    labels=np.full(len(uniform_wing),"upper")
    result=pressure_section(uniform_wing,geometry,reference,0,.5,50,"exported",labels)
    assert result.table.cp_lower.isna().all()
    assert result.table.delta_cp.isna().all()
    assert "pressure_difference_Pa" not in result.table
    with pytest.raises(ValueError,match="labels"):
        surface_labels(uniform_wing,geometry,ForceConvention())


def test_edge_assignment_includes_overhang_forces(uniform_wing,geometry,reference,convention):
    uniform_wing.columns["x"][0]=-0.01
    f=face_forces(uniform_wing,reference,convention).pressure
    result=chordwise_distribution(uniform_wing,f,geometry,reference,-1.95,.1,10)
    assert result.metadata["outside_chord_faces_assigned_to_edge_bins"]==1
    assert result.metadata["force_conservation_error_N"]<1e-12


def test_chord_bin_convergence_is_conservative(uniform_wing,geometry,reference,convention):
    from src.validation import chordwise_convergence
    f=face_forces(uniform_wing,reference,convention).pressure
    result=chordwise_convergence(uniform_wing,f,geometry,reference,0,.4)
    np.testing.assert_allclose(result.lift_N_per_m,30)
    assert result.conservation_error_N.max()<1e-12
    assert result.empty_bins.iloc[-1]>result.empty_bins.iloc[0]
