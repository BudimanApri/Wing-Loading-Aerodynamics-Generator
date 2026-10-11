import numpy as np
import pytest
from src.fluent_parser import Dataset
from src.force_integration import Reference, ForceConvention, face_forces, integrate, InsufficientData


def flat_plate(n=10):
    x = np.tile((np.arange(n) + .5) / n, 2)
    return Dataset(dict(x=x, y=np.zeros(2*n), z=np.repeat([.05, -.05], n),
                        p=np.repeat([-10., 20.], n), ax=np.zeros(2*n), ay=np.zeros(2*n),
                        az=np.repeat([1/n, -1/n], n), tx=np.ones(2*n),
                        ty=np.zeros(2*n), tz=np.zeros(2*n)), {})


REF = Reference(1., 2., 0., 0.)
CONV = ForceConvention(True, "body_to_fluid", "on_body", True, True, True)


def test_pressure_viscous_moment_known_plate():
    d = flat_plate()
    f = face_forces(d, REF, CONV)
    result = integrate(f.total, d.vector(), (.25, 0, 0))
    np.testing.assert_allclose(result["force_N"], [2, 0, 30], atol=1e-12)
    np.testing.assert_allclose(result["moment_Nm"], [0, -7.5, 0], atol=1e-12)


def test_inward_vectors_and_fluid_shear_corrected():
    d = flat_plate()
    for key in ("ax", "ay", "az", "tx", "ty", "tz"):
        d.columns[key] *= -1
    result = face_forces(d, REF, ForceConvention(True, "fluid_to_body", "on_fluid", True, True, True))
    np.testing.assert_allclose(result.total.sum(axis=0), [2, 0, 30], atol=1e-12)


def test_unverified_and_missing_data_block_force():
    with pytest.raises(InsufficientData, match="face-centered"):
        face_forces(flat_plate(), REF, ForceConvention())
    d = flat_plate()
    del d.columns["az"]
    with pytest.raises(InsufficientData, match="missing"):
        face_forces(d, REF, CONV)


def test_pressure_reference_closed_and_open_surfaces():
    d = flat_plate()
    f0 = face_forces(d, REF, CONV)
    f1 = face_forces(d, Reference(1, 2, 100, 100), CONV)
    np.testing.assert_allclose(f0.pressure.sum(axis=0), f1.pressure.sum(axis=0), atol=1e-12)
    upper = d.subset(d.columns["z"] > 0)
    shift = face_forces(upper, Reference(1, 2, 100, 100), CONV).pressure.sum(axis=0)
    np.testing.assert_allclose(shift, [0, 0, 110], atol=1e-12)


def test_missing_shear_cannot_be_labelled_total():
    d = flat_plate()
    del d.columns["tx"]
    f = face_forces(d, REF, CONV)
    with pytest.raises(InsufficientData, match="Total"):
        f.component("total")


def test_bad_reference_and_area_rejected():
    with pytest.raises(ValueError):
        Reference(1, 0, 0, 0)
    with pytest.raises(ValueError):
        Reference(1, 1, 0, 0, drag=(0, 0, 1))
    d = flat_plate()
    d.columns["az"][0] = 0
    with pytest.raises(InsufficientData, match="zero"):
        face_forces(d, REF, CONV)


def test_known_nodal_export_cannot_be_enabled_by_other_verification_flags():
    from dataclasses import replace
    d=flat_plate()
    d.metadata["exact_duplicate_records"]=1
    nodal=replace(CONV,export_location="node")
    with pytest.raises(InsufficientData,match="Nodal export.*Cell Center"):
        face_forces(d,REF,nodal)


def test_whole_wing_force_summary_pressure_viscous_total():
    from src.force_integration import wing_force_summary
    force=face_forces(flat_plate(),REF,CONV)
    result=wing_force_summary(force,REF)
    rows=result.table.set_index("component")
    np.testing.assert_allclose(rows.loc["pressure",["Fx_N","Fy_N","Fz_N"]].to_numpy(dtype=float),[0,0,30])
    np.testing.assert_allclose(rows.loc["viscous",["Fx_N","Fy_N","Fz_N"]].to_numpy(dtype=float),[2,0,0])
    assert rows.loc["total","lift_N"]==pytest.approx(30)
    assert rows.loc["total","drag_N"]==pytest.approx(2)
    assert rows.loc["total","resultant_N"]==pytest.approx(np.sqrt(904))


def test_force_resultant_uses_vector_sum_and_missing_viscous_is_not_total():
    from src.force_integration import wing_force_summary,FaceForces
    force=FaceForces(np.array([[0.,0.,10.],[0.,0.,-10.]]),None)
    result=wing_force_summary(force,REF)
    assert result.table.iloc[0].resultant_N==0
    assert result.table.component.tolist()==["pressure"]
    assert not result.metadata["total_available"]
