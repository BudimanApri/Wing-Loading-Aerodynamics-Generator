"""Pressure and viscous forces on the BODY, with explicit export conventions."""
from dataclasses import dataclass
import numpy as np


class InsufficientData(ValueError):
    pass


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError("Direction must be a finite nonzero 3-vector")
    return vector / np.linalg.norm(vector)


@dataclass(frozen=True)
class Reference:
    rho: float
    speed: float
    p_inf: float
    p_force_ref: float
    lift: tuple = (0., 0., 1.)
    drag: tuple = (1., 0., 0.)

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.rho, self.speed, self.p_inf, self.p_force_ref)):
            raise ValueError("Reference conditions must be finite")
        if self.rho <= 0 or self.speed <= 0:
            raise ValueError("Density and speed must be positive")
        if abs(unit(self.lift) @ unit(self.drag)) > 1e-6:
            raise ValueError("Lift and freestream drag directions must be perpendicular")

    @property
    def q(self):
        return .5 * self.rho * self.speed ** 2


@dataclass(frozen=True)
class ForceConvention:
    face_centered_verified: bool = False
    area_orientation: str = "unverified"
    shear_direction: str = "unverified"
    pressure_basis_verified: bool = False
    si_units_verified: bool = False
    unique_wall_faces_verified: bool = False
    export_location: str = "unconfirmed"


@dataclass
class FaceForces:
    pressure: np.ndarray
    viscous: np.ndarray | None

    @property
    def total(self):
        return self.pressure if self.viscous is None else self.pressure + self.viscous

    def component(self, choice):
        if choice == "pressure":
            return self.pressure
        if choice == "viscous":
            if self.viscous is None:
                raise InsufficientData("Viscous traction is unavailable or unverified")
            return self.viscous
        if choice == "total":
            if self.viscous is None:
                raise InsufficientData("Total force requires verified viscous traction; select pressure-only")
            return self.total
        raise ValueError("Unknown force component")


def face_forces(data, reference, convention):
    if convention.export_location == "node":
        raise InsufficientData("Nodal export: pressure plots are available, but aerodynamic/structural face-force integration requires a new Fluent ASCII export with Location = Cell Center on the wing wall surfaces. Removing repeated nodes does not supply independent face values")
    if data.metadata.get("exact_duplicate_records", 0):
        count = data.metadata["exact_duplicate_records"]
        raise InsufficientData(f"{count} extra physical records are duplicated in force-input fields. Review CFD Data Manager → Repeated physical records; verify export location and mesh identity before summing forces")
    required = ("p", "ax", "ay", "az")
    missing = [k for k in required if not data.has(k)]
    if missing:
        raise InsufficientData(f"Force integration requires missing fields: {missing}")
    checks = {"face-centered export": convention.face_centered_verified,
              "pressure basis/reference": convention.pressure_basis_verified,
              "SI units": convention.si_units_verified,
              "unique wall faces, no wall-shadow duplication": convention.unique_wall_faces_verified}
    pending = [k for k, v in checks.items() if not v]
    if pending:
        raise InsufficientData("Verify " + "; ".join(pending))
    if convention.area_orientation not in ("body_to_fluid", "fluid_to_body"):
        raise InsufficientData("Verify the exported area-vector direction")
    area = data.vector(("ax", "ay", "az"))
    magnitude = np.linalg.norm(area, axis=1)
    if not np.isfinite(area).all() or np.any(magnitude <= 0):
        raise InsufficientData("Nonfinite or zero face-area vectors")
    pressure = data.columns["p"] - reference.p_force_ref
    if not np.isfinite(pressure).all():
        raise InsufficientData("Nonfinite pressures")
    sign = -1. if convention.area_orientation == "body_to_fluid" else 1.
    fp = sign * pressure[:, None] * area
    fv = None
    if data.has("tx", "ty", "tz") and convention.shear_direction in ("on_body", "on_fluid"):
        shear = data.vector(("tx", "ty", "tz"))
        if not np.isfinite(shear).all():
            raise InsufficientData("Nonfinite wall shear")
        fv = (1. if convention.shear_direction == "on_body" else -1.) * shear * magnitude[:, None]
    return FaceForces(fp, fv)


def integrate(force, positions, reference_point=(0., 0., 0.)):
    force, positions = np.asarray(force), np.asarray(positions)
    if force.shape != positions.shape or force.ndim != 2 or force.shape[1] != 3:
        raise ValueError("Force and position arrays must be matching N x 3 arrays")
    return {"force_N": force.sum(axis=0),
            "moment_Nm": np.cross(positions - reference_point, force).sum(axis=0)}


def wing_force_summary(forces, reference):
    """Integrate each force component over ALL supplied wing faces in N.

    Resultant is norm(sum of vectors), not sum of individual face magnitudes.
    Total is reported only with a known viscous contribution.
    """
    import pandas as pd
    from .spanwise_analysis import Distribution
    parts = [("pressure",forces.pressure)]
    if forces.viscous is not None:
        parts += [("viscous",forces.viscous),("total",forces.component("total"))]
    rows=[]
    for label,array in parts:
        summed=np.sum(array,axis=0,dtype=np.float64)
        rows.append({"component":label,"Fx_N":float(summed[0]),"Fy_N":float(summed[1]),"Fz_N":float(summed[2]),
            "lift_N":float(summed@unit(reference.lift)),"drag_N":float(summed@unit(reference.drag)),
            "resultant_N":float(np.linalg.norm(summed)),"face_count":len(array)})
    return Distribution(pd.DataFrame(rows),{
        "scope":"All exported faces in the selected wing/case after source/zone filtering; no symmetry scaling",
        "units":"N","resultant_definition":"Magnitude of the integrated 3D force vector; not a sum of face magnitudes",
        "total_available":forces.viscous is not None,
        "total_definition":"Pressure plus verified viscous traction" if forces.viscous is not None else "Total unavailable: missing or unverified viscous traction; pressure is reported separately",
        "lift_direction":unit(reference.lift).tolist(),"drag_direction":unit(reference.drag).tolist()})
