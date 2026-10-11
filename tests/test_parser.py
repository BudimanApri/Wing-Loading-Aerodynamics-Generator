from pathlib import Path
import numpy as np
import pytest
from src.fluent_parser import read_fluent, inspect_file, combine


@pytest.mark.parametrize("delimiter", [",", ";", "\t", " "])
def test_delimiters_preamble_aliases_and_cache(tmp_path, delimiter):
    path = tmp_path / "wing.txt"
    path.write_text("# Export\n"+delimiter.join(["nodenumber", "x-coordinate", "y-coordinate", "z-coordinate", "pressure", "x-coordinate"])+
                    "\n"+delimiter.join(["1", "0.2", "1", "0.1", "-10", "0.2"])+
                    "\n"+delimiter.join(["2", "0.8", "2", "0.1", "20", "0.8"]))
    result = read_fluent(path, tmp_path / "cache", chunk_size=1)
    np.testing.assert_allclose(result.columns["p"], [-10,20])
    assert result.metadata["record_hint"] == "node"
    assert inspect_file(path)["header_line"] == 1
    assert isinstance(read_fluent(path,tmp_path/"cache").columns["x"], np.memmap)
    assert len(list((tmp_path/"cache").iterdir())) == 1


def test_duplicate_coordinates_conflict_and_bad_records(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x,y,z,x\n0,0,0,1")
    with pytest.raises(ValueError, match="Conflicting"):
        read_fluent(path, tmp_path/"cache")
    path.write_text("x,y,z,p\n0,0,0,nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        read_fluent(path, tmp_path/"cache")
    assert not list((tmp_path/"cache").glob(".building-*"))


def test_strings_and_missing_fields(tmp_path):
    path = tmp_path / "wing.csv"
    path.write_text('"X-coordinate [m]",Y-coordinate,Z-coordinate,Static Pressure [Pa],zone,side\n0,1,2,3,wall-upper,upper\n1,1,2,4,wall-lower,lower')
    data = read_fluent(path, tmp_path/"cache", 1)
    assert data.columns["zone"].tolist() == ["wall-upper", "wall-lower"]
    assert data.has("p") and not data.has("ax")


def test_cache_invalidation_and_incompatible_files(tmp_path):
    path = tmp_path/"wing.csv"
    path.write_text("x,y,z,p\n0,0,0,1")
    one = read_fluent(path, tmp_path/"cache")
    path.write_text("x,y,z,p\n0,0,0,100")
    two = read_fluent(path, tmp_path/"cache")
    assert two.columns["p"][0] == 100
    assert len(list((tmp_path/"cache").iterdir())) == 2
    del two.columns["p"]
    with pytest.raises(ValueError, match="matching"):
        combine([one,two])


def test_fluent_cell_export_keeps_location_and_coordinate_quantities_distinct(tmp_path):
    path=tmp_path/"face.csv"
    path.write_text("cellnumber,x-coordinate,y-coordinate,z-coordinate,pressure,x-coordinate,y-coordinate,z-coordinate,x-face-area,y-face-area,z-face-area\n"
                    "1,.5,1,0,10,.5001,1.01,.001,0,0,.1\n"
                    "2,.6,1,0,20,.6002,1.01,.001,0,0,.1")
    data=read_fluent(path,tmp_path/"cache",chunk_size=1)
    np.testing.assert_allclose(data.columns["x"],[.5,.6])
    np.testing.assert_allclose(data.columns["scalar_x"],[.5001,.6002])
    np.testing.assert_allclose(data.columns["scalar_z"],[.001,.001])
    np.testing.assert_allclose(data.columns["id"],[1,2])
    assert data.metadata["record_hint"]=="cell_center"
    assert data.metadata["coordinate_quantity_differences"]["x"]["different_values"]==2
    assert data.metadata["coordinate_quantity_differences"]["x"]["maximum_absolute_difference"]==pytest.approx(.0002)


def test_fluent_node_export_preserves_coordinate_quantities_even_when_equal(tmp_path):
    path=tmp_path/"node.csv"
    path.write_text("nodenumber,x-coordinate,y-coordinate,z-coordinate,pressure,x-coordinate\n1,.5,1,0,10,.5")
    data=read_fluent(path,tmp_path/"cache")
    assert data.metadata["coordinate_quantity_differences"]["x"]["different_values"]==0
    np.testing.assert_allclose(data.columns["scalar_x"],data.columns["x"])


def test_force_moment_uses_face_location_not_requested_coordinate_quantity(tmp_path):
    from src.force_integration import Reference,ForceConvention,face_forces,integrate
    path=tmp_path/"moment.csv"
    path.write_text("cellnumber,x-coordinate,y-coordinate,z-coordinate,pressure,x-coordinate,y-coordinate,z-coordinate,x-face-area,y-face-area,z-face-area\n"
                    "1,.5,1,0,-10,.7,1.2,.1,0,0,1")
    data=read_fluent(path,tmp_path/"cache")
    force=face_forces(data,Reference(1,2,0,0),ForceConvention(True,"body_to_fluid","unverified",True,True,True))
    result=integrate(force.pressure,data.vector())
    np.testing.assert_allclose(result["moment_Nm"],[10,-5,0])
