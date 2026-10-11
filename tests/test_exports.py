import json
from io import BytesIO
import zipfile
import numpy as np
import pandas as pd
import pytest
from src.export import csv_bytes, excel_bytes, bundle_results, to_parquet, figure_bytes
from src.spanwise_analysis import Distribution
from src.configuration import default_case
from src.force_integration import Reference


def test_corrected_supplied_reference():
    case=default_case("wing")
    ref=Reference(case["rho"],case["speed"],case["p_inf"],case["p_force_ref"],tuple(case["lift"]),tuple(case["freestream"]))
    assert case["aoa_deg"]==2
    assert ref.q==pytest.approx(61.25)
    np.testing.assert_allclose(ref.drag,[.999390827,0,-.034899497],atol=1e-9)
    np.testing.assert_allclose(ref.lift,[-.034899497,0,-.999390827],atol=1e-9)
    assert not case["face_centered_verified"]


def test_new_case_angle_directions_and_filename_suggestion():
    from src.configuration import suggested_aoa
    assert suggested_aoa("Front wing AoA 7")==7
    assert suggested_aoa("Rear wing AoA-3.5")==-3.5
    case=default_case("front",7)
    assert case["aoa_deg"]==7
    alpha=np.deg2rad(7)
    np.testing.assert_allclose(case["freestream"],[np.cos(alpha),0,-np.sin(alpha)])
    np.testing.assert_allclose(case["lift"],[-np.sin(alpha),0,-np.cos(alpha)])


def test_csv_metadata_and_archive():
    table=pd.DataFrame({"lift_N_per_m":[1.,2.]})
    meta={"q_Pa":61.25,"sign":"up positive","reference":{"AoA_deg":2}}
    content=csv_bytes(table,meta)
    recovered=pd.read_csv(BytesIO(content),comment="#")
    np.testing.assert_allclose(recovered.lift_N_per_m,[1,2])
    archive=bundle_results({"span":Distribution(table,{})},meta)
    with zipfile.ZipFile(BytesIO(archive)) as z:
        assert json.loads(z.read("metadata.json"))["reference"]["AoA_deg"]==2
        assert "span.csv" in z.namelist()


def test_excel_metadata():
    pytest.importorskip("openpyxl")
    result=excel_bytes(pd.DataFrame({"lift_N":[1,2]}),{"q_Pa":61.25})
    sheets=pd.read_excel(BytesIO(result),sheet_name=None)
    assert set(sheets)=={"Results","References"}


def test_optional_parquet_roundtrip(uniform_wing,tmp_path):
    pq=pytest.importorskip("pyarrow.parquet")
    path=tmp_path/"wing.parquet"
    to_parquet(uniform_wing,path)
    table=pq.read_table(path)
    assert table.num_rows==len(uniform_wing)
    np.testing.assert_allclose(table.column("p").to_numpy(),uniform_wing.columns["p"])
    assert b"hale_metadata" in table.schema.metadata
