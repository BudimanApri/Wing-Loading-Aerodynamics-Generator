import pytest
from src.file_selection import require_file,split_local_inputs,folder_exports
from src.fluent_parser import read_fluent


def test_folder_is_not_opened_as_a_file(tmp_path):
    with pytest.raises(ValueError,match="This is a folder"):
        read_fluent(tmp_path,tmp_path/"cache")


def test_folder_exports_include_extensionless_fluent_files_only(tmp_path):
    for name in ("Front wing","Rear wing.csv"):
        (tmp_path/name).write_text("x-coordinate,y-coordinate,z-coordinate,pressure\n0,0,0,1")
    (tmp_path/"notes.txt").write_text("not CFD data")
    (tmp_path/"case.dat.h5").write_bytes(b"binary")
    sub=tmp_path/"nested"; sub.mkdir()
    (sub/"third.csv").write_text("x,y,z,p\n0,0,0,1")
    from pathlib import Path
    assert {Path(p).name for p in folder_exports(tmp_path)}=={"Front wing","Rear wing.csv"}


def test_quoted_inputs_split_files_and_folders_without_duplicates(tmp_path):
    file=tmp_path/"wing.csv";file.write_text("x,y,z\n0,0,0")
    files,folders=split_local_inputs(f'"{file}"\n{tmp_path}\n{file}')
    assert files==[str(file)]
    assert folders==[str(tmp_path)]
    assert require_file(f'"{file}"')==file


def test_missing_file_gives_actionable_message(tmp_path):
    with pytest.raises(ValueError,match="not found"):
        split_local_inputs(str(tmp_path/"missing.csv"))
