"""Local file selection: folders are browsed, never passed to the parser."""
from pathlib import Path

EXPORT_SUFFIXES = {"", ".csv", ".txt", ".asc", ".ascii", ".tsv", ".dat"}


def local_path(value):
    if not isinstance(value,(str,Path)) or not str(value).strip():
        raise ValueError("Enter a local CFD file or folder path")
    text=str(value).strip()
    if len(text)>1 and text[0]==text[-1] and text[0] in "\"'":
        text=text[1:-1]
    return Path(text).expanduser().resolve()


def require_file(value):
    path=local_path(value)
    if path.is_dir():
        raise ValueError(f"This is a folder: {path}. Select a CFD file inside it, or use the folder selector under Add a wing or CFD case.")
    if not path.is_file():
        raise ValueError(f"CFD file not found: {path}")
    return path


def folder_candidates(value):
    path=local_path(value)
    if not path.is_dir():
        raise ValueError(f"Folder not found: {path}")
    # Immediate files only; do not descend into caches or other application data.
    return sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXPORT_SUFFIXES),key=lambda p:p.name.lower())


def folder_exports(value):
    from .fluent_parser import inspect_file
    found=[]
    for path in folder_candidates(value):
        try:
            inspect_file(path)
        except (ValueError,OSError,UnicodeError):
            continue
        found.append(str(path))
    return found


def split_local_inputs(text):
    files,folders=[],[]
    for line in text.splitlines():
        if not line.strip():
            continue
        path=local_path(line)
        if path.is_dir():
            if str(path) not in folders:
                folders.append(str(path))
        elif path.is_file():
            if str(path) not in files:
                files.append(str(path))
        else:
            raise ValueError(f"File or folder not found: {path}")
    return files,folders
