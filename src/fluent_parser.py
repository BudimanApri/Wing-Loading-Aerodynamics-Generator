"""Chunked Fluent ASCII ingestion with atomic, memory-mapped column caches."""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import time
import uuid

import numpy as np
import pandas as pd
from .file_selection import require_file

PARSER_VERSION = 2
ALIASES = {
    "x": ("x", "xcoordinate", "coordinatex"),
    "y": ("y", "ycoordinate", "coordinatey"),
    "z": ("z", "zcoordinate", "coordinatez"),
    "p": ("pressure", "staticpressure", "p"),
    "cp": ("pressurecoefficient", "cp"),
    "ax": ("xfacearea", "faceareax", "xcomponentoffaceareavector"),
    "ay": ("yfacearea", "faceareay", "ycomponentoffaceareavector"),
    "az": ("zfacearea", "faceareaz", "zcomponentoffaceareavector"),
    "tx": ("xwallshear", "wallshearx", "xwallshearstress"),
    "ty": ("ywallshear", "wallsheary", "ywallshearstress"),
    "tz": ("zwallshear", "wallshearz", "zwallshearstress"),
    "id": ("nodenumber", "cellnumber", "facenumber", "faceid", "cellid", "nodeid", "id"),
    "zone": ("zone", "zoneid", "zonename", "wallzone"),
    "surface": ("surface", "surfacename", "side"),
}


def canonical(name):
    name = re.sub(r"\([^)]*\)|\[[^]]*\]", "", name.lower())
    key = re.sub(r"[^a-z0-9]", "", name)
    return next((k for k, aliases in ALIASES.items() if key in aliases), None)


def inspect_file(path):
    """Inspect only a small prefix; fail explicitly on unsupported mesh formats."""
    path = require_file(path)
    with path.open(encoding="utf-8-sig", errors="strict") as stream:
        for line_number in range(100):
            line = stream.readline(65536)
            if not line:
                break
            sep = next((s for s in (",", ";", "\t") if s in line), None)
            names = next(csv.reader([line], delimiter=sep)) if sep else line.split()
            names = [n.strip().strip('"') for n in names]
            mapped = [canonical(n) for n in names]
            if all(k in mapped for k in ("x", "y", "z")):
                token = re.sub(r"[^a-z0-9]","",names[0].lower())
                native_layout = token in ("nodenumber","cellnumber","facenumber") and mapped[1:4]==["x","y","z"]
                parser_mapping = mapped.copy()
                roles = {}
                if native_layout:
                    for axis in ("x","y","z"):
                        occurrences = [i for i,key in enumerate(mapped) if key==axis]
                        quantities = []
                        for number,column in enumerate(occurrences[1:],start=1):
                            key = f"scalar_{axis}"+(f"_{number}" if number>1 else "")
                            parser_mapping[column]=key
                            quantities.append({"column_index_0based":column,"parsed_name":key})
                        roles[axis] = {"location_column_index_0based":occurrences[0],"quantity_columns":quantities}
                return {"path": str(path.resolve()), "header_line": line_number,
                        "delimiter": sep or "whitespace", "headers": names,
                        "mapping": mapped, "parser_mapping":parser_mapping,
                        "record_hint":"node" if token.startswith("node") else "cell_center" if token.startswith("cell") else "face" if token.startswith("face") else "unknown",
                        "coordinate_column_roles":roles,
                        "coordinate_policy":"Fluent leading location triplet is primary; separately requested coordinate quantities are preserved as scalar_x/scalar_y/scalar_z" if native_layout else "Repeated coordinate columns must agree; ambiguous layouts are not resolved automatically"}
    raise ValueError("No supported tabular x/y/z header in the first 100 lines. "
                     "Mesh/profile Lisp exports need topology-aware conversion.")


@dataclass
class Dataset:
    columns: dict
    metadata: dict

    def __len__(self):
        return len(self.columns["x"])

    def has(self, *keys):
        return all(k in self.columns for k in keys)

    def vector(self, keys=("x", "y", "z")):
        return np.column_stack([self.columns[k] for k in keys])

    def subset(self, mask):
        return Dataset({k: v[mask] for k, v in self.columns.items()}, self.metadata.copy())

    def to_frame(self):
        return pd.DataFrame(self.columns)


def load_cache(directory):
    directory = Path(directory)
    meta = json.loads((directory / "manifest.json").read_text())
    columns = {k: np.load(directory / f"{k}.npy", mmap_mode="r", allow_pickle=False)
               for k in meta["columns"]}
    return Dataset(columns, meta)


def read_fluent(path, cache_dir, chunk_size=100_000):
    """Preserve float64 precision; one raw pass, bounded ingestion memory.

    Cache key includes resolved path, size, nanosecond mtime and parser version.
    Do not change files in place while importing. Native Fluent location
    coordinates are separate from coordinate quantities. Ambiguous repeated
    coordinates must match. Invalid rows are never silently dropped.
    """
    path, cache_dir = require_file(path), Path(cache_dir)
    stat = path.stat()
    signature = [str(path.resolve()), stat.st_size, stat.st_mtime_ns, PARSER_VERSION]
    key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:24]
    target = cache_dir / key
    if (target / "manifest.json").exists():
        return load_cache(target)
    info = inspect_file(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging = cache_dir / f".building-{uuid.uuid4().hex}"
    staging.mkdir()
    groups = {}
    for i, key in enumerate(info["parser_mapping"]):
        if key:
            groups.setdefault(key, []).append(i)
    sep = r"\s+" if info["delimiter"] == "whitespace" else info["delimiter"]
    pieces, rows = [], 0
    coordinate_stats = {axis:{"compared_values":0,"different_values":0,"maximum_absolute_difference":0.,"squared_difference_sum":0.}
                        for axis,role in info["coordinate_column_roles"].items() if role["quantity_columns"]}
    try:
        reader = pd.read_csv(path, sep=sep, skiprows=info["header_line"] + 1,
                             header=None, names=list(range(len(info["headers"]))),
                             chunksize=chunk_size, skipinitialspace=True,
                             encoding="utf-8-sig", on_bad_lines="error")
        for index, frame in enumerate(reader):
            piece = staging / f"chunk-{index}"
            piece.mkdir()
            for key, indices in groups.items():
                if key in ("zone", "surface"):
                    if frame[indices[0]].isna().any():
                        raise ValueError(f"Missing {key} values in chunk {index}")
                    values = frame[indices[0]].astype(str).str.strip().to_numpy(dtype=str)
                else:
                    values = pd.to_numeric(frame[indices[0]], errors="raise").to_numpy(dtype=np.float64)
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite/missing {key} values in chunk {index}")
                    for duplicate in indices[1:]:
                        other = pd.to_numeric(frame[duplicate], errors="raise").to_numpy(dtype=float)
                        if not np.allclose(values, other, rtol=1e-10, atol=1e-12):
                            raise ValueError(f"Conflicting repeated {key} columns")
                np.save(piece / f"{key}.npy", values, allow_pickle=False)
            for axis,stats in coordinate_stats.items():
                role=info["coordinate_column_roles"][axis]
                primary=frame[role["location_column_index_0based"]].to_numpy(dtype=float)
                for quantity in role["quantity_columns"]:
                    other=frame[quantity["column_index_0based"]].to_numpy(dtype=float)
                    difference=other-primary
                    stats["compared_values"]+=len(primary)
                    stats["different_values"]+=int((~np.isclose(primary,other,rtol=1e-10,atol=1e-12)).sum())
                    stats["maximum_absolute_difference"]=max(stats["maximum_absolute_difference"],float(np.abs(difference).max()))
                    stats["squared_difference_sum"]+=float(difference@difference)
            pieces.append((piece, len(frame)))
            rows += len(frame)
        if not rows:
            raise ValueError("Export has no records")
        for key in groups:
            dtype = np.dtype(np.float64)
            if key in ("zone", "surface"):
                max_chars = 1
                for piece, _ in pieces:
                    source = np.load(piece / f"{key}.npy", mmap_mode="r")
                    max_chars = max(max_chars, source.dtype.itemsize // 4)
                    source._mmap.close()
                    del source
                dtype = np.dtype(f"U{max_chars}")
            combined = np.lib.format.open_memmap(staging / f"{key}.npy", mode="w+",
                                                 dtype=dtype, shape=(rows,))
            offset = 0
            for piece, count in pieces:
                source = np.load(piece / f"{key}.npy", mmap_mode="r")
                combined[offset:offset + count] = source
                source._mmap.close()
                del source
                offset += count
            combined.flush()
            combined._mmap.close()
            del combined
        for piece, _ in pieces:
            shutil.rmtree(piece)
        if path.stat().st_size != stat.st_size or path.stat().st_mtime_ns != stat.st_mtime_ns:
            raise ValueError("Input changed while reading; import again")
        for stats in coordinate_stats.values():
            stats["rms_difference"]=float(np.sqrt(stats.pop("squared_difference_sum")/stats["compared_values"]))
        info.update({"rows": rows, "columns": list(groups), "signature": signature,
                     "coordinate_quantity_differences":coordinate_stats,
                     "cache_format": "float64 column NPY memory maps", "parser_version": PARSER_VERSION})
        (staging / "manifest.json").write_text(json.dumps(info, indent=2))
        if target.exists():
            shutil.rmtree(staging)
        else:
            # Windows scanners can briefly hold newly-created files. All our
            # mappings are closed; bound retries for transient rename locks.
            for attempt in range(5):
                try:
                    staging.rename(target)
                    break
                except (PermissionError, FileExistsError):
                    if (target / "manifest.json").exists():
                        shutil.rmtree(staging)
                        break
                    if attempt == 4:
                        raise
                    time.sleep(.05 * 2**attempt)
        return load_cache(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def combine(datasets):
    """Combine exports only with the same physical fields; preserve source IDs."""
    if not datasets:
        raise ValueError("Select at least one dataset")
    fields = set(datasets[0].columns) - {"source"}
    if any(set(d.columns) - {"source"} != fields for d in datasets):
        raise ValueError("Combined exports must have matching fields; analyze separately otherwise")
    columns = {k: np.concatenate([d.columns[k] for d in datasets]) for k in fields}
    columns["source"] = np.concatenate([np.full(len(d), i, dtype=int) for i, d in enumerate(datasets)])
    return Dataset(columns, {"sources": [d.metadata for d in datasets], "rows": len(columns["x"])})
