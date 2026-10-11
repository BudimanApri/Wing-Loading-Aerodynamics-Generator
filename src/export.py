"""Portable CSV + JSON metadata; optional spreadsheet and Plotly artifacts."""
from dataclasses import asdict, is_dataclass
from io import BytesIO
import json
import zipfile
import numpy as np


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def metadata_text(metadata):
    return json.dumps(metadata, indent=2, default=json_default)


def csv_bytes(table, metadata):
    comments = "\n".join("# " + line for line in metadata_text(metadata).splitlines())
    return (comments + "\n" + table.to_csv(index=False)).encode("utf-8-sig")


def bundle_results(results, metadata):
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", metadata_text(metadata))
        for name, result in results.items():
            archive.writestr(name + ".csv", csv_bytes(result.table, {**metadata, **result.metadata}))
    return output.getvalue()


def excel_bytes(table, metadata):
    import pandas as pd
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Results", index=False)
        pd.DataFrame({"metadata_json": metadata_text(metadata).splitlines()}).to_excel(writer, sheet_name="References", index=False)
    return output.getvalue()


def figure_bytes(figure, format="html"):
    if format == "html":
        return figure.to_html(include_plotlyjs=True).encode()
    if format not in ("png", "svg"):
        raise ValueError("Plot export must be HTML, PNG or SVG")
    return figure.to_image(format=format, width=1400, height=850, scale=2)


def to_parquet(data, path):
    """Optional conversion, bounded chunks; preserved float64 columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    with_metadata = {b"hale_metadata": metadata_text(data.metadata).encode()}
    writer = None
    try:
        for start in range(0, len(data), 100_000):
            table = pa.table({k: v[start:start+100_000] for k, v in data.columns.items()})
            table = table.replace_schema_metadata(with_metadata)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer:
            writer.close()
