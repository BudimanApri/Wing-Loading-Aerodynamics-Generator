import numpy as np
import plotly.graph_objects as go


def line_plot(table, x, ys, labels=None, invert_y=False, title="", x_label=None):
    figure = go.Figure()
    labels = labels or {}
    groups = table.groupby("panel", sort=False) if "panel" in table else [("", table)]
    for group, rows in groups:
        for y in ys:
            figure.add_trace(go.Scatter(x=rows[x], y=rows[y], mode="lines+markers",
                name=(f"{group} " if group else "") + labels.get(y, y), connectgaps=False))
    figure.update_layout(title=title, xaxis_title=x_label or x,
        yaxis_title=", ".join(labels.get(y, y) for y in ys), template="plotly_white",
        hovermode="x unified", legend=dict(orientation="h"))
    if invert_y:
        figure.update_yaxes(autorange="reversed")
    return figure


def compare_distributions(results, x, y, title=""):
    figure = go.Figure()
    for name, result in results.items():
        groups = result.table.groupby("panel", sort=False) if "panel" in result.table else [("", result.table)]
        for panel, rows in groups:
            figure.add_trace(go.Scatter(x=rows[x], y=rows[y], mode="lines", name=f"{name} {panel}", connectgaps=False))
    figure.update_layout(title=title, xaxis_title=x, yaxis_title=y,
                         template="plotly_white", hovermode="x unified")
    return figure
