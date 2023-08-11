import os
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots

blue = "#4353de"
pink = "#de43a0"
yellow = "#dece43"
green = "green"


def plot_result(ts_df, sensor, args):
    plot_dir = args["plot"]["directory"] or "plots"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(
        go.Scatter(
            x=ts_df.index,
            y=ts_df["data"],
            name="Values",
            line=dict(color=blue),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ts_df.index,
            y=ts_df["imputed_data"],
            name="Imputed values",
            line=dict(color=pink),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ts_df.index,
            y=ts_df["anomaly_score"],
            name="Anomaly score",
            line=dict(color=yellow),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ts_df.index,
            y=ts_df["anomaly_label"],
            name="Anomaly label",
            line=dict(color=green),
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        title_text=sensor["name"],
        xaxis4=dict(rangeslider=dict(visible=True), type="date"),
    )

    plot(
        fig,
        filename=f"{plot_dir}/{sensor['name']}.html",
        auto_open=False,
    )
