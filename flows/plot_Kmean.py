import os
import pandas as pd
from .Kmean_Clustering import kmean_clustering
from onecode import file_input, dropdown, text_input, file_output, Logger


def run():
    # 1) CSV upload
    csv_path = file_input(
        key="csv_file",
        label="Upload CSV File",
        value=""
    )
    if not os.path.exists(csv_path):
        Logger.error(f"CSV not found: {csv_path}")
        return

    # 2) Read header + DataFrame
    # try:
    #     df = pd.read_csv(csv_path)
    #     cols = list(df.columns)
    # except:
    #     cols = ["feat1", "feat2"]

    # 3) Feature selectors
    feat1 = dropdown(
        key="feat1",
        label="Feature 1 (X axis)",
        # options=cols,
        # value=cols[0]
        value=""
    )
    feat2 = dropdown(
        key="feat2",
        label="Feature 2 (Y axis)",
        # options=cols,
        # value=cols[1] if len(cols) > 1 else cols[0]
        value=""
    )

    # 4) K and iterations
    k_str = text_input(
        key="n_clusters",
        label="Number of Clusters (k)",
        value="3"
    )
    it_str = text_input(
        key="n_iters",
        label="Number of Iterations",
        value="10"
    )

    # 5) Outputs
    out_csv = file_output(
        key="out_csv",
        label="Clustered CSV",
        value="clustered_output.csv"
    )
    out_html = file_output(
        key="out_html",
        label="Plotly HTML",
        value="kmeans_plot.html"
    )
    out_png = file_output(
        key="out_png",
        label="Clustered PNG",
        value="clustered_image.png"
    )

    # 6) Parse & validate numeric inputs
    try:
        k = int(k_str)
        n_iters = int(it_str)
    except ValueError:
        Logger.error("k and iterations must be integers")
        return

    # Logger.info(f"Saved Plotly HTML to {out_html}")

    try:
        kmean_clustering(csv_path, feat1, feat2, k, n_iters, out_csv, out_html, out_png)
    except ValueError:
        Logger.error("Please check parameters")
        return
