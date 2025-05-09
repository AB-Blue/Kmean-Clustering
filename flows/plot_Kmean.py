import os
import pandas as pd
from .Kmean_Clustering import kmean_clustering
from onecode import file_input, dropdown, text_input, file_output, Logger

def run():
    # Step 1: CSV upload
    csv_path = file_input(
        key="csv_file",
        label="1) Upload CSV File",
        value=""
    )
    if not csv_path or not os.path.exists(csv_path):
        Logger.info("Please upload a valid CSV to continue.")
        return

    # Step 2: Feature selectors as LineEdit
    feat1 = text_input(
        key="feat1",
        label="2) Feature 1 (X axis)",
        value=""
    )
    feat2 = text_input(
        key="feat2",
        label="3) Feature 2 (Y axis)",
        value=""
    )

    # Step 3: Other parameters
    k_str = text_input(
        key="n_clusters",
        label="4) Number of Clusters (k)",
        value="3"
    )
    it_str = text_input(
        key="n_iters",
        label="5) Number of Iterations",
        value="10"
    )

    # Step 4: Outputs
    out_csv = file_output(
        key="out_csv",
        label="6) Save Clustered CSV",
        value="clustered_output.csv"
    )
    out_html = file_output(
        key="out_html",
        label="7) Save Plotly HTML",
        value="kmeans_plot.html"
    )
    out_png = file_output(
        key="out_png",
        label="8) Save Snapshot PNG",
        value="clustered_image.png"
    )

    # Step 5: Run clustering when all inputs are set
    try:
        k = int(k_str)
        n_iters = int(it_str)
    except ValueError:
        Logger.error("k and iterations must be integers")
        return

    try:
        # Call your K-means function here:
        kmean_clustering(
            csv_path, str(feat1), str(feat2),
            k, n_iters,
            out_csv, out_html, out_png
        )
        Logger.info("Clustering complete!")
    except Exception as e:
        Logger.error(f"Clustering failed: {e}")
")
        return

