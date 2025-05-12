import os
from .Kmean_Clustering import kmean_clustering
from onecode import file_input, dropdown, text_input, file_output, Logger

def run():
    # CSV upload
    csv_path = file_input(
        key="csv_file",
        label="Upload CSV File",
        value=""
    )
    if not csv_path or not os.path.exists(csv_path):
        Logger.info("Please upload a valid CSV to continue.")
        return

    # Feature selection
    feat1 = text_input(
        key="feat1",
        label="Feature 1 (X axis)",
        value=""
    )
    feat2 = text_input(
        key="feat2",
        label="Feature 2 (Y axis)",
        value=""
    )

    # Other parameters
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

    # Outputs
    out_csv = file_output(
        key="out_csv",
        label="Save Clustered CSV",
        value="clustered_output.csv"
    )
    out_html = file_output(
        key="out_html",
        label="Save Plotly HTML",
        value="kmeans_animation.html"
    )
    out_png = file_output(
        key="out_png",
        label="Save Snapshot PNG",
        value="clustered_image.png"
    )

    # Run clustering when all inputs are set
    try:
        k = int(k_str)
        n_iters = int(it_str)
    except ValueError:
        Logger.error("k and iterations must be integers")
        return

    try:
        # Call K-means function
        kmean_clustering(
            csv_path, str(feat1), str(feat2),
            k, n_iters,
            out_csv, out_html, out_png
        )
        Logger.info("Clustering complete!")
    except Exception as e:
        Logger.error(f"Clustering failed: {e}")
        return
