import os
from .Kmean_Clustering import kmean_clustering
from onecode import csv_reader, file_input, dropdown, text_input, number_input, file_output, slider, Logger

def run():

    # CSV upload
    csv = csv_reader(
           key="csv_file",
           value=None,
           label="Upload CSV File"
    )
    Logger.info(f"{csv}")
    
    if not csv or not os.path.exists(csv):
        Logger.info("Please upload a valid CSV to continue.")
        return

    # Feature selection
    feat1 = dropdown(
        'Feature 1 (X axis)',
        value=None,
        options="$csv_file$.columns" 
    )

    feat2 = dropdown(
        'Feature 2 (Y axis)',
        value=None,
        options="$csv_file$.columns" 
    )

    # Other parameters
    n_cluters = slider(
        'Number of Clusters (k)',
        3,
        min=2,
        max=30,
        step=1
    )

    n_itr = number_input(
        'Number of Iterations',
        10,
        min=1,
        step=1
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
        k = int(n_cluters)
        n_iters = int(n_itr)
    except ValueError:
        Logger.error("k and iterations must be integers")
        return

    Logger.info(f"X axis = {feat1} | Y axis = {feat2}")
    Logger.info(f"N clusters  = {n_cluters}")
    Logger.info(f"N Iterations  = {n_itr}")
    Logger.info(f"Data Frame CSV description:")
    print(csv.describe())


    try:
        # Call K-means function
        kmean_clustering(
            csv, str(feat1), str(feat2),
            k, n_iters,
            out_csv, out_html, out_png
        )
        Logger.info("Clustering complete!")
    except Exception as e:
        Logger.error(f"Clustering failed: {e}")
        return
