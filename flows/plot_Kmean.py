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
        value="kmeans_animation.html"
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

    # try:
    #     # Call your K-means function here:
    #     kmean_clustering(
    #         csv_path, str(feat1), str(feat2),
    #         k, n_iters,
    #         out_csv, out_html, out_png
    #     )
    #     Logger.info("Clustering complete!")
    # except Exception as e:
    #     Logger.error(f"Clustering failed: {e}")
    #     return

    np.random.seed(42)

    # Read the CSV
    data = pd.read_csv(csv_path)

    # Fill missing values
    for col in data.columns:
        if data[col].dtype == np.float64:
            data[col] = data[col].fillna(data[col].mean())
        else:
            mode = data[col].mode()
            data[col] = data[col].fillna(mode[0] if not mode.empty else np.nan)

    # Keep originals
    orig1 = data[feat1].copy()
    orig2 = data[feat2].copy()

    # Prepare numeric matrix for clustering
    X = data[[feat1, feat2]].copy()

    encoders = {}
    for col in [feat1, feat2]:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    X = X.values.astype(float)
            
    # Define K-Means functions (initialize, assign, update)
    def initialize_centroids(X, k):
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        return np.random.uniform(mins, maxs, (k, X.shape[1]))

    def assign_clusters(X, centroids):
        labels = []
        for point in X:
            dists = [np.linalg.norm(point - c) for c in centroids]
            labels.append(np.argmin(dists))
        return np.array(labels)

    def update_centroids(X, labels, k):
        centroids = []
        for i in range(k):
            points = X[labels == i]
            if len(points) > 0:
                new_center = points.mean(axis=0)
            else:
                new_center = X[np.random.choice(len(X))]
            centroids.append(new_center)
        return np.array(centroids)

    # Run K-Means and record history
    centroids = initialize_centroids(X, k)
    history = []

    for it in range(1, n_iters + 1):
        labels = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, k)
        history.append((it, labels.copy(), centroids.copy()))

    # Save labels to CSV on last iteration
    data['cluster_label'] = history[-1][1]
    data[feat1] = orig1
    data[feat2] = orig2
    X = data[[feat1, feat2]].values
    data.to_csv(csv_output, index=True)
    print("Saved updated CSV to dataset_with_clusters.csv")

    # Prepare Plotly frames with iteration number in title
    frames = []
    for iter_num, labels_i, cents_i in history:
        
        labels_i = np.asarray(labels_i, dtype=int)
        scatter = go.Scatter(
            x=X[:, 0], y=X[:, 1], mode='markers',
            marker=dict(color=labels_i, showscale=False),
            name='Data'
        )

        # Inverse-transform any encoded centroid dims
        arr = np.asarray(cents_i, dtype=float).reshape(-1, 2)
        cent_plot = arr.copy()
        if feat1 in encoders:
            cent_plot[:, 0] = encoders[feat1].inverse_transform(
                cent_plot[:, 0].round().astype(int)
            )
        if feat2 in encoders:
            cent_plot[:, 1] = encoders[feat2].inverse_transform(
                cent_plot[:, 1].round().astype(int)
            )
        cent_scatter = go.Scatter(
            x=cent_plot[:, 0], y=cent_plot[:, 1], mode='markers',
            marker=dict(symbol='x', size=12, color='black'),
            name='Centroids'
        )
        frames.append(go.Frame(data=[scatter, cent_scatter], name=f'iter{iter_num}'))

    layout = go.Layout(
        title=dict(text=f'K-Means Clustering Animation {feat1} vs {feat2}, k={k}, Iteration 1'),
        xaxis=dict(title=feat1),
        yaxis=dict(title=feat2),
        updatemenus=[dict(
            type='buttons', showactive=False,
            buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}])]
        )]
    )
    fig = go.Figure(data=frames[0].data, layout=layout, frames=frames)

    # Update title per frame
    for f in fig.frames:
        f.layout = {'title': f'K-Means Clustering Animation {feat1} vs {feat2}, k={k}, Iteration {f.name.replace("iter","")}' }

    # Show interactive animation
    # fig.show()

    # Save last frame as PNG
    last_frame_fig = go.Figure(data=fig.frames[-1].data, layout=go.Layout(
        title=fig.frames[-1].layout['title'],
        xaxis=fig.layout.xaxis,
        yaxis=fig.layout.yaxis
    ))

    last_frame_fig.write_image(png_output, format='png', width=1200, height=800)
    print(f"Saved last iteration as PNG to {png_output}")

    fig.write_html(
        html_output,
        include_plotlyjs=True,   
        full_html=True,
        auto_play=False
    )
    print(f"Saved animated K-mean as offline HTML to {html_output}")
