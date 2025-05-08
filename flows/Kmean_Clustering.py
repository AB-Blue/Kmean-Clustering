# import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# import imageio
# import kaleido
# import narwhals
# import imageio_ffmpeg as ffm


def kmean_clustering(data, feat1, feat2, k, n_iters, csv_output, html_output, png_output):
    # Make the results reproducible
    np.random.seed(42)

    # Fill missing values
    for col in data.columns:
        if data[col].dtype == np.float64:
            data[col] = data[col].fillna(data[col].mean())
        else:
            mode = data[col].mode()
            data[col] = data[col].fillna(mode[0] if not mode.empty else np.nan)

    # data = data[data['height']>100]
    X = data[[feat1, feat2]].values

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
    data.to_csv(csv_output, index=True)
    print("Saved updated CSV to dataset_with_clusters.csv")

    # Prepare Plotly frames with iteration number in title
    frames = []
    for iter_num, labels_i, cents_i in history:
        scatter = go.Scatter(
            x=X[:, 0], y=X[:, 1], mode='markers',
            marker=dict(color=labels_i, showscale=False),
            name='Data'
        )
        cent_scatter = go.Scatter(
            x=cents_i[:, 0], y=cents_i[:, 1], mode='markers',
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

    last_frame_fig.write_image(png_output, format='png')
    print(f"Saved last iteration as PNG to {png_output}")

    fig.write_html(
    html_output,
    include_plotlyjs='cdn',    # or True for full self-contained file
    auto_play=True            
    )
    print(f"Saved animated K-mean as HTML file to {html_output}")


if __name__ == '__main__': 
    
    # Load the open-source dataset
    url = 'https://vincentarelbundock.github.io/Rdatasets/csv/carData/Davis.csv'
    data = pd.read_csv(url, index_col=0)
    feat1 = 'height' #'repwt'
    feat2 = 'weight' #'repht'
    k = 3
    n_iters = 10

    csv_output = 'E:\AB\Visualization\DeepLime\Kmean_clustering_May_2025_AB\csv_output.csv'
    html_output = 'E:\AB\Visualization\DeepLime\Kmean_clustering_May_2025_AB\html_output.html'
    png_output = 'E:\AB\Visualization\DeepLime\Kmean_clustering_May_2025_AB\png_output.png'

    kmean_clustering(data, feat1, feat2, k, n_iters, csv_output, html_output, png_output)
