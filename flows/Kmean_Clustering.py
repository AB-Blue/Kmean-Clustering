
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import kaleido


def kmean_clustering(csv_path, feat1, feat2, k, n_iters, csv_output, html_output, png_output):
    # Make the results reproducible
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

    cat_cols = [
    c for c in (feat1, feat2)
    if data[c].dtype == 'object' or data[c].dtype.name == 'category'
    ]
    
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
        # centroids = []
        # for i in range(k):
        #     points = X[labels == i]
        #     if len(points) > 0:
        #         new_center = points.mean(axis=0)
        #     else:
        #         new_center = X[np.random.choice(len(X))]
        #     centroids.append(new_center)
        # return np.array(centroids)


        new_cents = []
        for i in range(k):
            pts = data.loc[labels == i, [feat1, feat2]]
            # numeric means
            num_means = pts.select_dtypes(include='number').mean().values
            # categorical modes
            cat_modes = [
                pts[c].mode().iloc[0]
                if not pts[c].mode().empty else np.nan
                for c in cat_cols
            ]
            # assemble centroid in original order
            cent = []
            for col in (feat1, feat2):
                if col in cat_cols:
                    cent.append(cat_modes[cat_cols.index(col)])
                else:
                    # match numeric column position
                    num_idx = list(pts.select_dtypes('number').columns).index(col)
                    cent.append(num_means[num_idx])
            new_cents.append(cent)
        centroids = np.array(new_cents, dtype=object)
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


if __name__ == '__main__': 
    
    # Load the open-source dataset
    url = 'https://vincentarelbundock.github.io/Rdatasets/csv/carData/Davis.csv'
    data = pd.read_csv(url, index_col=0)
    feat1 = 'height' #'repwt'
    feat2 = 'weight' #'repht'
    k = 3
    n_iters = 10

    kmean_clustering(data, feat1, feat2, k, n_iters, csv_output, html_output, png_output)
