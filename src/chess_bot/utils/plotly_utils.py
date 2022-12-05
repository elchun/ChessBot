import numpy as np
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go

def multiplot(point_list: 'list[np.ndarray]', fname='debug.html'):
    """
    Plot each group of points in {point_list} in a different color on the same
    graph and saves to {fname}.
    Args:
        point_list (list[np.ndarray]): List of pointclouds in the form
            of (n_i x 3)
        fname (str, optional): Name of file to save result to.
            Defaults to 'debug.html'.
    Returns:
        plotly plot: Plot that is produced.
    """

    plot_pts = np.vstack(point_list)

    color = np.ones(plot_pts.shape[0])

    idx = 0
    for i, pts in enumerate(point_list):
        next_idx = idx + pts.shape[0]
        color[idx:next_idx] *= (i + 1)
        idx = next_idx


    fig = px.scatter_3d()

    fig.add_trace(go.Scatter3d(
        x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2],
        mode='markers',
        marker=dict(color=color, size=2)
    ))

    fig.update_layout(scene_aspectmode='data')
    # fig.update_traces(marker=dict(size=4))
    # fig.update_traces(marker=dict(size=5),
    #               selector=dict(mode='markers'))

    # fig.write_html(fname)
    # iplot(fig)

    return fig