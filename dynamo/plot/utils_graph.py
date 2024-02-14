from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.patches import ConnectionPatch

from ..utils import areinstance


def create_edge_patch(
    posA,
    posB,
    node_rad=0,
    connectionstyle="arc3, rad=0.25",
    facecolor="k",
    head_length=10,
    head_width=10,
    tail_width=3,
    **kwargs,
):
    style = "simple,head_length=%d,head_width=%d,tail_width=%d" % (
        head_length,
        head_width,
        tail_width,
    )
    return pat.FancyArrowPatch(
        posA=posA,
        posB=posB,
        arrowstyle=style,
        connectionstyle=connectionstyle,
        facecolor=facecolor,
        shrinkA=node_rad,
        shrinkB=node_rad,
        **kwargs,
    )


def create_edge_patches_from_markov_chain(
    P,
    X,
    width=3,
    node_rad=0,
    tol=1e-7,
    connectionstyle="arc3, rad=0.25",
    facecolor="k",
    **kwargs,
):
    arrows = []
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            if P[j, i] > tol:
                arrows.append(
                    create_edge_patch(
                        X[i],
                        X[j],
                        tail_width=P[j, i] * width,
                        node_rad=node_rad,
                        connectionstyle=connectionstyle,
                        facecolor=facecolor,
                        alpha=min(2 * P[j, i], 1),
                        **kwargs,
                    )
                )
    return arrows


def plot_alternate_function(X, E, arrowstype="-|>", node_rad=5, arrow_size=10, fc="w"):
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if E[i, j] != 0:
                xa = X[i] if E[i, j] > 0 else X[j]
                xb = X[j] if E[i, j] > 0 else X[i]
                con = ConnectionPatch(
                    xa,
                    xb,
                    "data",
                    "data",
                    arrowstyle=arrowstype,
                    shrinkA=node_rad,
                    shrinkB=node_rad,
                    mutation_scale=arrow_size,
                    fc=fc,
                )
                plt.gca().add_artist(con)


def arcplot(
    x: np.ndarray,
    E: np.ndarray,
    node_names: Optional[List[str]] = None,
    edge_threshold: float = 0,
    curve_radius: float = 0.7,
    node_rad: float = 10,
    width: float = 1,
    arrow_head: float = 5,
    curve_alpha: float = 1,
    arrow_direction: int = -1,
    node_name_rotation: float = -45,
    hide_frame: bool = True,
    edge_mode: str = "alpha",
    edge_pos_color: str = "r",
    edge_neg_color: str = "b",
    edge_color: str = "k",
    **kwargs,
) -> None:
    """Plot a matrix form network as an arc plot.

    Args:
        x: The x-coordinates of the nodes.
        E: The adjacency matrix of the network.
        node_names: The names of the nodes.
        edge_threshold: The threshold to show the edge.
        curve_radius: The radius of the curve.
        node_rad: The radius of the node.
        width: The width of the edge.
        arrow_head: The size of the arrow head.
        curve_alpha: The alpha of the curve.
        arrow_direction: The direction of the arrow.
        node_name_rotation: The rotation of the node name on the plot.
        hide_frame: Whether to hide the frame.
        edge_mode: The mode of the edge.
        edge_pos_color: The color of the positive edge.
        edge_neg_color: The color of the negative edge.
        edge_color: The color of the edge.
        **kwargs: Additional arguments.
    """
    X = np.vstack((x, np.zeros(len(x)))).T
    plt.scatter(X[:, 0], X[:, 1], **kwargs)

    # calculate alpha
    E_abs = np.abs(E)
    alp_scale = np.max(E_abs - np.diag(np.diag(E_abs)))

    # arrow style
    curve_radius = np.abs(curve_radius) * arrow_direction
    cstyle = "arc3, rad=%f" % curve_radius

    if np.min(E) >= 0:
        edge_pos_color = edge_neg_color = edge_color

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if i != j and E_abs[i, j] > edge_threshold:
                posA, posB = X[i], X[j]
                if edge_mode == "alpha":
                    alpha = E_abs[i, j] / alp_scale * curve_alpha
                    tail_width = width
                elif edge_mode == "width":
                    alpha = curve_alpha
                    tail_width = E_abs[i, j] / alp_scale * width * 5
                elif edge_mode == "const":
                    alpha = curve_alpha
                    tail_width = width
                else:
                    raise NotImplementedError("Unidentified edge mode. Options are `alpha`, `width`, and `const`.")

                ec = edge_pos_color if E[i, j] > 0 else edge_neg_color

                head_width = tail_width * arrow_head
                arrow = create_edge_patch(
                    posA,
                    posB,
                    node_rad=node_rad,
                    head_width=head_width,
                    tail_width=tail_width,
                    connectionstyle=cstyle,
                    alpha=alpha,
                    facecolor=ec,
                    edgecolor=ec,
                )
                plt.gca().add_patch(arrow)

    plt.gca().get_yaxis().set_ticks([])

    if node_names is not None:
        plt.xticks(x, node_names, rotation=node_name_rotation)

    if hide_frame:
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)


class ArcPlot:
    """The class to plot a matrix form network as an arc plot."""
    def __init__(
        self,
        x: Optional[np.ndarray] = None,
        E: Optional[np.ndarray] = None,
        network: Optional[nx.classes.DiGraph] = None,
        node_names: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """Initialize the ArcPlot class."""
        self.x = x if x is not None else None
        self.E = E if E is not None else None
        self.node_names = node_names if node_names is not None else None
        if network is not None:
            self.set_network(network)
        self.kwargs = kwargs

    def set_network(self, network: nx.classes.DiGraph) -> None:
        """Set the network information to the plot."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                f"You need to install the packages `networkx`." f"install networkx via `pip install networkx`."
            )
        self.E = nx.to_numpy_array(network)
        self.node_names = list(network.nodes)

    def compute_node_positions(self, node_order: Optional[Union[str, np.ndarray]] = None) -> None:
        """Compute the node positions."""
        if self.E is None:
            raise Exception("The adjacency matrix is not set.")

        if node_order is None:
            self.x = np.arange(self.E.shape[0])
        else:
            if areinstance(node_order, str):
                self.x = np.zeros(self.E.shape[0], dtype=int)
                for i, n in enumerate(node_order):
                    self.x[np.where(np.array(self.node_names) == n)[0][0]] = i
            else:
                self.x = np.argsort(node_order)

    def draw(self, node_order: Optional[Union[str, np.ndarray]] = None) -> None:
        """Draw the arc plot."""
        if self.x is None:
            self.compute_node_positions(node_order=node_order)
        arcplot(self.x, self.E, node_names=self.node_names, **self.kwargs)
