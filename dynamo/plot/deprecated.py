import warnings
from typing import Any, Dict, List, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import functools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from anndata import AnnData
from .scatters import scatters
from .networks import nxvizPlot


def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and will be removed in a future release. "
            f"Please update your code to use the new replacement function.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


@deprecated
def infomap(*args, **kwargs):
    return _infomap_legacy(*args, **kwargs)


def _infomap_legacy(adata: AnnData, basis: str = "umap", color: str = "infomap", *args, **kwargs) -> Optional[plt.Axes]:
    """Scatter plot for infomap community detection in selected basis.

    Args:
        adata: an Annodata object.
        basis: the reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following
            priority if exits: 1) specific layer input + basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters`
            is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`.
            Defaults to "umap".
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "infomap".

    Returns:
        None would be returned in default and the plotted figure would be shown directly. If set
        `save_show_or_return='return'` as a kwarg, the axes of the plot would be returned.
    """

    return scatters(adata, basis=basis, color=color, *args, **kwargs)


@deprecated
def dynamics_(*args, **kwargs):
    return _dynamics_legacy(*args, **kwargs)


def _dynamics_legacy(
    adata,
    gene_names,
    color,
    dims=[0, 1],
    current_layer="spliced",
    use_raw=False,
    Vkey="S",
    Ekey="spliced",
    basis="umap",
    mode="all",
    cmap=None,
    gs=None,
    **kwargs,
):
    """

    Parameters
    ----------
    adata
    basis
    mode: `str` (default: all)
        Support mode includes: phase, expression, velocity, all

    Returns
    -------

    """

    import matplotlib.pyplot as plt

    genes = list(set(gene_names).intersection(adata.var.index))
    for i, gn in enumerate(genes):
        ax = plt.subplot(gs[i * 3])
        try:
            ix = np.where(adata.var["Gene"] == gn)[0][0]
        except:
            continue

        scatters(
            adata,
            gene_names,
            color,
            dims=[0, 1],
            current_layer="spliced",
            use_raw=False,
            Vkey="S",
            Ekey="spliced",
            basis="umap",
            mode="all",
            cmap=None,
            gs=None,
            **kwargs,
        )

        scatters(
            adata,
            gene_names,
            color,
            dims=[0, 1],
            current_layer="spliced",
            use_raw=False,
            Vkey="S",
            Ekey="spliced",
            basis="umap",
            mode="all",
            cmap=None,
            gs=None,
            **kwargs,
        )

        scatters(
            adata,
            gene_names,
            color,
            dims=[0, 1],
            current_layer="spliced",
            use_raw=False,
            Vkey="S",
            Ekey="spliced",
            basis="umap",
            mode="all",
            cmap=None,
            gs=None,
            **kwargs,
        )

    plt.tight_layout()


@deprecated
def circosPlotDeprecated(*args, **kwargs):
    return _circosPlot_legacy(*args, **kwargs)


def _circosPlot_legacy(
    adata: AnnData,
    cluster: str,
    cluster_name: str,
    edges_list: Dict[str, pd.DataFrame],
    network: nx.classes.digraph.DiGraph = None,
    weight_scale: float = 5e3,
    weight_threshold: float = 1e-4,
    figsize: Tuple[float, float] = (12, 6),
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> Optional[Any]:

    """Deprecated.

    A wrapper of `dynamo.pl.networks.nxvizPlot` to plot Circos graph. See the `nxvizPlot` for more information.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the generated `nxviz` plot
        object would be returned.
    """

    nxvizPlot(
        adata,
        cluster,
        cluster_name,
        edges_list,
        plot="circosplot",
        network=network,
        weight_scale=weight_scale,
        weight_threshold=weight_threshold,
        figsize=figsize,
        save_show_or_return=save_show_or_return,
        save_kwargs=save_kwargs,
        **kwargs,
    )
