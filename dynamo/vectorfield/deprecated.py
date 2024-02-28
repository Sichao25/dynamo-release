import warnings
from typing import Callable, Optional, Tuple

import functools
import igraph as ig
import numpy as np
from scipy.optimize import OptimizeResult, minimize


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
def action(*args, **kwargs):
    return _action_legacy(*args, **kwargs)


def _action_legacy(path: np.ndarray, vf_func: Callable, D=1, dt=1) -> float:
    """Compute the action of a path by taking the sum of the squared distance between the path and the vector field and dividing by twice the diffusion constant. Conceptually, the action represents deviations from the streamline of the vector field. Reference Box 3 in the publication for more information.

    Args:
        path: sequence of points in the state space collectively representing the path of interest
        vf_func: function that takes a point in the state space and returns the vector field at that point
        D: Diffusion constant, Defaults to 1.
        dt: Time step for moving from one state to another within the path, Defaults to 1.

    Returns:
        the action of the path
    """
    # centers
    x = (path[:-1] + path[1:]) * 0.5
    v = np.diff(path, axis=0) / dt

    s = (v - vf_func(x)).flatten()
    s = 0.5 * s.dot(s) * dt / D

    return s


@deprecated
def action_aux(*args, **kwargs):
    return _action_aux_legacy(*args, **kwargs)


def _action_aux_legacy(path_flatten, vf_func, dim, start=None, end=None, **kwargs):
    path = _reshape_path_legacy(path_flatten, dim, start=start, end=end)
    return _action_legacy(path, vf_func, **kwargs)


@deprecated
def action_grad(*args, **kwargs):
    return _action_grad_legacy(*args, **kwargs)


def _action_grad_legacy(path: np.ndarray, vf_func: Callable, jac_func: Callable, D: float = 1, dt: float = 1) -> np.ndarray:
    """Compute the gradient of the action with respect to each component of each point in the path using the analytical Jacobian.

    Args:
        path: sequence of points in the state space collectively representing the path of interest
        vf_func: function that takes a point in the state space and returns the vector field at that point
        jac_func: function for computing Jacobian given cell state
        D: Diffusion constant, Defaults to 1.
        dt: Time step for moving from one state to another within the path, Defaults to 1.

    Returns:
        gradient of the action with respect to each component of each point in the path
    """
    x = (path[:-1] + path[1:]) * 0.5
    v = np.diff(path, axis=0) / dt

    dv = v - vf_func(x)
    J = jac_func(x)
    z = np.zeros(dv.shape)
    for s in range(dv.shape[0]):
        z[s] = dv[s] @ J[:, :, s]
    grad = (dv[:-1] - dv[1:]) / D - dt / (2 * D) * (z[:-1] + z[1:])
    return grad


@deprecated
def action_grad_aux(*args, **kwargs):
    return _action_grad_aux_legacy(*args, **kwargs)


def _action_grad_aux_legacy(path_flatten, vf_func, jac_func, dim, start=None, end=None, **kwargs):
    path = _reshape_path_legacy(path_flatten, dim, start=start, end=end)
    return _action_grad_legacy(path, vf_func, jac_func, **kwargs).flatten()


@deprecated
def reshape_path(*args, **kwargs):
    return _reshape_path_legacy(*args, **kwargs)


def _reshape_path_legacy(path_flatten, dim, start=None, end=None):
    path = path_flatten.reshape(int(len(path_flatten) / dim), dim)
    if start is not None:
        path = np.vstack((start, path))
    if end is not None:
        path = np.vstack((path, end))
    return path


@deprecated
def least_action_path(*args, **kwargs):
    return _least_action_path_legacy(*args, **kwargs)


def _least_action_path_legacy(
    start: np.ndarray,
    end: np.ndarray,
    vf_func: Callable,
    jac_func: Callable,
    n_points: int = 20,
    init_path: Optional[np.ndarray] = None,
    D: int = 1,
) -> Tuple[np.ndarray, OptimizeResult]:
    """Compute the least action path between two points using gradient descent optimization.

    Args:
        start: The starting point for the path
        end: The ending point for the path
        vf_func: A function that returns the vector field of the system at a given point.
        jac_func: A function that returns the Jacobian at a given point.
        n_points: The number of intermediate points to use when initializing the path. Defaults to 20.
        init_path: An optional initial path to use instead of the default initialization. Defaults to None.
        D: The diffusion constant. Defaults to 1.

    Returns:
        A tuple containing the optimized least action path and the optimization result. The least action path is a numpy array of shape (n_points + 2, D), where n_points is the number of intermediate points used in the initialization and D is the dimension of start and end. The optimization result is a scipy.optimize.OptimizeResult object containing information about the optimization process.
    """

    dim = len(start)
    if init_path is None:
        path_0 = (
            np.tile(start, (n_points + 1, 1))
            + (np.linspace(0, 1, n_points + 1, endpoint=True) * np.tile(end - start, (n_points + 1, 1)).T).T
        )
    else:
        path_0 = init_path
    fun = lambda x: _action_aux_legacy(x, vf_func, dim, start=path_0[0], end=path_0[-1], D=D)
    jac = lambda x: _action_grad_aux_legacy(x, vf_func, jac_func, dim, start=path_0[0], end=path_0[-1], D=D)
    sol_dict = minimize(fun, path_0[1:-1], jac=jac)
    path_sol = _reshape_path_legacy(sol_dict["x"], dim, start=path_0[0], end=path_0[-1])

    return path_sol, sol_dict


class ConverterMixin(object):
    """Answer by FastTurtle https://stackoverflow.com/questions/18020074/convert-a-baseclass-object-into-a-subclass-object-idiomatically"""

    @classmethod
    def convert_to_class(cls, obj):
        obj.__class__ = cls


class vfGraph(ConverterMixin, ig.Graph):
    """A class for manipulating the graph creating from the transition matrix, built from the (reconstructed) vector
    field. This is a derived class from igraph's Graph.
    """

    def __init__(self, *args, **kwds):
        super(vfGraph, self).__init__(*args, **kwds)

    def build_graph(self, adj_mat):
        """build sparse diffusion graph. The adjacency matrix needs to preserves divergence."""

        sources, targets = adj_mat.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        self.__init__(
            edgelist,
            edge_attrs={"weight": adj_mat.data.tolist()},
            directed=True,
        )

    def multimaxflow(self, sources, sinks):
        """Multi-source multi-sink maximum flow. Ported from https://github.com/kazumits/ddhodge/blob/master/R/graphConstr.R"""

        v_num, e_num = self.vcount(), self.ecount()
        usrc = v_num  # super-source
        usink = usrc + 1  # super-sink
        new_edges = np.hstack(
            [
                np.vstack(([usrc] * len(sources), sources)),
                np.vstack((sinks, [usink] * len(sinks))),
            ]
        ).T
        self.add_vertices(2)
        self.add_edges(new_edges)
        w_sum = sum(self.es.get_attribute_values("weight")[:e_num])
        self.es["weight"] = [i if i is not None else w_sum for i in self.es["weight"]]

        mf = self.maxflow(usrc, usink, self.es.get_attribute_values("weight"))
        self.es.set_attribute_values("flow", mf.flow)
        self.vs.set_attribute_values("pass", self.strength(mode="in", weights=mf.flow))
        self.delete_vertices([usrc, usink])

    # add gradop, ...
