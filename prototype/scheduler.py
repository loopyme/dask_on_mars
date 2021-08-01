from typing import List, Tuple

from dask.core import istask, ishashable
from mars.remote import spawn

from prototype.utils import reduce


def mars_scheduler(dsk: dict, keys: List[List]):
    """A Dask-Mars scheduler prototype. This scheduler is just for user interface compatibility,
    no callbacks are implemented"""

    return [[reduce(mars_dask_get(dsk, keys)).execute().fetch()]]


def mars_dask_get(dsk: dict, keys: List[List]):
    """A Dask-Mars get prototype, which only supports single-key computation"""

    def _execute_layer(layer: tuple):
        def _get_arg(a):
            # if arg contains layer index or callable objs, handle it
            if ishashable(a) and a in dsk.keys():
                while ishashable(a) and a in dsk.keys():
                    a = dsk[a]
                return _execute_layer(a)
            elif not isinstance(a, str) and hasattr(a, "__getitem__"):
                if istask(a):  # TODO:Handle `SubgraphCallable`, which may contains dsk in it
                    return spawn(a[0], args=tuple(_get_arg(i) for i in a[1:]))
                elif isinstance(a, dict):
                    return {k: _get_arg(v) for k, v in a.items()}
                elif isinstance(a, List) or isinstance(a, Tuple):
                    return type(a)(_get_arg(i) for i in a)
            return a

        if not istask(layer):
            return _get_arg(layer)
        return spawn(layer[0], args=tuple(_get_arg(a) for a in layer[1:]))

    return [[_execute_layer(dsk[k]) for k in keys_d] for keys_d in keys]
