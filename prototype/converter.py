from dask import is_dask_collection, optimize

from prototype.scheduler import mars_dask_get
from prototype.utils import reduce


def convert_dask_collection(dc):
    """Convert dask collections into mars.core.Object"""
    assert is_dask_collection(dc)
    dc.__dask_graph__().validate()

    dsk = optimize(dc)[0].__dask_graph__()

    first_key = next(iter(dsk.keys()))
    if isinstance(first_key, str):
        key = [first_key]
    elif isinstance(first_key, tuple):
        key = sorted([i for i in dsk.keys() if i[0] == first_key[0]], key=lambda x: x[1])
    else:
        raise Exception(f"Weired key type:{type(first_key)}")

    return reduce(mars_dask_get(dsk, [key]))
