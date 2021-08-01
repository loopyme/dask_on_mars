from typing import List

from dask import is_dask_collection
from dask.array.core import _concatenate2 as a_concat
from dask.dataframe import concat as d_concat
from dask.utils import is_arraylike, is_dataframe_like, is_series_like, is_index_like
from mars.remote import spawn


def concat_wrapper(objs, *args, **kwargs):
    if is_arraylike(objs[0]):
        res = a_concat(objs, axes=[0])
    elif any((is_dataframe_like(objs[0]), is_series_like(objs[0]), is_index_like(objs[0]))):
        res = d_concat(objs, *args, **kwargs)
    else:
        res = objs
        while isinstance(res, List):
            res = res[0]
    return res.compute() if is_dask_collection(res) else res


def reduce(objs: List):
    return spawn(
        concat_wrapper,
        args=([spawn(concat_wrapper, args=(objs_d,)) for objs_d in objs],),
    )
