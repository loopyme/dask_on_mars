import pytest
from dask import delayed
from numpy.core.numeric import array_equal

from prototype.converter import convert_dask_collection
from prototype.scheduler import mars_scheduler


def test_delayed():
    from typing import List
    import numpy as np

    def calc_chunk(n: int, i: int):
        # 计算n个随机点（x和y轴落在-1到1之间）到原点距离小于1的点的个数
        rs = np.random.RandomState(i)
        a = rs.uniform(-1, 1, size=(n, 2))
        d = np.linalg.norm(a, axis=1)
        return (d < 1).sum()

    def calc_pi(fs: List[int], N: int):
        # 将若干次 calc_chunk 计算的结果汇总，计算 pi 的值
        return sum(fs) * 4 / N

    N = 200_000  # _000
    n = 10_000  # _000

    fs = [delayed(calc_chunk)(n, i) for i in range(N // n)]
    pi = delayed(calc_pi)(fs, N)

    dask_res = pi.compute()
    assert dask_res == pi.compute(scheduler=mars_scheduler)
    assert dask_res == convert_dask_collection(pi).execute().fetch()


def test_partitioned_dataframe():
    import numpy as np
    import pandas as pd
    from dask import dataframe as dd
    from pandas._testing import assert_frame_equal

    data = np.random.randn(10000, 100)
    df = dd.from_pandas(
        pd.DataFrame(data, columns=[f"col{i}" for i in range(100)]), npartitions=4
    )
    df["col0"] = df["col0"] + df["col1"] / 2
    col2_mean = df["col2"].mean()
    df = df[df["col2"] > col2_mean]

    dask_res = df.compute()
    assert_frame_equal(dask_res, df.compute(scheduler=mars_scheduler))
    assert_frame_equal(dask_res, convert_dask_collection(df).execute().fetch())


def test_unpartitioned_dataframe():
    from dask import dataframe as dd
    from pandas._testing import assert_frame_equal

    df = dd.read_csv(r"D:\summer2021\daskMars\playground\boston_housing_data.csv")
    df["CRIM"] = df["CRIM"] / 2

    dask_res = df.compute()
    assert_frame_equal(dask_res, df.compute(scheduler=mars_scheduler))
    assert_frame_equal(dask_res, convert_dask_collection(df).execute().fetch())


def test_array():
    import dask.array as da

    x = da.random.random((10000, 10000), chunks=(1000, 1000))
    y = x + x.T
    z = y[::2, 5000:].mean(axis=1)

    dask_res = z.compute()
    print(dask_res)
    assert array_equal(dask_res, z.compute(scheduler=mars_scheduler))
    assert array_equal(dask_res, convert_dask_collection(z).execute().fetch())


def test_bag():
    import dask

    b = dask.datasets.make_people()  # Make records of people
    result = (
        b.filter(lambda record: record["age"] > 30)
            .map(lambda record: record["occupation"])
            .frequencies(sort=True)
            .topk(10, key=1)
    )

    dask_res = result.compute()
    assert dask_res == result.compute(scheduler=mars_scheduler)
    assert dask_res == list(
        convert_dask_collection(result).execute().fetch()
    )  # TODO: dask-bag computation will return weired tuple, which we don't know why


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
