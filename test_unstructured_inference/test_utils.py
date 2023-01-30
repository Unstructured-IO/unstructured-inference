import pytest
from unstructured_inference.utils import LazyDict, LazyEvaluateInfo


def test_dict_same():
    d = {"a": 1, "b": 2, "c": 3}
    ld = LazyDict(**d)
    assert all(kd == kld for kd, kld in zip(d, ld))
    assert all(d[k] == ld[k] for k in d)
    assert len(ld) == len(d)


def test_lazy_evaluate():
    called = 0

    def func(x):
        nonlocal called
        called += 1
        return x

    lei = LazyEvaluateInfo(func, 3)
    assert called == 0
    ld = LazyDict(a=lei)
    assert called == 0
    assert ld["a"] == 3
    assert called == 1


@pytest.mark.parametrize("cache, expected", [(True, 1), (False, 2)])
def test_caches(cache, expected):
    called = 0

    def func(x):
        nonlocal called
        called += 1
        return x

    lei = LazyEvaluateInfo(func, 3)
    assert called == 0
    ld = LazyDict(cache=cache, a=lei)
    assert called == 0
    assert ld["a"] == 3
    assert ld["a"] == 3
    assert called == expected
