from collections.abc import Mapping
from typing import Any, Hashable, Union, Iterator, Callable


class LazyEvaluateInfo:
    def __init__(self, evaluate: Callable, *args, **kwargs):
        self.evaluate = evaluate
        self.info = (args, kwargs)


class LazyDict(Mapping):
    """Class that wraps a dict and only evaluates keys of the dict when the key is accessed. Keys
    that should be evaluated lazily should use LazyEvaluateInfo objects as values."""

    def __init__(self, *args, **kwargs):
        self._raw_dict = dict(*args, **kwargs)

    def __getitem__(self, key: Hashable) -> Union[LazyEvaluateInfo, Any]:
        value = self._raw_dict.__getitem__(key)
        if isinstance(value, LazyEvaluateInfo):
            evaluate = value.evaluate
            args, kwargs = value.info
            return evaluate(*args, **kwargs)
        else:
            return value

    def __iter__(self) -> Iterator:
        return iter(self._raw_dict)

    def __len__(self) -> int:
        return len(self._raw_dict)
