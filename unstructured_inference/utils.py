from collections.abc import Mapping
from typing import Any, Hashable, Union, Iterator, Callable


class LazyEvaluateInfo:
    """Class that stores the information needed to lazily evaluate a function with given arguments.
    The object stores the information needed for evaluation as a function and its arguments.
    """

    def __init__(self, evaluate: Callable, *args, **kwargs):
        self.evaluate = evaluate
        self.info = (args, kwargs)


class LazyDict(Mapping):
    """Class that wraps a dict and only evaluates keys of the dict when the key is accessed. Keys
    that should be evaluated lazily should use LazyEvaluateInfo objects as values. By default when
    a value is computed from a LazyEvaluateInfo object, it is converted to the raw value in the
    internal dict, so subsequent accessing of the key will produce the same value. Set cache=False
    to avoid storing the raw value.
    """

    def __init__(self, *args, cache=True, **kwargs):
        self.cache = cache
        self._raw_dict = dict(*args, **kwargs)

    def __getitem__(self, key: Hashable) -> Union[LazyEvaluateInfo, Any]:
        value = self._raw_dict.__getitem__(key)
        if isinstance(value, LazyEvaluateInfo):
            evaluate = value.evaluate
            args, kwargs = value.info
            value = evaluate(*args, **kwargs)
            if self.cache:
                self._raw_dict[key] = value
        return value

    def __iter__(self) -> Iterator:
        return iter(self._raw_dict)

    def __len__(self) -> int:
        return len(self._raw_dict)
