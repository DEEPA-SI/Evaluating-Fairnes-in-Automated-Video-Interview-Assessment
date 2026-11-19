"""Recursively immutable, JSON-compatible containers for self-checking evidence objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn


def _immutable(*_args: object, **_kwargs: object) -> NoReturn:
    raise TypeError("frozen JSON evidence cannot be mutated")


class FrozenDict(dict[str, Any]):
    """A concrete ``dict`` that rejects every ordinary mutation operation."""

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, _memo: dict[int, object]) -> FrozenDict:
        return self


class FrozenList(list[Any]):
    """A concrete ``list`` that rejects every ordinary mutation operation."""

    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable

    def __deepcopy__(self, _memo: dict[int, object]) -> FrozenList:
        return self


def freeze_json(value: Any) -> Any:
    """Deep-copy a JSON value into mutation-blocking concrete containers.

    Tuples are accepted for convenience and represented as ``FrozenList`` so the
    result remains directly serializable by :mod:`json` and compatible with
    validators that intentionally require concrete JSON ``dict``/``list`` types.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("frozen JSON object keys must be strings")
        return FrozenDict({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return FrozenList(freeze_json(item) for item in value)
    raise TypeError(f"unsupported frozen JSON value type: {type(value).__name__}")
