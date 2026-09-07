"""Recording stubs: any Isaac Lab name upstream instantiates at import time."""

from __future__ import annotations

import types


class StubMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (Stub,), {})


class Stub(metaclass=StubMeta):
    """``Stub(a=1, b=2).a == 1``; unknown attributes are fresh stubs; callable; copyable."""

    def __init__(self, *args, **kw):
        self.args = args
        self.__dict__.update(kw)

    def __call__(self, *a, **k):
        # used as a decorator (@configclass-like) -> identity; otherwise a new record
        if len(a) == 1 and not k and callable(a[0]):
            return a[0]
        return Stub(*a, **k)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Stub()

    def copy(self):
        return self

    def __getitem__(self, k):
        return Stub()

    def __repr__(self):
        return f"Stub({ {k: v for k, v in self.__dict__.items() if k != 'args'} })"


def configclass(cls=None, **kw):
    """Upstream decorates its cfg classes with isaaclab's ``configclass``; a plain class does."""
    return cls if cls is not None else (lambda c: c)


class StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "configclass":
            return configclass
        return type(name, (Stub,), {})
