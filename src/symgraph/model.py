from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ModelContexts:
    contexts: set[Model] = field(default_factory=set)
    active_contexts: list[Model] = field(default_factory=list)

    @property
    def current_context(self) -> Model | None:
        """Return the innermost context of any current contexts."""
        return self.active_contexts[-1] if self.active_contexts else None


model_contexts = ModelContexts()


class Param: ...


def normal(mu: float, sigma: float): ...


class Dist:
    name: str = "dist"
    params: list[Param] = []

    def __init__(self):
        print("init")


class Model:
    parent: Model | None = None
    distributions: list[Dist] = []

    def __init__(self, name: str, model: Model | None = None):
        self.name: str = self._validate_name(name)
        model_contexts.contexts.add(self)
        if model:
            self.parent = model
        else:
            self.parent = (
                model_contexts.active_contexts[-1]
                if model_contexts.active_contexts
                else None
            )

    def __enter__(self):
        model_contexts.active_contexts.append(self)
        return self

    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        _ = model_contexts.active_contexts.pop()

    def __repr__(self):
        return self.name

    @staticmethod
    def _validate_name(name: str) -> str:
        if name.endswith(":"):
            raise KeyError("name should not end with `:`")
        return name
