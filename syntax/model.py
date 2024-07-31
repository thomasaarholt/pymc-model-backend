from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ModelContexts:
    models: list[Model] = field(default_factory=list)
    active_model: Model | None = None


model_contexts = ModelContexts()


def normal(mu: float, sigma: float):
    ...


class Dist:
    name: str = "dist"
    params: list = []

    def __init__(self):
        print("init")


class Model:
    distributions: list[Dist] = []

    def __init__(self, name: str):
        self.name: str = name
        model_contexts.models.append(self)

    def __enter__(self):
        model_contexts.active_model = self
        return self

    def __exit__(self, *args, **kwargs):
        model_contexts.active_model = None

    def __repr__(self):
        return self.name
