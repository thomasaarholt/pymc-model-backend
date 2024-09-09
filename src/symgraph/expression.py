# expression.py
from __future__ import annotations
from collections.abc import Mapping
from typing import Any, override

import numpy as np

from numpy.typing import ArrayLike, NDArray

from symgraph.utils import COLORS, RESET_COLOR


class Node:
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        raise NotImplementedError

    def _colorize(self, text: str, level: int) -> str:
        """Apply color based on the node's level in the tree."""
        color = COLORS[level % len(COLORS)]  # Cycle through the 6 colors
        return f"{color}{text}{RESET_COLOR}"

    def __str__(self) -> str:
        return self._str_with_indent(0)

    def _str_with_indent(self, level: int) -> str:
        raise NotImplementedError(f"Not implemented for {type(self)}")

    @override
    def __eq__(self, other: object) -> bool:
        """Compare the node with another Constant or numeric."""
        if isinstance(self, Constant):
            if isinstance(other, int):
                return self.value == other
            elif isinstance(other, Constant):
                return self.value == other.value
        return id(self) == id(other)

    # Addition
    def __add__(self, other: Node | float) -> Node:
        if isinstance(other, (int, float)):
            other = Constant(value=other)
        return Add(left=self, right=other)

    def __radd__(self, other: float) -> Node:
        return Constant(value=other) + self

    # Subtraction
    def __sub__(self, other: Node | float) -> Node:
        if isinstance(other, (int, float)):
            other = Constant(value=other)
        return Subtract(left=self, right=other)

    def __rsub__(self, other: float) -> Node:
        return Constant(value=other) - self

    # Multiplication
    def __mul__(self, other: Node | float) -> Node:
        if isinstance(other, (int, float)):
            other = Constant(value=other)
        return Multiply(left=self, right=other)

    def __rmul__(self, other: float) -> Node:
        return Constant(value=other) * self

    # Division
    def __truediv__(self, other: Node | float) -> Node:
        if isinstance(other, (int, float)):
            other = Constant(value=other)
        return Divide(left=self, right=other)

    def __rtruediv__(self, other: float) -> Node:
        return Constant(value=other) / self

    # Exponentiation
    def __pow__(self, other: Node | float) -> Node:
        if isinstance(other, (int, float)):
            other = Constant(value=other)
        return Exponentiation(left=self, right=other)

    def __rpow__(self, other: float) -> Node:
        return Constant(value=other) ** self

    def __neg__(self) -> Node:
        return -1 * self


class Constant(Node):
    def __init__(self, value: float):
        self.value = value

    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return np.array(self.value)

    @override
    def _str_with_indent(self, level: int) -> str:
        return self._colorize(f"{'    ' * level}{self.value}", level)


class Symbol(Node):
    def __init__(self, name: str):
        self.name = name

    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        if self.name not in values:
            raise ValueError(f"Value for symbol {self.name} not provided")
        return np.array(values[self.name])

    @override
    def _str_with_indent(self, level: int) -> str:
        return self._colorize(f"{'    ' * level}{self.name}", level)


class Operation(Node):
    operator: str

    @property
    def operands(self) -> list[Node]:
        raise NotImplementedError(f"Not implemented for {type(self)}")

    @operands.setter
    def operands(self, new_operands: list[Node]) -> None:
        raise NotImplementedError(f"Not implemented for {type(self)}")


class BinaryOperation(Operation):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

    @property
    @override
    def operands(self) -> list[Node]:
        return [self.left, self.right]

    @operands.setter
    def operands(self, new_operands: list[Node]) -> None:
        self.left, self.right = new_operands

    @override
    def _str_with_indent(self, level: int) -> str:
        # Print the operator and recursively print left and right with indentation
        result = self._colorize(f"{'    ' * level}{self.operator}\n", level)
        result += self.left._str_with_indent(level + 1) + "\n"
        result += self.right._str_with_indent(level + 1)
        return result


class UnaryOperation(Operation):
    def __init__(self, operand: Node):
        self.operand = operand

    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        raise NotImplementedError

    def diff(self, var: "Symbol") -> "Node":
        raise NotImplementedError

    @property
    @override
    def operands(self) -> list[Node]:
        return [self.operand]

    @operands.setter
    def operands(self, new_operands: list[Node]):
        self.operand = new_operands[0]

    @override
    def _str_with_indent(self, level: int) -> str:
        result = self._colorize(f"{'    ' * level}{self.operator}\n", level)
        result += self.operand._str_with_indent(level + 1) + "\n"
        return result


class Add(BinaryOperation):
    operator: str = "Add"

    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return self.left.evaluate(values) + self.right.evaluate(values)


class Subtract(BinaryOperation):
    operator: str = "Subtract"

    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return self.left.evaluate(values) - self.right.evaluate(values)


class Multiply(BinaryOperation):
    operator: str = "Multiply"

    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return self.left.evaluate(values) * self.right.evaluate(values)


class Divide(BinaryOperation):
    operator: str = "Divide"

    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return self.left.evaluate(values) / self.right.evaluate(values)


class Exponentiation(BinaryOperation):
    operator: str = "Exponentiation"

    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return self.left.evaluate(values) ** self.right.evaluate(values)


class Function(Operation):
    arguments: list[Node]

    @override
    def _str_with_indent(self, level: int) -> str:
        result = self._colorize(f"{'    ' * level}{self.operator}\n", level)
        for arg in self.arguments:
            result += arg._str_with_indent(level + 1) + "\n"
        return result


class Sqrt(UnaryOperation):
    operator: str = "Sqrt"

    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return np.sqrt(self.operand.evaluate(values))


class Exp(UnaryOperation):
    operator: str = "Exp"

    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return np.exp(self.operand.evaluate(values))


class Ln(UnaryOperation):
    operator: str = "Ln"

    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return np.log(self.operand.evaluate(values))
