# expression.py
from __future__ import annotations
from collections.abc import Mapping
from typing import Any, override

import numpy as np

from numpy.typing import ArrayLike, NDArray

from symgraph.utils import COLORS, RESET_COLOR


class Node:
    """Base class for expression tree nodes in a computation graph."""

    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        """Evaluates the node with given variable values.

        Args:
            values: Mapping of variable names to their values.

        Returns:
            Computed result as a numpy array.
        """
        raise NotImplementedError

    def to_latex(self) -> str:
        """Converts the node to LaTeX representation.

        Returns:
            LaTeX string representation of the node.
        """
        raise NotImplementedError("to_latex not implemented for this node type")

    def _colorize(self, text: str, level: int) -> str:
        """Apply color based on the node's level in the tree."""
        color = COLORS[level % len(COLORS)]  # Cycle through the 6 colors
        return f"{color}{text}{RESET_COLOR}"

    def __str__(self) -> str:
        """Returns string representation of the node."""
        return self._str_with_indent(0)

    def _str_with_indent(self, level: int) -> str:
        """Creates indented string representation for pretty printing.

        Args:
            level: Current indentation level.
        """
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

    def __add__(self, other: Node | float) -> Node:
        """Addition operator implementation."""
        if isinstance(other, (int, float)):
            other = Constant(value=other)
        return Add(left=self, right=other)

    def __radd__(self, other: float) -> Node:
        """Reverse addition operator implementation."""
        return Constant(value=other) + self

    def __sub__(self, other: Node | float) -> Node:
        """Subtraction operator implementation."""
        if isinstance(other, (int, float)):
            other = Constant(value=other)
        return Subtract(left=self, right=other)

    def __rsub__(self, other: float) -> Node:
        """Reverse subtraction operator implementation."""
        return Constant(value=other) - self

    def __mul__(self, other: Node | float) -> Node:
        """Multiplication operator implementation."""
        if isinstance(other, (int, float)):
            other = Constant(value=other)
        return Multiply(left=self, right=other)

    def __rmul__(self, other: float) -> Node:
        """Reverse multiplication operator implementation."""
        return Constant(value=other) * self

    def __truediv__(self, other: Node | float) -> Node:
        """Division operator implementation."""
        if isinstance(other, (int, float)):
            other = Constant(value=other)
        return Divide(left=self, right=other)

    def __rtruediv__(self, other: float) -> Node:
        """Reverse division operator implementation."""
        return Constant(value=other) / self

    def __pow__(self, other: Node | float) -> Node:
        """Exponentiation operator implementation."""
        if isinstance(other, (int, float)):
            other = Constant(value=other)
        return Exponentiation(left=self, right=other)

    def __rpow__(self, other: float) -> Node:
        """Reverse exponentiation operator implementation."""
        return Constant(value=other) ** self

    def __neg__(self) -> Node:
        """Unary negation operator implementation."""
        return -1 * self


class Constant(Node):
    def __init__(self, value: float):
        self.value = value

    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return np.array(self.value)

    @override
    def to_latex(self) -> str:
        return str(self.value)

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
    def to_latex(self) -> str:
        return self.name

    @override
    def _str_with_indent(self, level: int) -> str:
        return self._colorize(f"{'    ' * level}{self.name}", level)

    @override
    def __eq__(self, other: object) -> bool:
        """Compare the node with another Constant or numeric."""
        if isinstance(other, Symbol):
            return self.name == other.name
        return False


class Operation(Node):
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
        result = self._colorize(f"{'    ' * level}{type(self).__name__}\n", level)
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
        result = self._colorize(f"{'    ' * level}{type(self).__name__}\n", level)
        result += self.operand._str_with_indent(level + 1) + "\n"
        return result


class Add(BinaryOperation):
    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return self.left.evaluate(values) + self.right.evaluate(values)

    @override
    def to_latex(self) -> str:
        return f"{self.left.to_latex()} + {self.right.to_latex()}"


class Subtract(BinaryOperation):
    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return self.left.evaluate(values) - self.right.evaluate(values)

    @override
    def to_latex(self) -> str:
        return f"{self.left.to_latex()} - {self.right.to_latex()}"


class Multiply(BinaryOperation):
    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return self.left.evaluate(values) * self.right.evaluate(values)

    @override
    def to_latex(self) -> str:
        return f"{self.left.to_latex()} \\cdot {self.right.to_latex()}"


class Divide(BinaryOperation):
    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return self.left.evaluate(values) / self.right.evaluate(values)

    @override
    def to_latex(self) -> str:
        return f"\\frac{{{self.left.to_latex()}}}{{{self.right.to_latex()}}}"


class Exponentiation(BinaryOperation):
    @property
    def base(self):
        return self.left

    @property
    def exponent(self):
        return self.right

    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return self.base.evaluate(values) ** self.exponent.evaluate(values)

    @override
    def to_latex(self) -> str:
        return f"{self.base.to_latex()}^{{{self.exponent.to_latex()}}}"


class Sqrt(UnaryOperation):
    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return np.sqrt(self.operand.evaluate(values))

    @override
    def to_latex(self) -> str:
        return f"\\sqrt{{{self.operand.to_latex()}}}"


class Exp(UnaryOperation):
    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return np.exp(self.operand.evaluate(values))

    @override
    def to_latex(self) -> str:
        return f"e^{{{self.operand.to_latex()}}}"


class Ln(UnaryOperation):
    @override
    def evaluate(self, values: Mapping[str, ArrayLike]) -> NDArray[Any]:
        return np.log(self.operand.evaluate(values))

    @override
    def to_latex(self) -> str:
        return f"\\ln({self.operand.to_latex()})"
