from __future__ import annotations
from dataclasses import dataclass


class Node:
    def __str__(self) -> str:
        raise NotImplementedError


@dataclass
class Constant(Node):
    value: float

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class Symbol(Node):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class Operation(Node):
    operator: str
    operands: list[Node]

    def __str__(self) -> str:
        operand_str = f" {self.operator} ".join(map(str, self.operands))
        return f"({operand_str})"


class BinaryOperation(Operation):
    @property
    def left(self) -> Node:
        return self.operands[0]

    @property
    def right(self) -> Node:
        return self.operands[-1]


class Add(BinaryOperation):
    def __init__(self, left: Node, right: Node) -> None:
        super().__init__("+", [left, right])


class Subtract(BinaryOperation):
    def __init__(self, left: Node, right: Node) -> None:
        super().__init__("-", [left, right])


class Multiply(BinaryOperation):
    def __init__(self, left: Node, right: Node) -> None:
        super().__init__("*", [left, right])


class Divide(BinaryOperation):
    def __init__(self, left: Node, right: Node) -> None:
        super().__init__("/", [left, right])


class Exponentiation(BinaryOperation):
    def __init__(self, base: Node, exponent: Node) -> None:
        super().__init__("^", [base, exponent])


class Function(Operation):
    def __str__(self) -> str:
        arg_str = ", ".join(map(str, self.operands))
        return f"{self.operator}({arg_str})"
