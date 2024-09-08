# expression.py
from __future__ import annotations

from pydantic import BaseModel


class Node(BaseModel):
    def is_zero(self) -> bool:
        return False

    def is_one(self) -> bool:
        return False

    def is_constant(self) -> bool:
        return False
    def __str__(self) -> str:
        raise NotImplementedError


class Constant(Node):
    value: float


    def is_zero(self) -> bool:
        return self.value == 0

    def is_one(self) -> bool:
        return self.value == 1

    def is_constant(self) -> bool:
        return True
    def __str__(self) -> str:
        return str(self.value)


class Symbol(Node):
    name: str
    def is_constant(self) -> bool:
        return False
    def __str__(self) -> str:
        return self.name


class Operation(Node):
    operator: str

    @property
    def operands(self) -> list[Node]:
        raise NotImplementedError

    @operands.setter
    def operands(self, new_operands: list[Node]):
        raise NotImplementedError

    def __str__(self) -> str:
        operand_str = f" {self.operator} ".join(map(str, self.operands))
        return f"({operand_str})"


class BinaryOperation(Operation):
    left: Node
    right: Node

    @property
    def operands(self) -> list[Node]:
        return [self.left, self.right]

    @operands.setter
    def operands(self, new_operands: list[Node]):
        self.left, self.right = new_operands

class Add(BinaryOperation):
    operator: str = "+"

class Subtract(BinaryOperation):
    operator: str = "-"


class Multiply(BinaryOperation):
    operator: str = "*"


class Divide(BinaryOperation):
    operator: str = "/"


class Exponentiation(BinaryOperation):
    operator: str = "**"


class Function(Operation):
    arguments: list[Node]

    def __str__(self) -> str:
        arg_str = ", ".join(map(str, self.operands))
        return f"{self.operator}({arg_str})"
