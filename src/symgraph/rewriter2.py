from symgraph.expression2 import (
    Add,
    Constant,
    Divide,
    Exponentiation,
    Function,
    Multiply,
    Node,
    Operation,
    Subtract,
    Symbol,
)


from dataclasses import dataclass
from typing import Callable


@dataclass
class Rewriter:
    rules: list[Callable[[Node], Node]]

    def __call__(self, node: Node) -> Node:
        simplified_node = node
        while True:
            new_node = self._apply_rules_recursively(simplified_node)
            if new_node == simplified_node:
                break
            simplified_node = new_node
        return simplified_node

    def _apply_rules_recursively(self, node: Node) -> Node:
        # First, apply rules to the operands (recursive step)
        if isinstance(node, Operation):
            node.operands = [self._apply_rules_recursively(op) for op in node.operands]

        # Now, apply the rules to the current node
        for rule in self.rules:
            simplified_node = rule(node)
            if simplified_node != node:
                # If a simplification happened, restart to apply all rules on the new simplified node
                return self._apply_rules_recursively(simplified_node)

        return node


def simplify_multiply_by_zero(node: Node) -> Node:
    if isinstance(node, Multiply):
        if isinstance(node.left, Constant) and node.left.value == 0:
            return Constant(value=0)
        if isinstance(node.right, Constant) and node.right.value == 0:
            return Constant(value=0)
    return node


def simplify_divide_by_zero_numerator(node: Node) -> Node:
    if isinstance(node, Divide):
        if isinstance(node.left, Constant) and node.left.value == 0:
            return Constant(value=0)
    return node


def simplify_multiply_by_one(node: Node) -> Node:
    if isinstance(node, Multiply):
        if isinstance(node.left, Constant) and node.left.value == 1:
            return node.right
        if isinstance(node.right, Constant) and node.right.value == 1:
            return node.left
    return node


def simplify_add_zero(node: Node) -> Node:
    if isinstance(node, Add):
        if isinstance(node.left, Constant) and node.left.value == 0:
            return node.right
        if isinstance(node.right, Constant) and node.right.value == 0:
            return node.left
    return node


def simplify_subtract_zero(node: Node) -> Node:
    if isinstance(node, Subtract):
        if isinstance(node.right, Constant) and node.right.value == 0:
            return node.left
    return node


def simplify_exponentiation(node: Node) -> Node:
    if isinstance(node, Exponentiation):
        if isinstance(node.right, Constant):
            if node.right.value == 0:
                return Constant(value=1)
            if node.right.value == 1:
                return node.left
    return node


def simplify_fractional_multiplication(node: Node) -> Node:
    if isinstance(node, Multiply):
        # Check if one operand is a division of the same symbol (a / a)
        if isinstance(node.left, Divide) and node.left.left == node.left.right:
            return node.right  # a/a * b = b
        if isinstance(node.right, Divide) and node.right.left == node.right.right:
            return node.left  # b * a/a = b
    elif isinstance(node, Divide):
        # Check if division comes first and is then multiplied (b * a/a = b)
        if node.left == node.right:
            return Constant(value=1)  # Simplify a / a = 1
    return node


def simplify_divide_with_common_factor(node: Node) -> Node:
    if isinstance(node, Divide):
        numerator = node.left
        denominator = node.right

        # If numerator is a multiplication, check for common factors
        if isinstance(numerator, Multiply):
            if numerator.left == denominator:
                return numerator.right  # a * b / a = b
            if numerator.right == denominator:
                return numerator.left  # b * a / a = b
    return node


### Differentiator ###
@dataclass
class Differentiator:
    def __call__(self, node: Node, var: Symbol) -> Node:
        return differentiate_node(node, var)


def differentiate_node(node: Node, var: Symbol) -> Node:
    match node:
        case Add(left=left, right=right):
            return Add(
                left=differentiate_node(left, var), right=differentiate_node(right, var)
            )

        case Subtract(left=left, right=right):
            return Subtract(
                left=differentiate_node(left, var), right=differentiate_node(right, var)
            )

        case Multiply(left=left, right=right):
            return Add(
                left=Multiply(left=differentiate_node(left, var), right=right),
                right=Multiply(left=left, right=differentiate_node(right, var)),
            )

        case Divide(left=left, right=right):
            numerator = Subtract(
                left=Multiply(left=differentiate_node(left, var), right=right),
                right=Multiply(left=left, right=differentiate_node(right, var)),
            )
            denominator = Multiply(left=right, right=right)
            return Divide(left=numerator, right=denominator)

        case Exponentiation(left=base, right=exponent):
            if isinstance(exponent, Constant):
                return Multiply(
                    left=Multiply(
                        left=exponent,
                        right=Exponentiation(
                            left=base, right=Constant(value=exponent.value - 1)
                        ),
                    ),
                    right=differentiate_node(base, var),
                )
            else:
                return Multiply(
                    left=Exponentiation(left=base, right=exponent),
                    right=Add(
                        left=Multiply(
                            left=differentiate_node(exponent, var),
                            right=Function(operator="ln", arguments=[base]),
                        ),
                        right=Multiply(
                            left=exponent,
                            right=Divide(
                                left=differentiate_node(base, var), right=base
                            ),
                        ),
                    ),
                )

        case Symbol(operator=name):
            return Constant(value=1) if name == var.name else Constant(value=0)

        case Constant():
            return Constant(value=0)

        case _:
            raise NotImplementedError(f"Differentiation not supported for node: {node}")
