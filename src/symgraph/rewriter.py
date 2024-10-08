# rewriter.py
from symgraph.expression import (
    Add,
    Constant,
    Divide,
    Exp,
    Exponentiation,
    Ln,
    Multiply,
    Node,
    Operation,
    Subtract,
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
    """Simplify multiplication by zero: x * 0 = 0, 0 * x = 0."""
    if isinstance(node, Multiply):
        if node.left == 0 or node.right == 0:
            return Constant(value=0)
    return node


def simplify_divide_by_zero_numerator(node: Node) -> Node:
    if isinstance(node, Divide):
        if isinstance(node.left, Constant) and node.left.value == 0:
            return Constant(value=0)
    return node


def simplify_multiply_by_one(node: Node) -> Node:
    """Simplify multiplication by one: x * 1 = x, 1 * x = x."""
    if isinstance(node, Multiply):
        if node.left == 1:
            return node.right
        if node.right == 1:
            return node.left
    return node


def simplify_add_zero(node: Node) -> Node:
    """Simplify addition with zero: x + 0 = x, 0 + x = x."""
    if isinstance(node, Add):
        if node.left == 0:
            return node.right
        if node.right == 0:
            return node.left
    return node


def simplify_subtract_zero(node: Node) -> Node:
    """Simplify subtraction of zero: x - 0 = x."""
    if isinstance(node, Subtract):
        if node.right == 0:
            return node.left
    return node


def simplify_exponentiation(node: Node) -> Node:
    """Simplify exponentiation: x^0 = 1, x^1 = x."""
    if isinstance(node, Exponentiation):
        if node.right == 0:
            return Constant(value=1)
        if node.right == 1:
            return node.left
    return node


def simplify_fractional_multiplication(node: Node) -> Node:
    """Simplify expressions like a/a * b = b and b * a/a = b."""
    if isinstance(node, Multiply):
        if isinstance(node.left, Divide) and node.left.left == node.left.right:
            return node.right  # a/a * b = b
        if isinstance(node.right, Divide) and node.right.left == node.right.right:
            return node.left  # b * a/a = b
    elif isinstance(node, Divide):
        if node.left == node.right:
            return Constant(value=1)  # a / a = 1
    return node


def simplify_divide_with_common_factor(node: Node) -> Node:
    """Simplify expressions like a * b / a = b."""
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


def simplify_ln_of_e_power(node: Node) -> Node:
    """Simplify subtraction of zero: x - 0 = x."""
    if isinstance(node, Ln):
        if isinstance(node.operand, Exp):
            return node.operand.operand
    return node


def simplify_e_power_of_ln(node: Node) -> Node:
    """Simplify subtraction of zero: x - 0 = x."""
    if isinstance(node, Exp):
        if isinstance(node.operand, Ln):
            return node.operand.operand
    return node


def simplify_ln_of_mul(node: Node) -> Node:
    if isinstance(node, Ln):
        mul_node = node.operand
        if isinstance(mul_node, Multiply):
            return Ln(mul_node.left) + Ln(mul_node.right)
    return node


all_rules = [
    simplify_multiply_by_zero,
    simplify_multiply_by_one,
    simplify_add_zero,
    simplify_subtract_zero,
    simplify_exponentiation,
    simplify_divide_by_zero_numerator,
    simplify_fractional_multiplication,
    simplify_divide_with_common_factor,
    simplify_ln_of_e_power,
    simplify_e_power_of_ln,
    simplify_ln_of_mul,
]

rewriter = Rewriter(rules=all_rules)
