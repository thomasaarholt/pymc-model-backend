# rewriter.py
from symgraph.expression import (
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
    """Simplify multiplication by zero: x * 0 = 0, 0 * x = 0."""
    if isinstance(node, Multiply):
        if node.left.is_zero() or node.right.is_zero():
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
        if node.left.is_one():
            return node.right
        if node.right.is_one():
            return node.left
    return node


def simplify_add_zero(node: Node) -> Node:
    """Simplify addition with zero: x + 0 = x, 0 + x = x."""
    if isinstance(node, Add):
        if node.left.is_zero():
            return node.right
        if node.right.is_zero():
            return node.left
    return node


def simplify_subtract_zero(node: Node) -> Node:
    """Simplify subtraction of zero: x - 0 = x."""
    if isinstance(node, Subtract):
        if node.right.is_zero():
            return node.left
    return node


def simplify_exponentiation(node: Node) -> Node:
    """Simplify exponentiation: x^0 = 1, x^1 = x."""
    if isinstance(node, Exponentiation):
        if node.right.is_zero():
            return Constant(value=1)
        if node.right.is_one():
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


### Differentiator ###
@dataclass
class Differentiator:
    def __call__(self, node: Node, var: Symbol) -> Node:
        return differentiate_node(node, var)


def product_rule(left: Node, right: Node, var: Symbol) -> Node:
    return Add(
        left=Multiply(left=differentiate_node(left, var), right=right),
        right=Multiply(left=left, right=differentiate_node(right, var)),
    )


def quotient_rule(left: Node, right: Node, var: Symbol) -> Node:
    numerator = Subtract(
        left=Multiply(left=differentiate_node(left, var), right=right),
        right=Multiply(left=left, right=differentiate_node(right, var)),
    )
    denominator = Multiply(left=right, right=right)
    return Divide(left=numerator, right=denominator)


def power_rule(base: Node, exponent: Node, var: Symbol) -> Node:
    if isinstance(exponent, Constant):
        # Power rule: d(u^n)/dx = n * u^(n-1) * du/dx
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
        # Chain rule for non-constant exponents: d(u^v)/dx = u^v * (v' * ln(u) + v * (u'/u))
        return Multiply(
            left=Exponentiation(left=base, right=exponent),
            right=Add(
                left=Multiply(
                    left=differentiate_node(exponent, var),
                    right=Function(operator="ln", arguments=[base]),
                ),
                right=Multiply(
                    left=exponent,
                    right=Divide(left=differentiate_node(base, var), right=base),
                ),
            ),
        )


def differentiate_symbol(symbol: Symbol, var: Symbol) -> Node:
    return Constant(value=1) if symbol.name == var.name else Constant(value=0)


def differentiate_constant(constant: Constant, var: Symbol) -> Node:
    return Constant(value=0)


def differentiate_node(node: Node, var: Symbol) -> Node:
    match node:
        case Add(left=left, right=right):
            # Sum rule: d(u + v)/dx = du/dx + dv/dx
            return Add(
                left=differentiate_node(left, var), right=differentiate_node(right, var)
            )

        case Subtract(left=left, right=right):
            # Difference rule: d(u - v)/dx = du/dx - dv/dx
            return Subtract(
                left=differentiate_node(left, var), right=differentiate_node(right, var)
            )

        case Multiply(left=left, right=right):
            # Product rule: d(u * v)/dx = u' * v + u * v'
            return product_rule(left, right, var)

        case Divide(left=left, right=right):
            # Quotient rule: d(u / v)/dx = (u' * v - u * v') / v^2
            return quotient_rule(left, right, var)

        case Exponentiation(left=base, right=exponent):
            # Power rule or chain rule for exponentiation
            return power_rule(base, exponent, var)

        case Symbol():
            # Symbol differentiation (dx/dx = 1, dy/dx = 0)
            return differentiate_symbol(node, var)

        case Constant():
            # Constant differentiation (d(constant)/dx = 0)
            return differentiate_constant(node, var)

        case _:
            raise NotImplementedError(f"Differentiation not supported for node: {node}")
