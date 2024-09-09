from symgraph.expression import (
    Add,
    Constant,
    Divide,
    Exp,
    Exponentiation,
    Ln,
    Multiply,
    Node,
    Sqrt,
    Subtract,
    Symbol,
)


from dataclasses import dataclass


def differentiate_node(node: Node, var: Symbol) -> Node:
    match node:
        case Add(left=left, right=right):
            # Sum rule: d(u + v)/dx = du/dx + dv/dx
            return differentiate_node(left, var) + differentiate_node(right, var)

        case Subtract(left=left, right=right):
            # Difference rule: d(u - v)/dx = du/dx - dv/dx
            return differentiate_node(left, var) - differentiate_node(right, var)

        case Multiply(left=left, right=right):
            # Product rule: d(u * v)/dx = u' * v + u * v'
            return product_rule(left, right, var)

        case Divide(left=left, right=right):
            # Quotient rule: d(u / v)/dx = (u' * v - u * v') / v^2
            return quotient_rule(left, right, var)

        case Exponentiation(left=base, right=exponent):
            # Power rule or chain rule for exponentiation
            return power_rule(base, exponent, var)

        case Sqrt(operand=operand):
            # Derivative of sqrt(f(x)) is 1 / (2 * sqrt(f(x))) * f'(x)
            return (1 / (2 * Sqrt(operand))) * differentiate_node(operand, var)

        case Exp(operand=operand):
            # Derivative of exp(f(x)) is exp(f(x)) * f'(x)
            return Exp(operand) * differentiate_node(operand, var)

        case Ln(operand=operand):
            # Derivative of ln(f(x)) is 1 / f(x) * f'(x)
            return (1 / operand) * differentiate_node(operand, var)

        case Symbol():
            # Symbol differentiation (dx/dx = 1, dy/dx = 0)
            return differentiate_symbol(node, var)

        case Constant():
            # Constant differentiation (d(constant)/dx = 0)
            return differentiate_constant(node, var)

        case _:
            raise NotImplementedError(f"Differentiation not supported for node: {node}")


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
                    right=Ln(base),
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
