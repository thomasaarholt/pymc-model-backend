from symgraph.expression import (
    Add,
    Constant,
    Divide,
    Exp,
    Exponentiation,
    Ln,
    Multiply,
    Negation,
    Node,
    Sqrt,
    Subtract,
    Symbol,
)


def is_constant_wrt(node: Node, var: Symbol) -> bool:
    match node:
        case Constant():
            return True  # Constants are constant
        case Symbol() if node == var:
            return False  # Variable itself is not constant
        case Symbol():
            return True  # Other symbols are considered constants wrt `var`
        case Add(left=left, right=right):
            return is_constant_wrt(left, var) and is_constant_wrt(right, var)
        case Subtract(left=left, right=right):
            return is_constant_wrt(left, var) and is_constant_wrt(right, var)
        case Multiply(left=left, right=right):
            return is_constant_wrt(left, var) and is_constant_wrt(right, var)
        case Negation(operand=operand):
            return is_constant_wrt(operand, var)
        case Divide(left=left, right=right):
            return is_constant_wrt(left, var) and is_constant_wrt(right, var)
        case Exp(operand=operand):
            return is_constant_wrt(operand, var)
        case Ln(operand=operand):
            return is_constant_wrt(operand, var)
        case _:
            return False


def differentiate_node(node: Node, var: Symbol) -> Node:
    """
    Differentiates a given expression node with respect to a variable.

    Args:
        node (Node): The expression node to be differentiated.
        var (Symbol): The variable with respect to which the differentiation is performed.

    Returns:
        Node: The derivative of the input expression node.
    """
    if is_constant_wrt(node, var):
        return Constant(value=0)

    match node:
        case Add(left=left, right=right):
            return differentiate_node(left, var) + differentiate_node(right, var)

        case Subtract(left=left, right=right):
            return differentiate_node(left, var) - differentiate_node(right, var)

        case Multiply(left=left, right=right):
            return product_rule(left, right, var)

        case Negation(operand=operand):
            return product_rule(Constant(-1), operand, var)

        case Divide(left=left, right=right):
            return quotient_rule(left, right, var)

        case Exponentiation(left=base, right=exponent):
            return power_rule(base, exponent, var)

        case Sqrt(operand=operand):
            return (1 / (2 * Sqrt(operand))) * differentiate_node(operand, var)

        case Exp(operand=operand):
            return Exp(operand) * differentiate_node(operand, var)

        case Ln(operand=operand):
            return (1 / operand) * differentiate_node(operand, var)

        case Symbol():
            return differentiate_symbol(node, var)

        case Constant():
            return differentiate_constant(node, var)

        case _:
            raise NotImplementedError(f"Differentiation not supported for node: {node}")


def product_rule(left: Node, right: Node, var: Symbol) -> Node:
    """
    Apply the product rule to differentiate the product of two expressions.

    Args:
        left (Node): The left-hand expression.
        right (Node): The right-hand expression.
        var (Symbol): The variable to differentiate with respect to.

    Returns:
        Node: The result of applying the product rule.
    """
    return Add(
        left=Multiply(left=differentiate_node(left, var), right=right),
        right=Multiply(left=left, right=differentiate_node(right, var)),
    )


def quotient_rule(left: Node, right: Node, var: Symbol) -> Node:
    """
    Apply the quotient rule to differentiate the given expression.

    Args:
        left (Node): The left operand of the expression to differentiate.
        right (Node): The right operand of the expression to differentiate.
        var (Symbol): The variable with respect to which the differentiation is performed.

    Returns:
        Node: The result of applying the quotient rule to the given expression.
    """
    numerator = Subtract(
        left=Multiply(left=differentiate_node(left, var), right=right),
        right=Multiply(left=left, right=differentiate_node(right, var)),
    )
    denominator = Multiply(left=right, right=right)
    return Divide(left=numerator, right=denominator)


def power_rule(base: Node, exponent: Node, var: Symbol) -> Node:
    """
    Apply the power rule to differentiate the given expression.

    Args:
        base (Node): The base operand of the expression to differentiate.
        exponent (Node): The exponent operand of the expression to differentiate.
        var (Symbol): The variable with respect to which the differentiation is performed.

    Returns:
        Node: The result of applying the quotient rule to the given expression.
    """
    # Power rule: d(u^n)/dx = n * u^(n-1) * du/dx
    if isinstance(exponent, Constant):
        if exponent.value == 0:
            return Constant(value=0)  # d(c^0)/dx = 0
        elif exponent.value == 1:
            return differentiate_node(base, var)  # d(c^1)/dx = du/dx
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
    """
    Differentiates a symbol with respect to a variable.

    Args:
        symbol (Symbol): The symbol to differentiate.
        var (Symbol): The variable to differentiate with respect to.

    Returns:
        Node: The derivative of the symbol with respect to the variable.
    """
    return Constant(value=1) if symbol.name == var.name else Constant(value=0)


def differentiate_constant(constant: Constant, var: Symbol) -> Node:
    """
    Differentiates a constant function with respect to a given variable.

    Args:
        constant (Constant): The constant function to be differentiated.
        var (Symbol): The variable with respect to which the function is to be differentiated.

    Returns:
        Node: The derivative of the constant function.
    """
    return Constant(value=0)
