from symgraph.expression import (
    Constant,
    Divide,
    Exponentiation,
    Function,
    Multiply,
    Node,
    Subtract,
    Symbol,
)
from symgraph.rewriter import (
    Rewriter,
    simplify_add_zero,
    simplify_divide_by_zero_numerator,
    simplify_divide_with_common_factor,
    simplify_exponentiation,
    simplify_fractional_multiplication,
    simplify_multiply_by_one,
    simplify_multiply_by_zero,
    simplify_subtract_zero,
)


import math


# Example usage with the SimplificationSystem
# Normal distribution representation using the symbolic math graph
def normal_distribution(mu: Node, sigma: Node, x: Node) -> Node:
    two = Constant(2)
    pi = Constant(math.pi)
    half = Constant(0.5)

    sqrt_2_pi = Function("sqrt", [Multiply(two, pi)])
    denom = Multiply(sigma, sqrt_2_pi)
    exponent = Multiply(
        half,
        Function(
            "exp",
            [
                Multiply(
                    Constant(-1),
                    Exponentiation(Divide(Subtract(x, mu), sigma), Constant(2)),
                )
            ],
        ),
    )
    return Multiply(Divide(Constant(1), denom), exponent)


# Example usage
mu = Symbol("mu")
sigma = Symbol("sigma")
x = Symbol("x")

# Create the normal distribution expression
normal_dist = normal_distribution(mu, sigma, x)
print(f"Normal distribution expression: {normal_dist}")

# Differentiate the normal distribution with respect to x
normal_dist_diff = normal_dist.diff(x)
print(f"Derivative of normal distribution w.r.t. x: {normal_dist_diff}")

# Simplification System
simplification_system = Rewriter(
    [
        simplify_multiply_by_zero,
        simplify_divide_by_zero_numerator,
        simplify_multiply_by_one,
        simplify_add_zero,
        simplify_subtract_zero,
        simplify_exponentiation,
        simplify_fractional_multiplication,
        simplify_divide_with_common_factor,
    ]
)

# Simplify the derivative
simplified_diff = simplification_system(normal_dist_diff)
print(f"Simplified derivative: {simplified_diff}")
