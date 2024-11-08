from symgraph import differentiator
from symgraph import rewriter
from symgraph.expression import (
    Constant,
    Exp,
    Node,
    Pi,
    Sqrt,
    Symbol,
)


# Example usage with the SimplificationSystem
# Normal distribution representation using the symbolic math graph


def normal_distribution(mu: Node, sigma: Node, x: Node) -> Node:
    sigma_squared = sigma**2
    preamble = 1 / Sqrt(Constant(2) * Pi * sigma_squared)
    exp = Exp(-((x - mu) ** 2) / (2 * sigma_squared))
    return preamble * exp


# Example usage
mu = Symbol("mu", symbol="μ")
sigma = Symbol("sigma", symbol="σ")
x = Symbol("x")

# Create the normal distribution expression
normal_dist = normal_distribution(mu, sigma, x)
# print(f"Normal distribution expression: {normal_dist}")
print(normal_dist.to_latex())

# Differentiate the normal distribution with respect to x
normal_dist_diff = differentiator.differentiate_node(normal_dist, x)
# print(f"Derivative of normal distribution w.r.t. x: {normal_dist_diff}")
# Simplification System

# Simplify the derivative
simplified_diff = rewriter(normal_dist_diff)
# print(f"Simplified derivative: {simplified_diff}")
# print(f"Simplified derivative: {simplified_diff.to_latex()}")
foo = Constant(-1) * sigma / (x - mu)
print(foo.to_latex())
print(rewriter(foo).to_latex())
