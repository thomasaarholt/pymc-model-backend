# Example usage
from symgraph.differentiator import differentiate_node
from symgraph.expression import Multiply, Symbol
from symgraph.rewriter import (
    Rewriter,
    simplify_add_zero,
    simplify_multiply_by_one,
    simplify_multiply_by_zero,
)


a = Symbol("a")
b = Symbol("b")
expr = Multiply(a, b)

# Differentiating the expression
diff_result = differentiate_node(expr, a)
print("Differentiation")
print(diff_result)  # Output: (1 * b + a * 0)

# Simplification
rewriter = Rewriter(
    rules=[simplify_multiply_by_zero, simplify_multiply_by_one, simplify_add_zero]
)
simplified_expr = rewriter(diff_result)
print("Then simplification")
print(simplified_expr)  # Output: b
