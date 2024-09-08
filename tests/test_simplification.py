import pytest

from symgraph.expression import Add, Constant, Divide, Exponentiation, Multiply, Subtract, Symbol
from symgraph.simplify import Rewriter, simplify_add_zero, simplify_divide_by_zero_numerator, simplify_divide_with_common_factor, simplify_exponentiation, simplify_fractional_multiplication, simplify_multiply_by_one, simplify_multiply_by_zero, simplify_subtract_zero

# Assuming all simplification rules, Node, Symbol, Constant, Multiply, Add, etc. have been imported
# from the previous code

@pytest.fixture
def simplification_system():
    return Rewriter([
        simplify_multiply_by_zero,
        simplify_multiply_by_one,
        simplify_add_zero,
        simplify_subtract_zero,
        simplify_exponentiation,
        simplify_divide_by_zero_numerator,
        simplify_fractional_multiplication,
        simplify_divide_with_common_factor
    ])


def test_multiply_by_zero(simplification_system):
    a = Symbol("a")
    expr = Multiply(a, Constant(0))
    simplified = simplification_system.apply(expr)
    assert str(simplified) == "0"


def test_multiply_by_one(simplification_system):
    a = Symbol("a")
    expr = Multiply(a, Constant(1))
    simplified = simplification_system.apply(expr)
    assert str(simplified) == "a"


def test_add_zero(simplification_system):
    a = Symbol("a")
    expr = Add(a, Constant(0))
    simplified = simplification_system.apply(expr)
    assert str(simplified) == "a"


def test_subtract_zero(simplification_system):
    a = Symbol("a")
    expr = Subtract(a, Constant(0))
    simplified = simplification_system.apply(expr)
    assert str(simplified) == "a"


def test_exponentiation_zero(simplification_system):
    a = Symbol("a")
    expr = Exponentiation(a, Constant(0))
    simplified = simplification_system.apply(expr)
    assert str(simplified) == "1"


def test_exponentiation_one(simplification_system):
    a = Symbol("a")
    expr = Exponentiation(a, Constant(1))
    simplified = simplification_system.apply(expr)
    assert str(simplified) == "a"


def test_divide_zero_numerator(simplification_system):
    a = Symbol("a")
    expr = Divide(Constant(0), a)
    simplified = simplification_system.apply(expr)
    assert str(simplified) == "0"


def test_divide_same_symbol(simplification_system):
    a = Symbol("a")
    expr = Divide(a, a)
    simplified = simplification_system.apply(expr)
    assert str(simplified) == "1"


def test_multiply_and_divide_cancel(simplification_system):
    a = Symbol("a")
    b = Symbol("b")
    expr = Divide(Multiply(a, b), a)
    simplified = simplification_system.apply(expr)
    assert str(simplified) == "b"


def test_divide_and_multiply_cancel(simplification_system):
    a = Symbol("a")
    b = Symbol("b")
    expr = Multiply(b, Divide(a, a))
    simplified = simplification_system.apply(expr)
    assert str(simplified) == "b"
