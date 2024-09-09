# test_simplification.py

from symgraph.expression import Add, Constant, Divide, Exponentiation, Multiply, Subtract, Symbol


def test_symbol_add_constant():
    a = Symbol(name="a")
    expr = a + 3
    assert isinstance(expr, Add)
    assert isinstance(expr.right, Constant)
    assert expr.right.value == 3


def test_radd_symbol():
    a = Symbol(name="a")
    expr = 3 + a
    assert isinstance(expr, Add)
    assert isinstance(expr.left, Constant)
    assert expr.left.value == 3
    assert expr.right == a


def test_symbol_mul_constant():
    a = Symbol(name="a")
    expr = a * 5
    assert isinstance(expr, Multiply)
    assert isinstance(expr.right, Constant)
    assert expr.right.value == 5


def test_rmul_symbol():
    a = Symbol(name="a")
    expr = 5 * a
    assert isinstance(expr, Multiply)
    assert isinstance(expr.left, Constant)
    assert expr.left.value == 5
    assert expr.right == a


def test_symbol_pow_constant():
    a = Symbol(name="a")
    expr = a ** 2
    assert isinstance(expr, Exponentiation)
    assert isinstance(expr.right, Constant)
    assert expr.right.value == 2


def test_rpow_symbol():
    a = Symbol(name="a")
    expr = 2 ** a
    assert isinstance(expr, Exponentiation)
    assert isinstance(expr.left, Constant)
    assert expr.left.value == 2
    assert expr.right == a


def test_rtruediv_symbol():
    a = Symbol(name="a")
    expr = 10 / a
    assert isinstance(expr, Divide)
    assert isinstance(expr.left, Constant)
    assert expr.left.value == 10
    assert expr.right == a


def test_rsub_symbol():
    a = Symbol(name="a")
    expr = 5 - a
    assert isinstance(expr, Subtract)
    assert isinstance(expr.left, Constant)
    assert expr.left.value == 5
    assert expr.right == a
