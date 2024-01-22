# Owner(s): ["module: inductor"]

from sympy import Symbol
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor.utils import sympy_subs


class TestUtils(TestCase):
    def testSympySubs(self):
        # integer and nonnegetaive attributes are preserved.
        expr = Symbol("x")
        result = sympy_subs(expr, {expr: "y"})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, None)
        self.assertEqual(result.is_nonnegative, None)

        expr = Symbol("x", integer=True, nonnegative=False)
        result = sympy_subs(expr, {expr: "y"})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, True)
        self.assertEqual(result.is_nonnegative, False)

        # invalid replacement.
        expr = Symbol("x", integer=True)
        result = sympy_subs(expr, {Symbol("x"): Symbol("y")})
        self.assertEqual(result.name, "x")

        # valid replacement since properties match.
        expr = Symbol("x", integer=True)
        result = sympy_subs(expr, {Symbol("x", integer=True): Symbol("y")})
        self.assertEqual(result.name, "y")

        # invalid replacement.
        expr = Symbol("x", integer=None)
        result = sympy_subs(expr, {Symbol("x", integer=False): Symbol("y")})
        self.assertEqual(result.name, "x")

        # When replaced is a string x, the functions will try to replace the symbols with name x
        # with all possible inetegr, non-negative attributes while also preserving the attributes.
        expr = Symbol("x")
        result = sympy_subs(expr, {"x": "y"})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, None)
        self.assertEqual(result.is_nonnegative, None)

        expr = Symbol("x", integer=True, nonnegative=True)
        result = sympy_subs(expr, {"x": "y"})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, True)
        self.assertEqual(result.is_nonnegative, True)

        expr = Symbol("x", integer=None, nonnegative=False)
        result = sympy_subs(expr, {"x": "y"})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, None)
        self.assertEqual(result.is_nonnegative, False)

        # replace is string and replacement is a symbol.
        expr = Symbol("x")
        result = sympy_subs(expr, {"x": Symbol("y", integer=True, nonnegative=False)})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, True)
        self.assertEqual(result.is_nonnegative, False)


if __name__ == "__main__":
    run_tests()
