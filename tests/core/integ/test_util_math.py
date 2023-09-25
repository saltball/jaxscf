
from jaxdft.core.utilmath import double_factorial, factorial


def test_factorial():
    assert factorial(-1) == 1
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(2) == 2
    assert factorial(3) == 6
    assert factorial(4) == 24
    assert factorial(5) == 120
    assert factorial(6) == 720
    assert factorial(7) == 5040
    assert factorial(8) == 40320
    assert factorial(9) == 9 * factorial(8)
    assert double_factorial(-1) == 1
    assert double_factorial(0) == 1
    assert double_factorial(1) == 1
    assert double_factorial(2) == 2
    assert double_factorial(3) == 3
    assert double_factorial(4) == 8
    assert double_factorial(5) == 15
    assert double_factorial(6) == 48
    assert double_factorial(7) == 105