from jax import numpy as jnp
import numpy as np

from jaxdft.core.utilmath import double_factorial


def k_gto(a, b, dAB_2):
    """
    Return the often Used K value in GTO calculation.
    :param a:
    :param b:
    :param dAB:
    :return: K $K=\\exp{-ab/(a+b)|AB|^2}$
    """
    return jnp.exp(-a * b / (a + b) * dAB_2)

def norm_gto(a, la=0, ma=0, na=0):
    return (2 * a / jnp.pi) ** (3 / 4) * ((4 * a) ** (la + ma + na) / (
                double_factorial(2 * la - 1) * double_factorial(2 * ma - 1) * double_factorial(2 * na - 1))) ** (1 / 2)
