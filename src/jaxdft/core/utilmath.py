from jax import numpy as jnp

SQRT_PI = jnp.sqrt(jnp.pi)


def factorial(n):
    return jnp.prod(jnp.arange(1, n + 1))


def double_factorial(n):
    if n < 0:
        return 1
    elif n == 0:
        return 1
    elif n == 1:
        return 1
    elif n == 2:
        return 2
    else: # n >= 3
        return n * double_factorial(n - 2)
