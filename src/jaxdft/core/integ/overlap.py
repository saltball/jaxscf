import logging
import jax
from jax import lax
import numpy as np
from jax import numpy as jnp
from jaxdft.core.integ.gtomath import k_gto, norm_gto
from jaxdft.core.utilmath import SQRT_PI, double_factorial, factorial


# recursive version

def _sij_slow(a, b, pa_x, pb_x, la, lb):
    """component of overlap <x^la_A a r_A|x^lb_B b r_B>
    
    """
    # if la < 0 or lb < 0:
    def la_lt_0_lb_lt_0(a, b, pa_x, pb_x, la, lb):
        return 0.0
    
    # elif la == 0 and lb == 0:
    def la_eq_0_lb_eq_0(a, b, pa_x, pb_x, la, lb):
        return SQRT_PI / np.power(a + b, 0.5)
    
    # elif la > 0 and lb == 0:
    def la_gt_0_lb_eq_0(a, b, pa_x, pb_x, la, lb):
        return pa_x * _sij_slow(a, b, pa_x, pb_x, la - 1, lb) + 1 / (2 * (a + b)) * ((la - 1) * _sij_slow(a, b, pa_x, pb_x, la - 2, lb))
    
    # elif la == 0 and lb > 0:
    def la_eq_0_lb_gt_0(a, b, pa_x, pb_x, la, lb):
        return pb_x * _sij_slow(a, b, pa_x, pb_x, la, lb - 1) + 1 / (2 * (a + b)) * ((lb - 1) * _sij_slow(a, b, pa_x, pb_x, la, lb - 2))
    
    # elif la > 0 and lb > 0:
    def la_gt_0_lb_gt_0(a, b, pa_x, pb_x, la, lb):
        return pa_x * _sij_slow(a, b, pa_x, pb_x, la - 1, lb) + 1 / (2 * (a + b)) * (
                    (la - 1) * _sij_slow(a, b, pa_x, pb_x, la - 2, lb) + lb * _sij_slow(a, b, pa_x, pb_x, la - 1, lb - 1))
    
    variable = (a, b, pa_x, pb_x, la, lb)
    
    if la < 0 or lb < 0:
        return la_lt_0_lb_lt_0(*variable)
    elif la == 0 and lb == 0:
        return la_eq_0_lb_eq_0(*variable)
    elif la > 0 and lb == 0:
        return la_gt_0_lb_eq_0(*variable)
    elif la == 0 and lb > 0:
        return la_eq_0_lb_gt_0(*variable)
    elif la > 0 and lb > 0:
        return la_gt_0_lb_gt_0(*variable)
    else:
        raise NotImplementedError("How can you reach here?")


def _Sxyz_slow(a, b, a_array, b_array, la, ma, na, lb, mb, nb, Norm=True):
    Na = 1
    Nb = 1
    if la < 0 or ma < 0 or na < 0 or lb < 0 or mb < 0 or nb < 0:
        return 0
    if Norm:
        Na = norm_gto(a, la, ma, na)
        Nb = norm_gto(b, lb, mb, nb)
    dAB_2 = ((a_array - b_array) ** 2).sum()
    K = k_gto(a, b, dAB_2)
    p_array = b / (a + b) * b_array + a / (a + b) * a_array
    PA_array = p_array - a_array
    PB_array = p_array - b_array
    Ix = _sij_slow(a, b, PA_array[0], PB_array[0], la, lb)
    Iy = _sij_slow(a, b, PA_array[1], PB_array[1], ma, mb)
    Iz = _sij_slow(a, b, PA_array[2], PB_array[2], na, nb)
    return Na * Nb * K * Ix * Iy * Iz

# jax version

def _i_defold_func(la, lb):
    """
    Get the function to calculate the overlap integral of <x^la_A a r_A|x^lb_B b r_B>
    """
    # if la < 0 or lb < 0:
    @jax.jit
    def la_lt_0_lb_lt_0(a, b, pa_x, pb_x):
        return 0.0
    
    # 0
    # 0, 0
    # elif la == 0 and lb == 0:
    @jax.jit
    def la_eq_0_lb_eq_0(a, b, pa_x, pb_x):
        return SQRT_PI / jnp.power(a + b, 0.5)
    
    # 1
    # 0, 1
    # elif la == 0 and lb == 1:
    @jax.jit
    def la_eq_0_lb_eq_1(a, b, pa_x, pb_x):
        return pb_x * SQRT_PI / jnp.sqrt(a + b)
    
    # 1, 0
    # elif la == 1 and lb == 0:
    @jax.jit
    def la_eq_1_lb_eq_0(a, b, pa_x, pb_x):
        return pa_x * SQRT_PI / jnp.sqrt(a + b)
    
    # 2
    # 0, 2
    # elif la == 0 and lb == 2:
    @jax.jit
    def la_eq_0_lb_eq_2(a, b, pa_x, pb_x):
        p = a + b
        return SQRT_PI * (pb_x * pb_x / jnp.sqrt(p) + 1 / ( jnp.power(p, 1.5) * 2))
    
    # 1, 1
    # elif la == 1 and lb == 1:
    @jax.jit
    def la_eq_1_lb_eq_1(a, b, pa_x, pb_x):
        p = a + b
        return SQRT_PI * (pb_x * pa_x / jnp.sqrt(p) + 1 / ( jnp.power(p, 1.5) * 2))
    
    # 2, 0
    # elif la == 2 and lb == 0:
    @jax.jit
    def la_eq_2_lb_eq_0(a, b, pa_x, pb_x):
        p = a + b
        return SQRT_PI * (pa_x * pa_x / jnp.sqrt(p) + 1 / ( jnp.power(p, 1.5) * 2))
    
    # 3
    # 0, 3
    # elif la == 0 and lb == 3:
    @jax.jit
    def la_eq_0_lb_eq_3(a, b, pa_x, pb_x):
        p = a + b
        return SQRT_PI / 2 * pb_x * (3 / jnp.power(p, 1.5) + 2 * jnp.power(pb_x, 2) / jnp.sqrt(p))
    
    # 1, 2
    # elif la == 1 and lb == 2:
    @jax.jit
    def la_eq_1_lb_eq_2(a, b, pa_x, pb_x):
        p = a + b
        return SQRT_PI * (pb_x / jnp.power(p, 1.5) + pa_x * (0.5 / jnp.power(p, 1.5) + jnp.power(pb_x, 2) / jnp.sqrt(p)))
    
    # 2, 1
    # elif la == 2 and lb == 1:
    @jax.jit
    def la_eq_2_lb_eq_1(a, b, pa_x, pb_x):
        p = a + b
        return SQRT_PI * (pa_x / jnp.power(p, 1.5) + pb_x * (0.5 / jnp.power(p, 1.5) + jnp.power(pa_x, 2) / jnp.sqrt(p)))
    
    # 3, 0
    # elif la == 3 and lb == 0:
    @jax.jit
    def la_eq_3_lb_eq_0(a, b, pa_x, pb_x):
        p = a + b
        return SQRT_PI / 2 * pa_x * (3 / jnp.power(p, 1.5) + 2 * jnp.power(pa_x, 2) / jnp.sqrt(p))
    
    # general
    # elif la > 0 and lb > 0:
    @jax.jit
    def la_gt_0_lb_gt_0(a, b, pa_x, pb_x):
        p = a + b
        return pa_x * _i_defold_func(la - 1, lb)(a, b, pa_x, pb_x) + 1 / (2 * p) * (
                    (la - 1) * _i_defold_func(la - 2, lb)(a, b, pa_x, pb_x) + lb * _i_defold_func(la - 1, lb - 1)(a, b, pa_x, pb_x))
    
    #
    if la < 0 or lb < 0:
        return la_lt_0_lb_lt_0
    # 0
    elif la == 0 and lb == 0:
        return la_eq_0_lb_eq_0
    # 1
    # 0, 1
    elif la == 0 and lb == 1:
        return la_eq_0_lb_eq_1
    # 1, 0
    elif la == 1 and lb == 0:
        return la_eq_1_lb_eq_0
    # 2
    # 0, 2
    elif la == 0 and lb == 2:
        return la_eq_0_lb_eq_2
    # 1, 1
    elif la == 1 and lb == 1:
        return la_eq_1_lb_eq_1
    # 2, 0
    elif la == 2 and lb == 0:
        return la_eq_2_lb_eq_0
    # 3
    # 0, 3
    elif la == 0 and lb == 3:
        return la_eq_0_lb_eq_3
    # 1, 2
    elif la == 1 and lb == 2:
        return la_eq_1_lb_eq_2
    # 2, 1
    elif la == 2 and lb == 1:
        return la_eq_2_lb_eq_1
    # 3, 0
    elif la == 3 and lb == 0:
        return la_eq_3_lb_eq_0
    # general
    else:
        return la_gt_0_lb_gt_0

def _Sxyz_func(la, ma, na, lb, mb, nb, norm=True):
    """Get function to calculate overlap integral of <x^la y^ma z^na| x^lb y^mb z^nb>"""
    Ix_func = _i_defold_func(la, lb)
    Iy_func = _i_defold_func(ma, mb)
    Iz_func = _i_defold_func(na, nb)

    @jax.jit
    def Sxyz_default(a, b, a_xyz, b_xyz):
        return 0

    # @jax.jit
    def Sxyz(a, b, a_xyz, b_xyz):
        Na = norm_gto(a, la, ma, na) if norm else 1.0
        Nb = norm_gto(b, lb, mb, nb) if norm else 1.0
        dAB_2 = ((a_xyz - b_xyz) ** 2).sum()
        k = k_gto(a, b, dAB_2)
        p_array = b/(a+b)*b_xyz + a/(a+b)*a_xyz
        pa_array = p_array - a_xyz
        pb_array = p_array - b_xyz
        return Na * Nb * k * Ix_func(a, b, pa_array[0], pb_array[0]) * Iy_func(a, b, pa_array[1], pb_array[1]) * Iz_func(a, b, pa_array[2], pb_array[2])
    
    if la < 0 or lb < 0 or ma < 0 or mb < 0 or na < 0 or nb < 0:
        return Sxyz_default
    else:
        return Sxyz


if __name__ == '__main__':
    for _ in range(100):
        a = np.random.rand()*10
        b = np.random.rand()*10
        pa_x = np.random.rand(3)*10
        pb_x = np.random.rand(3)*10
        la = np.random.randint(-1, 3)
        lb = np.random.randint(-1, 3)
        ma = np.random.randint(-1, 3)
        mb = np.random.randint(-1, 3)
        na = np.random.randint(-1, 3)
        nb = np.random.randint(-1, 3)
        norm = np.random.choice([True, False])
        slow_result = _Sxyz_slow(a, b, pa_x, pb_x, la, ma, na, lb, mb, nb, norm)
        S_func = _Sxyz_func(la, ma, na, lb, mb, nb, norm)
        i_result = S_func(a, b, np.array(pa_x), np.array(pb_x))
        if np.allclose(slow_result, i_result):
            print(_, a, b, pa_x, pb_x, la, ma, na, lb, mb, nb, norm, slow_result, i_result, "equal")
        else:
            print(_, a, b, pa_x, pb_x, la, ma, na, lb, mb, nb, norm, slow_result, i_result, "Not equal")
            raise ValueError("Not equal, slow_result = {}, i_result = {}".format(slow_result, i_result))
    # print("All passed")
