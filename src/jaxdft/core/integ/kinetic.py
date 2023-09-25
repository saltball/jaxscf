import jax
from jaxdft.core.integ.gtomath import k_gto, norm_gto
from jaxdft.core.integ.overlap import _i_defold_func, _sij_slow
from jaxdft.core.utilmath import SQRT_PI


# recursive version

def _tij_slow(a, b, pa_x, pb_x, la, lb):
    r"""Compute the kinetic integral T_{ij} = <i|-1/2\nabla^2|j> between two primitive Gaussian functions i and j.
    The integral is defined as
    
    .. math::
        T_{ij} = \frac{1}{2(a+b)}\left\{a\left[2b\mathbf{S}_{ij}-(l_b-1)\mathbf{S}_{i,j-2}\right]+b\left[2a\mathbf{S}_{ij}-(l_a-1)\mathbf{S}_{i-2,j}\right]\right\}
        
    where :math:`\mathbf{S}_{ij}` is the overlap integral between the two primitive Gaussian functions i and j.
    
    Args:
        a (float): The exponent of the Gaussian function i.
        b (float): The exponent of the Gaussian function j.
        pa_x (float): The x-coordinate of the center of the Gaussian function i.
        pb_x (float): The x-coordinate of the center of the Gaussian function j.
        la (int): The angular momentum of the Gaussian function i.
        lb (int): The angular momentum of the Gaussian function j.
        
    Returns:
        float: The kinetic integral T_{ij} between the two primitive Gaussian functions i and j.
    """
    if la < 0 or lb < 0:
        return 0
    elif la==0 and lb==0:
        return (a-2*a**2*(pa_x**2+1/(2*(a+b))))*SQRT_PI/(a+b)**0.5
    elif la==0 and lb > 0:
        return pb_x*_tij_slow(a, b, pa_x, pb_x, la, lb-1)+1/(2*(a + b))*((lb-1)*_tij_slow(a, b, pa_x, pb_x, la, lb-2))\
               +a/(a+b)*(2*b*_sij_slow(a, b, pa_x, pb_x, la, lb)-(lb-1)*_sij_slow(a, b, pa_x, pb_x, la, lb-2))
    elif la > 0 and lb==0:
        return pa_x*_tij_slow(a, b, pa_x, pb_x, la-1, lb)+1/(2*(a + b))*((la-1)*_tij_slow(a, b, pa_x, pb_x, la-2, lb))\
               +b/(a+b)*(2*a*_sij_slow(a, b, pa_x, pb_x, la, lb)-(la-1)*_sij_slow(a, b, pa_x, pb_x, la-2, lb))
    else:
        return pa_x*_tij_slow(a, b, pa_x, pb_x, la-1, lb)+1/(2*(a + b))*((la-1)*_tij_slow(a, b, pa_x, pb_x, la-2, lb)+lb*_tij_slow(a, b, pa_x, pb_x, la-1, lb-1))\
               +b/(a+b)*(2*a*_sij_slow(a, b, pa_x, pb_x, la, lb)-(la-1)*_sij_slow(a, b, pa_x, pb_x, la-2, lb))


def _kinetic_slow(a, b, a_array, b_array, la, ma, na, lb, mb, nb, norm=True):
    """Kinetic integral between two Gaussian functions.
    
    :param a: exponent of the Gaussian function i
    :param b: exponent of the Gaussian function j
    :param a_array: position of the center of the Gaussian function i
    :param b_array: position of the center of the Gaussian function j
    :param la: angular momentum of the Gaussian function i
    :param ma: angular momentum of the Gaussian function i
    :param na: angular momentum of the Gaussian function i
    :param lb: angular momentum of the Gaussian function j
    :param mb: angular momentum of the Gaussian function j
    :param nb: angular momentum of the Gaussian function j
    :param norm: whether to normalize the integral
    :return: the kinetic integral between the two Gaussian functions i and j
    """
    Na = norm_gto(a, la, ma, na) if norm else 1.0
    Nb = norm_gto(b, lb, mb, nb) if norm else 1.0
    
    if la < 0 or lb < 0 or ma < 0 or mb < 0 or na < 0 or nb < 0:
        return 0
    dAB_2 = ((a_array - b_array) ** 2).sum()
    K = k_gto(a, b, dAB_2)
    p_array = b / (a + b) * b_array + a / (a + b) * a_array
    PA_array = p_array - a_array
    PB_array = p_array - b_array
    Kl = _tij_slow(a, b, PA_array[0], PB_array[0], la, lb) * _sij_slow(a, b, PA_array[1], PB_array[1], ma, mb) * _sij_slow(a, b, PA_array[2], PB_array[2], na, nb)
    Km = _sij_slow(a, b, PA_array[0], PB_array[0], la, lb) * _tij_slow(a, b, PA_array[1], PB_array[1], ma, mb) * _sij_slow(a, b, PA_array[2], PB_array[2], na, nb)
    Kn = _sij_slow(a, b, PA_array[0], PB_array[0], la, lb) * _sij_slow(a, b, PA_array[1], PB_array[1], ma, mb) * _tij_slow(a, b, PA_array[2], PB_array[2], na, nb)
    return Na * Nb * K * (Kl + Km + Kn)

# jax version
def _t_ij_func(la, lb):
    """Ix part in the kinetic integral, <x^la_A a r_A|-1/2\nabla^2|x^lb_B b r_B>"""
    _s_i_j_func = _i_defold_func(la, lb)
    _s_im2_j_func = _i_defold_func(la-2, lb)
    _s_i_jm2_func = _i_defold_func(la, lb-2)
    
    @jax.jit
    def la_lt_0_lb_lt_0(a, b, pa_x, pb_x):
        return 0.0
    
    @jax.jit
    def la_eq_0_lb_eq_0(a, b, pa_x, pb_x):
        return (a-2*a**2*(pa_x**2+1/(2*(a+b))))*SQRT_PI/(a+b)**0.5
    
    @jax.jit
    def la_eq_0_lb_gt_0(a, b, pa_x, pb_x):
        return pb_x*_t_ij_func(la, lb-1)(a, b, pa_x, pb_x)+1/(2*(a + b))*((lb-1)*_t_ij_func(la, lb-2)(a, b, pa_x, pb_x))\
               +a/(a+b)*(2*b*_s_i_j_func(a, b, pa_x, pb_x)-(lb-1)*_s_i_jm2_func(a, b, pa_x, pb_x))
    
    @jax.jit
    def la_gt_0_lb_eq_0(a, b, pa_x, pb_x):
        return pa_x*_t_ij_func(la-1, lb)(a, b, pa_x, pb_x)+1/(2*(a + b))*((la-1)*_t_ij_func(la-2, lb)(a, b, pa_x, pb_x))\
               +b/(a+b)*(2*a*_s_i_j_func(a, b, pa_x, pb_x)-(la-1)*_s_im2_j_func(a, b, pa_x, pb_x))
    
    @jax.jit
    def la_gt_0_lb_gt_0(a, b, pa_x, pb_x):
        return pa_x*_t_ij_func(la-1, lb)(a, b, pa_x, pb_x)+1/(2*(a + b))*((la-1)*_t_ij_func(la-2, lb)(a, b, pa_x, pb_x)+lb*_t_ij_func(la-1, lb-1)(a, b, pa_x, pb_x))\
                +b/(a+b)*(2*a*_s_i_j_func(a, b, pa_x, pb_x)-(la-1)*_s_im2_j_func(a, b, pa_x, pb_x))

               
    if la < 0 or lb < 0:
        return la_lt_0_lb_lt_0
    elif la==0 and lb==0:
        return la_eq_0_lb_eq_0
    elif la==0 and lb > 0:
        return la_eq_0_lb_gt_0
    elif la > 0 and lb==0:
        return la_gt_0_lb_eq_0
    else:
        return la_gt_0_lb_gt_0
    
def _kinetic_func(a, b, la, ma, na, lb, mb, nb, norm=True):
    """Get the function to calculate the kinetic integral between two Gaussian functions.
    
    :param a: exponent of the Gaussian function i
    :param b: exponent of the Gaussian function j
    :param la: angular momentum of the Gaussian function i
    :param ma: angular momentum of the Gaussian function i
    :param na: angular momentum of the Gaussian function i
    :param lb: angular momentum of the Gaussian function j
    :param mb: angular momentum of the Gaussian function j
    :param nb: angular momentum of the Gaussian function j
    :param norm: whether to normalize the integral
    :return: the kinetic integral between the two Gaussian functions i and j
    """
    Na = norm_gto(a, la, ma, na) if norm else 1.0
    Nb = norm_gto(b, lb, mb, nb) if norm else 1.0
    Ix_func = _i_defold_func(la, lb)
    Iy_func = _i_defold_func(ma, mb)
    Iz_func = _i_defold_func(na, nb)
    Tx_func = _t_ij_func(la, lb)
    Ty_func = _t_ij_func(ma, mb)
    Tz_func = _t_ij_func(na, nb)
    
    def _kinetic_default(a, b, a_xyz, b_xyz):
        return 0
    
    def _kinetic(a, b, a_xyz, b_xyz):
        dAB_2 = jnp.dot(a_xyz - b_xyz, a_xyz - b_xyz)
        k = k_gto(a, b, dAB_2)
        p_array = b/(a+b)*b_xyz + a/(a+b)*a_xyz
        pa_array = p_array - a_xyz
        pb_array = p_array - b_xyz
        Kl = Tx_func(a, b, pa_array[0], pb_array[0]) * Iy_func(a, b, pa_array[1], pb_array[1]) * Iz_func(a, b, pa_array[2], pb_array[2])
        Km = Ix_func(a, b, pa_array[0], pb_array[0]) * Ty_func(a, b, pa_array[1], pb_array[1]) * Iz_func(a, b, pa_array[2], pb_array[2])
        Kn = Ix_func(a, b, pa_array[0], pb_array[0]) * Iy_func(a, b, pa_array[1], pb_array[1]) * Tz_func(a, b, pa_array[2], pb_array[2])
        return Na * Nb * k * (Kl + Km + Kn)
    
    if la < 0 or lb < 0 or ma < 0 or mb < 0 or na < 0 or nb < 0:
        return _kinetic_default
    else:
        return _kinetic


if __name__ == "__main__":
    import time
    def time_func(
        callable,
        *args, **kwargs
    ):
        st_time = time.time()
        result = callable(*args, **kwargs)
        end_time = time.time()
        return (end_time - st_time, result)
    
    import numpy as np
    import jax.numpy as jnp
    from jax import jit

    for _ in range(100):
        a = np.random.rand()*10
        b = np.random.rand()*10
        pa_x = np.random.rand(3)*10
        pb_x = np.random.rand(3)*10
        la = np.random.randint(0, 3)
        lb = np.random.randint(0, 3)
        ma = np.random.randint(0, 3)
        mb = np.random.randint(0, 3)
        na = np.random.randint(0, 3)
        nb = np.random.randint(0, 3)
        norm = np.random.choice([True, False])
        func_time1, slow_result = time_func(_kinetic_slow,a, b, pa_x, pb_x, la, ma, na, lb, mb, nb, norm)
        func_time2, T_func = time_func(_kinetic_func,a, b, la, ma, na, lb, mb, nb, norm)
        i_result = T_func(a, b, np.array(pa_x), np.array(pb_x))
        if np.allclose(slow_result, i_result):
            print(_, a, b, pa_x, pb_x, ma, na, lb, mb, nb, norm, slow_result, i_result, "equal", func_time1/func_time2)
        else:
            print(_, a, b, pa_x, pb_x, ma, na, lb, mb, nb, norm, slow_result, i_result, "Not equal", func_time1/func_time2)
            raise ValueError("Not equal, slow_result = {}, i_result = {}".format(slow_result, i_result))
    # print("All passed")