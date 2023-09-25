import jax
from jax import lax
from jax import scipy as jscipy


@jax.jit
def Fm(m, w):
    """ 
    Where it defines F_m(w) = int_0^1(exp(-wt^2)t^(2m)dt)
    Used in 徐光宪; 黎乐民; 王德民. 量子化学：基本原理和从头计算法, 第二版.; 科学出版社: 北京, 2007. C.2 p.67
    
    :param m:
    :param w:
    :return: F_m(w)
    """

    def Fm_w_eq_0(m, w):
        return 1 / (2 * m + 1)

    def Fm_w_gt_0(m, w):
        return jscipy.special.gamma(m + 0.5) * jscipy.special.gammainc(m + 0.5, w) / (2 * lax.pow(w, (m + 0.5)))

    return lax.cond(
        lax.eq(w, 0.),
        Fm_w_eq_0,
        Fm_w_gt_0,
        m,
        w,
    )
