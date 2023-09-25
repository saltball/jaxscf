import numpy as np
import pytest
import jax
from jaxdft.core.integ.overlap import _Sxyz_func, _Sxyz_slow, _i_defold_func, _sij_slow


jax.config.update('jax_platform_name', 'cpu')


@pytest.mark.parametrize("a", [0.1, 5.0])
@pytest.mark.parametrize("b", [0.1, 5.0])
@pytest.mark.parametrize("pa_x", [5.0])
@pytest.mark.parametrize("pb_x", [0.1])
@pytest.mark.parametrize("la", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("lb", [0, 1, 2, 3])
def test_overlap_i(a, b, pa_x, pb_x, la, lb):
    slow_result = _sij_slow(a, b, pa_x, pb_x, la, lb)
    i_func = _i_defold_func(la, lb)
    i_result = i_func(a, b, pa_x, pb_x)
    if np.allclose(slow_result, i_result):
        pass
    else:
        raise ValueError("Not equal, slow_result = {}, i_result = {}".format(slow_result, i_result))
        
@pytest.mark.parametrize("a", [1.0, 5.0])
@pytest.mark.parametrize("b", [1.0, 5.0])
@pytest.mark.parametrize(
    "pa_xyz", [
        np.array([0., 0., 0.]),
        np.array([0.1, 0.1, 0.1]),
    ]
)
@pytest.mark.parametrize(
    "pb_xyz", [
        np.array([1., 0., 0.]),
        np.array([0., 1., 0.]),
        np.array([0., 0., 1.]),
    ]
)
@pytest.mark.parametrize("la", [-1, 1])
@pytest.mark.parametrize("lb", [0, 1])
@pytest.mark.parametrize("na", [0, 1])
@pytest.mark.parametrize("nb", [0, 1])
@pytest.mark.parametrize("ma", [0, 1])
@pytest.mark.parametrize("mb", [0, 1])
@pytest.mark.parametrize("norm", [True, False])
def test_overlap_S(a, b, pa_xyz, pb_xyz, la, lb, na, nb, ma, mb, norm):
    slow_result = _Sxyz_slow(a, b, pa_xyz, pb_xyz, la, ma, na, lb, mb, nb, norm)
    S_func = _Sxyz_func(la, ma, na, lb, mb, nb, norm)
    S_result = S_func(a, b, pa_xyz, pb_xyz)
    if np.allclose(slow_result, S_result):
        pass
    else:
        raise ValueError("Not equal, slow_result = {}, S_result = {}".format(slow_result, S_result))
