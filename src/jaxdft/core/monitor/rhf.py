
from itertools import product
import jax
import numpy as np
import jax.numpy as jnp

from jaxdft.core.basis.atombasis import ANGUL_NAME, AO, L_LIST, L_NAME_DICT
from jaxdft.core.integ.gtomath import norm_gto
from jaxdft.core.integ.overlap import _Sxyz_func, _Sxyz_slow

def e_nn(z_list, pos_list):
    """Calculate the nuclear repulsion energy of a molecule.
    
    Args:
        z_list (list): A list of atomic numbers.
        pos_list (list): A list of atomic positions.
        
    Returns:
        float: The nuclear repulsion energy of the molecule.
    """
    return jnp.sum(
        jnp.triu(
            jnp.outer(
                z_list, 
                z_list
            ), k=1
        ) / jnp.linalg.norm(
            pos_list[:, None, :] - pos_list[None, :, :], axis=-1
        )
    ).sum()

@jax.jit
def coef_norm_func(coef1, coef2):
    return jnp.outer(
        coef1,
        coef2
    )

def Sxyz_func(l1, l2, norm=True):
    # the input a, b are contracted GTOs, so the shape of them are (n, 1)
    # thus the S value is sum of the contracted GTOs
    item_Sxyz_func = _Sxyz_func(
        l1[0],
        l1[1],
        l1[2],
        l2[0],
        l2[1],
        l2[2],
        norm=norm
    )

    item_batch_Sxyz_func = jax.vmap(
        item_Sxyz_func,
        in_axes=(0, 0, None, None)
    )
    
    @jax.jit
    def Sxyz_contraction(a, b, coef1, coef2, pa_xyz, pb_xyz):
        # the input a, b are contracted GTOs, so the shape of them are (n, 1)
        # thus the S value is sum of the contracted GTOs
        a_shape = a.shape
        b_shape = b.shape
        a = jnp.outer(
            a,
            jnp.ones(b_shape)
        ).flatten()
        b = jnp.outer(
            jnp.ones(a_shape),
            b
        ).flatten()

        result = jnp.sum(
            coef_norm_func(
                coef1,
                coef2
            ).flatten() * item_batch_Sxyz_func(
                a,
                b,
                pa_xyz,
                pb_xyz
            )
        )
        return result
    return Sxyz_contraction

def int_overlap_and_kinetic(ao_list):
    
    # establish the list for jax.map
    expon = jnp.array([ao.expon for ao in ao_list])
    coef = jnp.array([ao.coef for ao in ao_list])
    l = [ao.l for ao in ao_list]
    r = jnp.array([ao.r for ao in ao_list])
    
    # calculate the overlap and kinetic matrix
    S_mat = jnp.zeros(len(ao_list) * len(ao_list))
    K_mat = jnp.zeros(len(ao_list) * len(ao_list))
    
    # establish the funcs of overlaps for every orbitals
    combinations_dict = {}
    combined = list(product(l,l))
    for index, combo in enumerate(combined):
        combination = tuple(combo)
        if combination not in combinations_dict:
            combinations_dict[combination] = []
        combinations_dict[combination].append(index)
    
    
    funcs_S = {}
    for k in combinations_dict.keys():
        funcs_S[k] = jax.vmap(
            Sxyz_func(k[0], k[1]),
            in_axes=(0, 0, 0, 0, 0, 0)
        )


    
    for k, v in combinations_dict.items():
        v = jnp.array(v)
        indices1 = v // len(ao_list)
        indices2 = v % len(ao_list)
        
        S_mat = S_mat.at[jnp.array(v).transpose()].set(
            funcs_S[k](
                expon[indices1],
                expon[indices2],
                coef[indices1],
                coef[indices2],
                r[indices1],
                r[indices2]
            )
        )
        # K_mat = K_mat.at[jnp.array(v).transpose()].set(
        #     funcs_S[k](
        #         expon[indices1],
        #         expon[indices2],
        #         coef[indices1],
        #         coef[indices2],
        #         r[indices1],
        #         r[indices2]
        #     )
        # )
    return S_mat.reshape(len(ao_list), len(ao_list))
        #, K_mat.reshape(len(ao_list), len(ao_list))

def coeff_mat(coeff1,coeff2):
    coef1 = np.array([coeff1]) * np.ones([1, len(coeff2)])
    coef2 = np.array([coeff2]) * np.ones([len(coeff1), 1])
    return coef1.transpose() * coef2

def int_overlap_slow(ao_list):
    atom_basis_num = len(ao_list)
    S_mat = np.zeros((atom_basis_num, atom_basis_num))
    for i in range(atom_basis_num):
        for j in range(atom_basis_num):
            Stemp = coeff_mat(ao_list[i].coef, ao_list[j].coef)
            Stemp_2 = np.zeros_like(Stemp)
            a_mat = np.zeros_like(Stemp)
            b_mat = np.zeros_like(Stemp)
            for i1, p1 in enumerate(ao_list[i].expon):
                for i2, p2 in enumerate(ao_list[j].expon):
                    Stemp_2[i1][i2] = (
                        _Sxyz_slow(
                            p1,
                            p2,
                            ao_list[i].r,
                            ao_list[j].r,
                            *ao_list[i].l,
                            *ao_list[j].l,
                        )
                    )
                    a_mat[i1][i2] = p1
                    b_mat[i1][i2] = p2
            Stemp *= Stemp_2
            S_mat[i][j] = Stemp.sum()
            S_mat[j][i] = S_mat[i][j]
    return S_mat

def load_basis(atoms, basis_dict):
    atom_basis_list = []
    for index, elem in enumerate(atoms.get_chemical_symbols()):
        n_Obital = {"s": 0, "p": 0, "d": 0, "f": 0, "g": 0}
        for shell_basis in basis_dict[str(atoms.numbers[index])]:
            angul = shell_basis['angular_momentum']
            for idx_l, l_num in enumerate(L_LIST[ANGUL_NAME[angul]]):
                n_Obital[ANGUL_NAME[angul]] += 1
                atom_basis_list.append(
                    AO(
                        id="{}_{}{}{}".format(
                            "{}{}".format(elem, index + 1),
                            n_Obital[ANGUL_NAME[angul]],
                            ANGUL_NAME[angul],
                            L_NAME_DICT[ANGUL_NAME[angul]][idx_l]
                        ),
                        z=atoms.get_atomic_numbers()[index],
                        r=atoms.get_positions()[index],
                        angum=angul,
                        l=l_num,
                        norm=np.array([1., 1., 1.]),
                        coef=np.array([float(i) for i in shell_basis['coefficients']]),
                        expon=np.array([float(i) for i in shell_basis['exponents']])
                    )
                )
    return atom_basis_list

if __name__ == '__main__':
    # jax.config.update("jax_enable_x64", True)  

    from pprint import pprint
    from ase.atoms import Atoms
    from jaxdft.core.basis.get_basis import get_basis
    NH3 = Atoms(
        numbers = [7, 1, 1, 1],
        positions = [[0.30926, 0.30926, 0.30926],
                         [2.17510, 0.0, 0.0],
                         [0.0, 2.17510, 0.0],
                         [0.0, 0.0, 2.17510]],
    )
    basis_dict = get_basis(
        'sto-3g',
        ['N', 'H']
    )
    # basis_dict = {
    #     '1': [
    #         {'angular_momentum': 0,
    #     'coefficients': ['0.1543E+00',
    #                      '0.5353E+00',
    #                      '0.4446E+00'],
    #     'exponents': ['0.32078E+01',
    #                   '0.5843E+00',
    #                   '0.1581E+00']
    #     }
    #         ],
    #     '7': [
    #         {'angular_momentum': 0,
    #             'coefficients': ['0.1543E+00',
    #                             '0.5353E+00',
    #                             '0.4446E+00'],
    #             'exponents': ['0.999997E+02',
    #                         '0.182151E+02',
    #                         '0.49297E+01']
    #             },
    #         {'angular_momentum': 0,
    #             'coefficients': ['-0.9997E-01',
    #                             '0.3995E+00',
    #                             '0.7001E+00'],
    #             'exponents': ['0.37805E+01',
    #                         '0.8785E+00',
    #                         '0.2857E+00']
    #             },
    #         {'angular_momentum': 1,
    #             'coefficients': ['0.1559E+00',
    #                             '0.6077E+00',
    #                             '0.3920E+00'],
    #             'exponents': ['0.37805E+01',
    #                         '0.8785E+00',
    #                         '0.2857E+00']
    #             }
    #         ]
    #     }
    # pprint(basis_dict)
    ao_list= load_basis(
        NH3,
        basis_dict,
    )
    # pprint(ao_list)
    import time
    # timeit
    S = int_overlap_and_kinetic(ao_list)
    st_time = time.time()
    S = int_overlap_and_kinetic(ao_list)
    end_time = time.time()
    print("jax time:", end_time-st_time)
    st_time = time.time()
    S2 = int_overlap_slow(ao_list)
    end_time = time.time()
    print("python time:", end_time-st_time)
    delta_S = S-S2
    # for line in S:
    #     for item in line:
    #         print("{:10.6f}".format(item), end=" ")
    #     print()
    # print()
    # for line in S2:
    #     for item in line:
    #         print("{:10.6f}".format(item), end=" ")
    #     print()
    # print()
    # for line in delta_S:
    #     for item in line:
    #         print("{:10.6f}".format(item), end=" ")
    #     print()
    # print()
