import json

import basis_set_exchange as bse

"""
This script defines some methods to get the basis value from 
"""

def get_basis(basis_name, elements_identity, input_format='json', output_format="matrix"):
    '''
    Get basis values from BSE(basis_set_exchange) database.
    :param basis_name: str
        Name of the basis, such as "6-31G", "6-31+g*".
        It is not case sensitive.
        Check `basis_set_exchange.api` for more information.
    :param elements_identity: str or list
        List of elements that you want the basis set for.
    :param input_format: str
        "json" for now.
    :param output_format: str
        "matrix" as default.
        This defines the Output basis set format in the return value of the dict.
    :return: {"element_name":[`object_of_the_basis`],...}
        The value of the dict depends on the parameter `output_format`.
    '''
    if input_format == "json":
        basis_json = bse.get_basis(basis_name, elements_identity, fmt='json', header=False)
        return _resolve_basis_from_json(basis_json, output_format)
    else:
        raise NotImplementedError("Format {} isn't Implemented.".format(input_format))


def _resolve_basis_from_json(basis_json, format="matrix"):
    basis_json = json.loads(basis_json)
    basis_dict = {}
    if format == "matrix":
        # the structure of the basis_json is like:
        # {'electron_shells': [{'angular_momentum': [0],
        #                         'coefficients': [['0.1000000000E+01']],
        #                         'exponents': ['0.1687144782E+00'],
        #                         'function_type': 'gto',
        #                         'region': 'valence'},
        #                        {'angular_momentum': [0],
        #                         'coefficients': [['-0.1193324198E+00',
        #                                           '-0.1608541517E+00',
        #                                           '0.1143456438E+01']],
        #                         'exponents': ['0.7868272350E+01',
        #                                       '0.1881288540E+01',
        #                                       '0.5442492580E+00'],
        #                         'function_type': 'gto',
        #                         'region': 'valence'},
        #                        {'angular_momentum': [0],
        #                         'coefficients': [['0.1834737132E-02',
        #                                           '0.1403732281E-01',
        #                                           '0.6884262226E-01',
        #                                           '0.2321844432E+00',
        #                                           '0.4679413484E+00',
        #                                           '0.3623119853E+00']],
        #                         'exponents': ['0.3047524880E+04',
        #                                       '0.4573695180E+03',
        #                                       '0.1039486850E+03',
        #                                       '0.2921015530E+02',
        #                                       '0.9286662960E+01',
        #                                       '0.3163926960E+01'],
        #                         'function_type': 'gto',
        #                         'region': 'valence'},
        #                        {'angular_momentum': [1],
        #                         'coefficients': [['0.6899906659E-01',
        #                                           '0.3164239610E+00',
        #                                           '0.7443082909E+00']],
        #                         'exponents': ['0.7868272350E+01',
        #                                       '0.1881288540E+01',
        #                                       '0.5442492580E+00'],
        #                         'function_type': 'gto',
        #                         'region': 'valence'},
        #                        {'angular_momentum': [1],
        #                         'coefficients': [['0.1000000000E+01']],
        #                         'exponents': ['0.1687144782E+00'],
        #                         'function_type': 'gto',
        #                         'region': 'valence'}],
        #    'references': [{'reference_description': '6-31G Split-valence basis set',
        #                    'reference_keys': ['hehre1972a']}]}
        
        for element, values in basis_json['elements'].items():
            basis_dict[element] = []
            for shell in values['electron_shells']:
                # get all angular momentum to sperate basis
                for idx, angum in enumerate(shell['angular_momentum']):
                    basis_dict[element].append({
                        'angular_momentum': angum,
                        'exponents': shell['exponents'],
                        'coefficients': shell['coefficients'][idx],
                    })
        return basis_dict
    
    else:
        raise NotImplementedError("Format {} isn't Implemented.".format(format))


if __name__ == '__main__':
    # try:
    #     basis = json.loads(bse.get_basis("cc-pVDz", "H,C,S", fmt='json', header=False))
    #     print(basis)
    #     for ele_key, ele_value in basis['elements'].items():
    #         print(ele_key)
    #         for shell_value in ele_value['electron_shells']:
    #             print("angular_momentum: {}".format(shell_value['angular_momentum']))
    #             print("exponents: {}".format(shell_value['exponents']))
    #             print("coefficients: {}".format(shell_value['coefficients']))
    # except Exception:
    #     raise
    from pprint import pprint
    bs=bse.get_basis("6-31G", "H,C,S", header=False, uncontract_spdf=True)
    pprint(bs)
    # pprint(get_basis("6-31G", "H,C,S"))