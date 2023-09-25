ANGUL_NAME = ['s', 'p', 'd', 'f', 'g', 'h', 'i']
L_LIST = {"s": [(0, 0, 0)  # s
                   ],
             "p": [(1, 0, 0),  # px
                   (0, 1, 0),  # py
                   (0, 0, 1)  # pz
                   ],
             "d": [(2, 0, 0),  # dx^2
                   (0, 2, 0),  # dy^2
                   (0, 0, 2),  # dz^2
                   (1, 0, 1),  # dxz
                   (0, 1, 1),  # dyz
                   (1, 1, 0)  # dxy
                   ],
             "f": [(3, 0, 0),  # fx^3
                   (0, 3, 0),  # fy^3
                   (0, 0, 3),  # fz^3
                   (2, 1, 0),  # fx^2y
                   (2, 0, 1),  # fx^2z
                   (1, 2, 0),  # fxy^2
                   (1, 0, 2),  # fxz^2
                   (0, 2, 1),  # fy^2z
                   (0, 1, 2),  # fyz^2
                   (1, 1, 1)  # fxyz
                   ]
             }
L_NAME_DICT = {"s": [""  # s
                       ],
                 "p": ["x",  # px
                       "y",  # py
                       "z"  # pz
                       ],
                 "d": ["x^2",  # dx^2
                       "y^2",  # dy^2
                       "z^2",  # dz^2
                       "xz",  # dxz
                       "yz",  # dyz
                       "xy"  # dxy
                       ],
                 "f": ["x^3",  # fx^3
                       "y^3",  # fy^3
                       "z^3",  # fz^3
                       "x^2y",  # fx^2y
                       "x^2z",  # fx^2z
                       "xy^2",  # fxy^2
                       "xz^2",  # fxz^2
                       "y^2z",  # fy^2z
                       "yz^2",  # fyz^2
                       "xyz"  # fxyz
                       ]
                 }

class AO(object):

    def __init__(self, id, z, r, angum, l, coef, norm, expon):
        """Atomic orbital class of a single primitive Gaussian function.

        :param id:
        :param z: atomic number
        :param r: position of the center of the atomic orbital
        :param angum: angular momentum
        :param l: angular momentum
        :param coef: coefficient
        :param norm: normalization factor
        :param expon: exponent of the Gaussian function
        """
        self.id = id
        self.z = z
        self.r = r
        self.angum = angum
        self.l = l
        self.norm = norm
        self.coef = coef
        self.expon = expon

    def get_lmn(self):
        if self.angum == 0:
            return (0, 0, 0)
        else:
            return self.l
      
    def __str__(self) -> str:
        return "AO(id={}, z={}, r={}, angum={}, l={}, coef={}, norm={}, expon={})\n".format(
            self.id, self.z, self.r, self.angum, self.l, self.coef, self.norm, self.expon
        )
    
    def __repr__(self) -> str:
        return self.__str__()