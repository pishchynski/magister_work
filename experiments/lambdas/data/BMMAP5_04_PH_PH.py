import numpy as np


class Bmmap504PhPh:
    def __init__(self, matrS_elem=20):
        # BMMAP
        self.test_matrD_0 = np.array([[-3.708504824279462, 0.0004312214911952863],
                                      [0.0008624429823905726, -0.119017131569899]])

        self.test_matrD = np.array([[3.6653826751599334, 0.04269092762833334],
                                    [0.008624429823905726, 0.10953025876360271]])

        self.test_qs = (0.8, 0.2)
        self.test_ns = (5, 2)

        self.test_N = 10
        self.p_hp = 0.4

        self.priority_part = 0.1

        # Server PH
        self.test_vect_beta = np.array([[1., 0.]])
        self.test_matrS = np.array([[-matrS_elem, matrS_elem],
                                    [0., -matrS_elem]])

        # Timer PH
        self.test_vect_gamma = np.array([[1., 0.]])
        self.test_matrGamma = np.array([[-matrS_elem / 2, matrS_elem / 2],
                                        [0., -matrS_elem / 2]])
