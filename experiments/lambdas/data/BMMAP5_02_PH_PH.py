import numpy as np


class Bmmap502PhPh:
    def __init__(self, matrS_elem=20):
        # BMMAP
        self.test_matrD_0 = np.array([[-1.0351576533073386, 0.],
                                      [0., -0.03359285653761795]])

        self.test_matrD = np.array([[1.0282668109406474, 0.006890842366690861],
                                    [0.018704808779806417, 0.014888047757811535]])

        self.test_qs = (0.8, 0.2)
        self.test_ns = (5, 2)

        self.priority_part = 0.1

        self.test_N = 10
        self.p_hp = 0.4

        # Server PH
        self.test_vect_beta = np.array([[1., 0.]])
        self.test_matrS = np.array([[-matrS_elem, matrS_elem],
                                    [0., -matrS_elem]])

        # Timer PH
        self.test_vect_gamma = np.array([[1., 0.]])
        self.test_matrGamma = np.array([[-matrS_elem / 2, matrS_elem / 2],
                                        [0., -matrS_elem / 2]])
