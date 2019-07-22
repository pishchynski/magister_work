import numpy as np


class Bmmap3PoissonPhPh:
    """
        Experiment with BMMAP queries stream with max group of 3
        and c_cor = 0
        with PH serve time
        and PH timer
    """

    def __init__(self, matrS_elem=20):
        # BMMAP
        self.test_matrD_0 = np.array([[-17.274336283185839]])
        self.test_matrD = np.array([[17.274336283185839]])

        self.priority_part = 0.7

        self.test_qs = (0.8, 0.2)
        self.test_ns = (5, 2)

        self.test_N = 10
        self.p_hp = 0.4

        # Server PH
        self.test_vect_beta = np.array([[1., 0.]])
        self.test_matrS = np.array([[-matrS_elem, matrS_elem],
                                    [0., -matrS_elem]])

        # Timer PH
        self.test_vect_gamma = np.array([[1., 0.]])
        self.test_matrGamma = np.array([[-10., 10.],
                                        [0., -10.]])
