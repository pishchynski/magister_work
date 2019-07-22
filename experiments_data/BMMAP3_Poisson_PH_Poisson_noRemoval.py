import numpy as np


class Bmmap3PoissonPhPoissonNoRemoval:
    """
        Experiment with BMMAP queries stream with max group of 3
        and c_cor = 0
        with PH serve time
        and Poisson timer
        with removal probability = 10 ** (-5)
    """

    def __init__(self):
        # BMMAP
        self.test_matrD_0 = np.array([[-17.2743362832]])
        self.test_matrD = np.array([[17.2743362832]])

        self.test_qs = (0.8, 0.8)
        self.test_ns = (3, 3)

        self.test_N = 5
        self.p_hp = 10. ** (-5)

        # Server PH
        self.test_vect_beta = np.array([[1., 0.]])
        self.test_matrS = np.array([[-20., 20.],
                                    [0., -20.]])

        # Timer PH
        self.test_vect_gamma = np.array([[1.]])
        self.test_matrGamma = np.array([[10. ** (-5)]])
