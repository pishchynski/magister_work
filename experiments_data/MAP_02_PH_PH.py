import numpy as np


class Map02PhPh:
    def __init__(self):
        # BMMAP
        self.test_matrD_0 = np.array([[-5.408, 0.],
                                      [0., -0.1755]]) * 8.00137983712

        self.test_matrD = np.array([[5.372, 0.036],
                                    [0.09772, 0.07778]]) * 8.00137983712

        self.priority_part = 0.99999999999

        self.test_qs = (0.8, 0.8)
        self.test_ns = (1, 1)

        self.test_N = 5
        self.p_hp = 0.4

        # Server PH
        self.test_vect_beta = np.array([[1., 0.]])
        self.test_matrS = np.array([[-20., 20.],
                                    [0., -20.]])

        # Timer PH
        self.test_vect_gamma = np.array([[1., 0.]])
        self.test_matrGamma = np.array([[-10., 10.],
                                        [0., -10.]])
