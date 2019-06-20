import numpy as np

# BMMAP
test_matrD_0 = np.array([[-8.22458, 8.22458 * 10 ** (-6)],
                         [8.22458 * 10 ** (-6),  -0.180152]]) * 4.57012244894

test_matrD = np.array([[8.199422, 0.02515],
                       [0.140373, 0.039771]]) * 4.57012244894

test_q = 0.8
test_n = 1

# Server PH
test_vect_beta = np.array([[1.]])
test_matrS = np.array([[-10.]])

# Timer PH
test_vect_gamma = np.array([[1.]])
test_matrGamma = np.array([[-10 ** 5]])
