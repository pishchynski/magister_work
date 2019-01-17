import numpy as np


# BMMAP
# test_matrD_0 = np.array([[-32.88539908620601]])
# test_matrD = np.array([[32.88539908620601]])

test_matrD_0 = np.array([[-32.]])
test_matrD = np.array([[32.]])
test_q = 0.8
test_n = 1

# Server PH
test_vect_beta = np.array([[1., 0.]])
test_matrS = np.array([[-20., 20.],
                       [0., -20.]])

# Timer PH
test_vect_gamma = np.array([[1., 0.]])
test_matrGamma = np.array([[-(10. ** 5) * 2, (10. ** 5) * 2],
                           [0., -(10. ** 5) * 2]])
