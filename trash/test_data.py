import numpy as np

# BMMAP
test_matrD_0 = np.array([[-86, 0.01],
                         [0.02, -2.76]])  # * 1.85245901639

test_matrD = np.array([[85, 0.99],
                       [0.2, 2.54]])  # * 1.85245901639

test_q = 0.8
test_n = 1

# Server PH
test_vect_beta = np.array([[1., 0.]])
test_matrS = np.array([[-20., 20.],
                       [0., -20.]])

# Timer PH
test_vect_gamma = np.array([[1., 0.]])
test_matrGamma = np.array([[-(10. ** 5)*2, (10. ** 5)*2],
                           [0., -(10. ** 5)*2]])
