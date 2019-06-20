import numpy as np

# BMMAP
test_matrD_0 = np.array([[-5.408, 0.],
                         [0., -0.1755]]) * 4.319328938619707

test_matrD = np.array([[5.372, 0.036],
                       [0.09772, 0.07778]]) * 4.319328938619707

test_q = 0.8
test_n = 3

test_N = 5
test_p_hp = 0.4

# Server PH
test_vect_beta = np.array([[1., 0.]])
test_matrS = np.array([[-20., 20.],
                       [0., -20.]])

# Timer PH
test_vect_gamma = np.array([[1., 0.]])
test_matrGamma = np.array([[-10., 10.],
                           [0., -10.]])
