from ramaswami import *


if __name__ == '__main__':
    N = 5
    matr_S = np.array([[-8, 5],
                   [1, -2]])

    vect_beta = np.array([0.98, 0.02])

    matr_S0 = r_multiply_e(-matr_S)

    matr_tilde_S = np.block([
        [np.zeros((1, matr_S0.shape[1])), np.zeros((1, matr_S.shape[1]))],
        [matr_S0, matr_S]
    ])

    matr_L, matr_A, matr_P = calc_ramaswami_matrices(matr_S, matr_tilde_S, vect_beta, N)
    matr_L1, matr_A1, matr_P1 = calc_ramaswami_matrices(matr_S, matr_tilde_S, vect_beta, N - 1)
    print()