from utils import *


def get_matr_LzNT(matr_T: np.ndarray, N: int):
    matr_L = [0 for _ in range(N)]
    if (matr_T.shape[0] != matr_T.shape[1]):
        print("T matrix must be square!")
        return None
    r = matr_T.shape[0]

    for w in range(r - 1):
        matr_Lz = [0 for _ in range(N)]
        for z in range(N):
            if w == 0:
                matr_L[z] = (N - z) * matr_T[r - 1][0] * np.identity(1)
            else:
                m_A = 0
                n_A = 0
                for k in range(N - 1, z - 1, -1):
                    m_A += matr_L[k].shape[0]
                    n_A += matr_L[k].shape[1]

                m_A += matr_L[N - 1].shape[1]
                temp = np.array([[0 for _ in range(n_A)] for _ in range(m_A)])
                n_pos = 0
                m_pos = 0

                for l in range(N - z):
                    shift = matr_L[N - l - 1].shape[1]
                    copy_matrix_block(temp, (N - z - 1) * matr_T[r - 1 - w][0] * np.identity(shift), m_pos, n_pos)
                    m_pos += shift
                    copy_matrix_block(temp, matr_L[N - l - 1], m_pos, n_pos)
                    n_pos += shift
                matr_Lz[z] = temp

        if w != 0:
            for z in range(N):
                matr_L[z] = matr_Lz[z]

    return matr_L


def get_matr_UzNT(matr_T: np.ndarray, N: int):
    if (matr_T.shape[0] != matr_T.shape[1]):
        print("T matrix must be square!")
        return None
    matr_U = [0 for _ in range(N + 1)]
    matr_U_temp = [0 for _ in range(N + 1)]

    r = matr_T.shape[0]
    for w in range(r - 1):
        matr_Uz = [0 for _ in range(N + 1)]

        for z in range(1, N + 1):
            if w == 0:
                matr_U_temp[z] = matr_T[0][r - 1] * np.identity(1)
            else:
                m_A = 0
                n_A = 0

                for k in range(N, z - 1, -1):
                    m_A += matr_U_temp[k].shape[0]
                    n_A += matr_U_temp[k].shape[1]

                n_A += matr_U_temp[N].shape[0]

                temp = [[0 for _ in range(n_A)] for _ in range(m_A)]
                n_pos = 0
                m_pos = 0

                for l in range(N - z):
                    shift = matr_U_temp[N - 1].shape[0]
                    copy_matrix_block(temp, matr_T[0][r - 1 - w] * np.identity(shift), m_pos, n_pos)
                    n_pos += shift
                    copy_matrix_block(temp, matr_U_temp[N - 1], m_pos, n_pos)
                    m_pos += shift

                matr_Uz[z] = temp

        if w != 0:
            for z in range(1, N + 1):
                matr_U_temp[z] = matr_Uz[z]

    for z in range(1, N + 1):
        matr_U[z] = z * matr_U_temp[z]

    return matr_U


def calc_ramaswamy_matrices(matrS: np.ndarray, vect_beta: np.ndarray, N: int):
    matr_tau = []
    matr_T = []
    M = matrS.shape[0]
    for k in range(M):
        matr_tau.append(matrS[k:, k:])
    for j in range(M - 1):
        matr_T.append(matr_tau[M - 2 - j])

    matr_LzNT = []



    