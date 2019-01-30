import copy

from utils import *


def get_matr_LzNT(matr_T: np.ndarray, N: int):
    matr_L = [0 for _ in range(N)]

    if matr_T.shape[0] != matr_T.shape[1]:
        print("T matrix must be square!")
        return None

    r = matr_T.shape[0]

    for w in range(r - 1):
        matr_Lz = [0 for _ in range(N)]

        for z in range(N):
            if w == 0:
                matr_L[z] = (N - z) * matr_T[r - 1][0] * np.identity(1)
            else:
                m_A = 0  # L_i^w rows num
                n_A = 0  # L_I^w cols num

                for k in range(N - 1, z - 1, -1):
                    m_A += matr_L[k].shape[0]
                    n_A += matr_L[k].shape[1]

                m_A += matr_L[N - 1].shape[1]  # added same num of rows as cols because matrix is square

                temp = np.array([[0 for _ in range(n_A)] for _ in range(m_A)])  # elements of matrix to use positions

                n_pos = 0
                m_pos = 0

                for l in range(N - z):
                    shift = matr_L[N - l - 1].shape[1]

                    copy_matrix_block(temp, (N - z - l) * matr_T[r - 1 - w][0] * np.identity(shift), m_pos, n_pos)

                    m_pos += shift

                    copy_matrix_block(temp, matr_L[N - l - 1], m_pos, n_pos)

                    n_pos += shift

                matr_Lz[z] = temp

        if w != 0:
            for z in range(N):
                matr_L[z] = matr_Lz[z]

    return matr_L


def get_matr_UzNT(matr_T: np.ndarray, N: int):

    if matr_T.shape[0] != matr_T.shape[1]:
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

                for l in range(N - z + 1):
                    shift = matr_U_temp[N - l].shape[0]

                    copy_matrix_block(temp, matr_T[0][r - 1 - w] * np.identity(shift), m_pos, n_pos)

                    n_pos += shift

                    copy_matrix_block(temp, matr_U_temp[N - l], m_pos, n_pos)

                    m_pos += shift

                matr_Uz[z] = temp

        if w != 0:
            for z in range(1, N + 1):
                matr_U_temp[z] = copy.deepcopy(matr_Uz[z])

    for z in range(1, N + 1):
        matr_U[z] = z * matr_U_temp[z]

    return matr_U


def calc_ramaswami_matrices(matr_S: np.ndarray, matr_tilde_S: np.ndarray, vect_beta: np.ndarray, N: int, verbose=False):
    """
    Calculates Ramaswami matrices L_i, A_i and P_i

    :param matr_S: PH matrix
    :param matr_tilde_S: [[0, 0], [S_0, S]] matrix
    :param vect_beta: PH vector
    :param N: number of servers
    :return:
    """
    matr_U = []
    matr_L = []
    matr_A = [np.array([[0]])]  # we start from A_1
    matr_P1 = [0 for _ in range(N)] if N > 0 else [0]
    M = matr_S.shape[0]  # number of phases
    matr_tau = [0]
    if M != 1:
        matr_tau[0] = matr_S

        for k in range(1, M):
            matr_tau.append(matr_S[k:, k:])

        for j in range(M - 1):
            if j != 0:
                matr_U = get_matr_UzNT(matr_tau[M - 2 - j], N)
                matr_L = get_matr_LzNT(matr_tau[M - 2 - j], N)

            matr_Az = [0 for _ in range(N + 1)]

            for m in range(1, N + 1):
                if j == 0:
                    temp = np.array([[0 for _ in range(m + 1)] for _ in range(m + 1)])

                    for l in range(m):
                        temp[l][l + 1] = (m - l) * matr_S[M - 2][M - 1]
                        temp[l + 1][l] = (l + 1) * matr_S[M - 1][M - 2]

                    matr_A.append(temp)
                else:
                    m_A = 0
                    n_A = 0

                    for k in range(N - 1, N - m - 1, -1):
                        m_A += matr_L[k].shape[0]
                        n_A += matr_L[k].shape[1]

                    m_A += matr_U[N].shape[0]
                    n_A += matr_U[N - m + 1].shape[1]

                    temp = np.array([[0 for _ in range(n_A)] for _ in range(m_A)])

                    m_pos = 0
                    n_pos = 0

                    for l in range(m):
                        u = 1.0 * (m - l) / (N - l)

                        copy_matrix_block(temp, u * matr_U[N - l], m_pos, n_pos + matr_L[N - l - 1].shape[1])

                        n_pos += matr_L[N - l - 1].shape[1]

                        copy_matrix_block(temp, matr_L[N - l - 1], m_pos + matr_U[N - l].shape[0], n_pos)

                        m_pos += matr_U[N - l].shape[0]
                        n_pos += matr_L[N - l - 1].shape[1]

                        copy_matrix_block(temp, matr_A[l + 1], m_pos, n_pos)

                    matr_Az[m] = temp

            if j != 0:
                for m in range(1, N + 1):
                    matr_A[m] = matr_Az[m]

        print('Calculating P_1_i(beta)')

        for j in range(M - 1):
            a = [[]]

            if j != 0:
                a = np.array([[0 for _ in range(j + 1)]])

                for k in range(M - j - 1, M):
                    a[0][k - M + j + 1] = vect_beta[0][k]  # beta_g_1 ???

            matr_Pz = [0 for _ in range(N)]

            for m in range(1, N):
                if j == 0:
                    temp = np.array([[0 for _ in range(m + 2)] for _ in range(m + 1)])

                    for l in range(m + 1):
                        temp[l][l + 1] = vect_beta[0][M - 1]
                        temp[l][l] = vect_beta[0][M - 2]

                    matr_P1[m] = temp
                else:
                    m_A = 1
                    n_A = 1

                    for k in range(1, m + 1):
                        m_A += matr_P1[k].shape[0]
                        n_A += matr_P1[k].shape[1]

                    n_A += a[0].shape[0]
                    temp = np.array([[0 for _ in range(n_A)] for _ in range(m_A)])

                    m_pos = 1
                    n_pos = 1

                    temp[0][0] = vect_beta[0][M - j - 2]  # beta1 ???

                    copy_matrix_block(temp, a, 0, 1)

                    for l in range(1, m + 1):
                        shift = matr_P1[l].shape[0]

                        copy_matrix_block(temp, vect_beta[0][M - j - 2] * np.identity(shift), m_pos, n_pos)

                        n_pos += shift

                        copy_matrix_block(temp, matr_P1[l], m_pos, n_pos)

                        m_pos += shift

                    matr_Pz[m] = temp

            if j != 0:
                for m in range(1, N):
                    matr_P1[m] = matr_Pz[m]

        temp = copy.deepcopy(vect_beta)
        matr_P1[0] = temp

        print('Calculating L_i(N, matr_tilde_S)')

        matr_L = get_matr_LzNT(matr_tilde_S, N)
    else:
        matr_A = [0 for _ in range(N + 1)]
        matr_L = [0 for _ in range(N)]
        matr_P1 = [0 for _ in range(N)]

        for n in range(N + 1):
            matr_A[n] = n * np.dot(matr_S, np.identity(1))
            if n != N:
                matr_L[n] = (-N + n) * np.dot(matr_S, np.identity(1))
                matr_P1[n] = np.identity(1)
    return matr_L, matr_A, matr_P1
