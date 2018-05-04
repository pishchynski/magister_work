import gc
import sys

sys.path.append("../")
from streams import *
from test_data import *
from ramaswami import calc_ramaswami_matrices

np.set_printoptions(threshold=np.inf, suppress=True, formatter={'float': '{: 0.8f}'.format}, linewidth=75)


class TwoPrioritiesQueueingSystem:
    """
    Class describing Heterogeneous Reliable Queueing System with Finite Buffer and
    Two priorities of queries.
    In Kendall designation (informal): BMMAP|PH|TimerPH|N - unchecked(?)
    """

    def __init__(self, name='Default system', p_max_num=100):
        self.name = name
        self._p_max_num = p_max_num

        self._eps_G = 10 ** (-8)
        self._eps_F = 10 ** (-8)

        self.queries_stream = BMMAPStream(test_matrD_0, test_matrD)
        self.serv_stream = PHStream(test_vect_beta, test_matrS)
        self.timer_stream = PHStream(test_vect_gamma, test_matrGamma)
        self.matr_hat_Gamma = np.bmat([[np.zeros(1, self.timer_stream.dim), np.zeros(1, self.timer_stream.dim)],
                                       [self.timer_stream.repres_matr_0, self.timer_stream.repres_matr]])
        self.n = 3
        self.N = 3
        self.ramatrL, self.ramatrA, self.ramatrP = self._calc_ramaswami_matrices(0, self.N)

    def set_BMMAP_queries_stream(self, matrD_0, matrD, q=0.8, n=3):
        self.queries_stream = BMAPStream(matrD_0, matrD, q, n)
        self.n = n

    def set_PH_serv_stream(self, vect, matr):
        self.serv_stream = PHStream(vect, matr)

    def set_PH_timer_stream(self, vect, matr):
        self.timer_stream = PHStream(vect, matr)

    def _calc_ramaswami_matrices(self, start=0, end=None):
        if not end:
            end = self.N

        matrL = []
        matrA = []
        matrP = []

        for i in range(0, end + 1):
            tempL, tempA, tempP = calc_ramaswami_matrices(self.timer_stream.repres_matr,
                                                          self.matr_hat_Gamma,
                                                          self.timer_stream.repres_vect,
                                                          i)
            matrL.append(tempL)
            matrA.append(tempA)
            matrP.append(tempP)

        return matrL, matrA, matrP

    def _calc_Q_00(self):
        block00 = copy.deepcopy(self.queries_stream.matrD_0)
        block01 = kron(self.queries_stream.transition_matrices[0][0] + self.queries_stream.transition_matrices[0][1],
                       self.serv_stream.repres_vect)
        block10 = kron(np.eye(self.queries_stream.dim_),
                       self.serv_stream.repres_matr_0)
        block11 = kronsum(self.queries_stream.matrD_0,
                          self.serv_stream.repres_matr)

        matrQ_00 = np.bmat([[block00, block01],
                            [block10, block11]])

        return np.array(matrQ_00)

    def _calc_Q_0k(self):
        matrQ_0k = [self._calc_Q_00()]

        for k in range(1, self.N - 1):
            block00 = np.zeros(self.queries_stream.transition_matrices[0][0].shape)
            if k + 1 > self.n:
                block00 = kron(np.zeros(self.queries_stream.transition_matrices[0][0].shape),
                               self.serv_stream.repres_vect)
            else:
                block00 = kron(self.queries_stream.transition_matrices[0][k + 1],
                               self.serv_stream.repres_vect)

            block10 = np.zeros(self.queries_stream.transition_matrices[0][0].shape)
            if k > self.n:
                block10 = kron(np.zeros(self.queries_stream.transition_matrices[0][0].shape),
                               np.eye(self.serv_stream.dim))
            else:
                block10 = kron(self.queries_stream.transition_matrices[0][k],
                               np.eye(self.serv_stream.dim))

            blocks0k = [block00]
            blocks1k = [block10]

            for j in range(1, k - 1):
                temp_block = np.zeros((self.queries_stream.dim_,
                                        self.queries_stream.dim * self.serv_stream.dim * sum(
                                            [ncr(self.timer_stream.dim, j + self.timer_stream.dim - 1)
                                             for j in range(1, k)]
                                        )))
                blocks0k.append(temp_block)

                temp_block = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                                       self.queries_stream.dim * self.serv_stream.dim * sum(
                                           [ncr(self.timer_stream.dim, j + self.timer_stream.dim - 1)
                                            for j in range(1, k)]
                                       )))
                blocks1k.append(temp_block)

            ramatrP_mul = copy.deepcopy(self.ramatrP[0][0])
            for i in range(2, self.N):
                ramatrP_mul = np.dot(ramatrP_mul, self.ramatrP[0][i])

            last_block0 = np.zeros((1, 1))
            if k + 1 > self.n:
                last_block0 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][0].shape),
                                                 self.serv_stream.repres_vect),
                                        ramatrP_mul)
            else:
                last_block0 = kron(kron(self.queries_stream.transition_matrices[1][k + 1],
                                        self.serv_stream.repres_vect),
                                   ramatrP_mul)

            last_block1 = np.zeros((1, 1))
            if k > self.n:
                last_block1 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][0].shape),
                                        np.eye(self.serv_stream.dim)),
                                   ramatrP_mul)
            else:
                last_block1 = kron(kron(self.queries_stream.transition_matrices[1][k],
                                        np.eye(self.serv_stream.dim)),
                                   ramatrP_mul)

            blocks0k.append(last_block0)
            blocks1k.append(last_block1)

            temp_matr = np.bmat([blocks0k,
                                 blocks1k])

            matrQ_0k.append(temp_matr)
            matrQ_0k.append(self._calc_Q_0N())

    def _calc_Q_0N(self):
        block00 = np.zeros((1, 1))
        if self.N + 1 > self.n:
            block00 = kron(np.zeros(self.queries_stream.transition_matrices[0][0].shape),
                           self.serv_stream.repres_vect)
        else:
            block00 = kron(self.queries_stream.transition_matrices[0][self.N + 1],
                        self.serv_stream.repres_vect)
            for i in range(self.N + 2, self.n + 1):
                block00 += kron(self.queries_stream.transition_matrices[0][i],
                                self.serv_stream.repres_vect)

        block10 = np.zeros((1, 1))
        if self.N > self.n:
            block10 = kron(np.zeros(self.queries_stream.transition_matrices[0][0].shape),
                           self.serv_stream.repres_vect)
        else:
            block10 = kron(self.queries_stream.transition_matrices[0][self.N],
                           np.eye(self.serv_stream.dim))
            for i in range(self.N + 1, self.n + 1):
                block10 += kron(self.queries_stream.transition_matrices[0][i],
                                np.eye(self.serv_stream.dim))

        blocks0k = [block00]
        blocks1k = [block10]

        for j in range(1, self.N - 1):
            temp_block = np.zeros((self.queries_stream.dim_,
                                   self.queries_stream.dim * self.serv_stream.dim * sum(
                                       [ncr(self.timer_stream.dim, j + self.timer_stream.dim - 1)
                                        for j in range(1, self.N)]
                                   )))
            blocks0k.append(temp_block)

            temp_block = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                                   self.queries_stream.dim * self.serv_stream.dim * sum(
                                       [ncr(self.timer_stream.dim, j + self.timer_stream.dim - 1)
                                        for j in range(1, self.N)]
                                   )))
            blocks1k.append(temp_block)

        ramatrP_mul = copy.deepcopy(self.ramatrP[0][0])
        for i in range(2, self.N):
            ramatrP_mul = np.dot(ramatrP_mul, self.ramatrP[0][i])

        last_block0 = np.zeros((1, 1))
        if self.N + 1 > self.n:
            last_block0 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][0].shape),
                                             self.serv_stream.repres_vect),
                                    ramatrP_mul)
        else:
            last_block0 = kron(kron(self.queries_stream.transition_matrices[1][self.N + 1],
                                    self.serv_stream.repres_vect),
                               ramatrP_mul)
            for i in range(self.N + 2, self.n + 1):
                last_block0 += kron(kron(self.queries_stream.transition_matrices[1][i],
                                         self.serv_stream.repres_vect),
                                    ramatrP_mul)

        last_block1 = np.zeros((1, 1))
        if self.N > self.n:
            last_block1 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][0].shape),
                                    np.eye(self.serv_stream.dim)),
                               ramatrP_mul)
        else:
            last_block1 = kron(kron(self.queries_stream.transition_matrices[1][self.N],
                                    np.eye(self.serv_stream.dim)),
                               ramatrP_mul)
            for i in range(self.N + 1, self.n + 1):
                last_block1 += kron(kron(self.queries_stream.transition_matrices[1][i],
                                         np.eye(self.serv_stream.dim)),
                                    ramatrP_mul)

        blocks0k.append(last_block0)
        blocks1k.append(last_block1)

        return np.bmat([blocks0k,
                        blocks1k])

