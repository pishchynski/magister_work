import gc
import sys

from matr_E import get_matr_E

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
        self.matr_hat_Gamma = np.array(np.bmat([[np.zeros((1, self.timer_stream.repres_matr_0.shape[1])),
                                        np.zeros((1, self.timer_stream.repres_matr.shape[1]))],
                                       [self.timer_stream.repres_matr_0, self.timer_stream.repres_matr]]))
        self.I_WM = np.eye(self.queries_stream.dim_ * self.serv_stream.dim)
        self.I_W = np.eye(self.queries_stream.dim_)
        self.S_0xBeta = np.dot(self.serv_stream.repres_matr_0,
                               self.serv_stream.repres_vect)

        self.p_hp = 0.5
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
        block01 = kron(self.queries_stream.transition_matrices[0][1] + self.queries_stream.transition_matrices[1][1],
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

        for k in range(1, self.N):
            block00 = np.zeros(self.queries_stream.transition_matrices[0][1].shape)
            if k + 1 > self.n:
                block00 = kron(np.zeros(self.queries_stream.transition_matrices[0][1].shape),
                               self.serv_stream.repres_vect)
            else:
                block00 = kron(self.queries_stream.transition_matrices[0][k + 1],
                               self.serv_stream.repres_vect)

            block10 = np.zeros(self.queries_stream.transition_matrices[0][1].shape)
            if k > self.n:
                block10 = kron(np.zeros(self.queries_stream.transition_matrices[0][1].shape),
                               np.eye(self.serv_stream.dim))
            else:
                block10 = kron(self.queries_stream.transition_matrices[0][k],
                               np.eye(self.serv_stream.dim))

            blocks0k = [block00]
            blocks1k = [block10]

            for j in range(1, k):
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
            for i in range(2, k):
                ramatrP_mul = np.dot(ramatrP_mul, self.ramatrP[-1][i])

            last_block0 = np.zeros((1, 1))
            if k + 1 > self.n:
                last_block0 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][1].shape),
                                                 self.serv_stream.repres_vect),
                                        ramatrP_mul)
            else:
                last_block0 = kron(kron(self.queries_stream.transition_matrices[1][k + 1],
                                        self.serv_stream.repres_vect),
                                   ramatrP_mul)

            last_block1 = np.zeros((1, 1))
            if k > self.n:
                last_block1 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][1].shape),
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
        return matrQ_0k

    def _calc_Q_0N(self):
        block00 = np.zeros((1, 1))
        if self.N + 1 > self.n:
            block00 = kron(np.zeros(self.queries_stream.transition_matrices[0][1].shape),
                           self.serv_stream.repres_vect)
        else:
            block00 = kron(self.queries_stream.transition_matrices[0][self.N + 1],
                        self.serv_stream.repres_vect)
            for i in range(self.N + 2, self.n + 1):
                block00 += kron(self.queries_stream.transition_matrices[0][i],
                                self.serv_stream.repres_vect)

        block10 = np.zeros((1, 1))
        if self.N > self.n:
            block10 = kron(np.zeros(self.queries_stream.transition_matrices[0][1].shape),
                           self.serv_stream.repres_vect)
        else:
            block10 = kron(self.queries_stream.transition_matrices[0][self.N],
                           np.eye(self.serv_stream.dim))
            for i in range(self.N + 1, self.n + 1):
                block10 += kron(self.queries_stream.transition_matrices[0][i],
                                np.eye(self.serv_stream.dim))

        blocks0k = [block00]
        blocks1k = [block10]

        for j in range(1, self.N):
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

        ramatrP_mul = copy.deepcopy(self.ramatrP[self.N][0])
        for i in range(1, self.N):
            ramatrP_mul = np.dot(ramatrP_mul, self.ramatrP[self.N][i])

        last_block0 = np.zeros((1, 1))
        if self.N + 1 > self.n:
            last_block0 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][1].shape),
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
            last_block1 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][1].shape),
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

    def _calc_Q_10(self):
        block00 = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                            self.queries_stream.dim_))
        block01 = kron(np.eye(self.queries_stream.dim_),
                       np.dot(self.serv_stream.repres_matr_0,
                              self.serv_stream.repres_vect))
        block10 = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim * self.timer_stream.dim,
                            self.queries_stream.dim_))
        block11 = kron(kron(np.eye(self.queries_stream.dim_),
                            np.dot(self.serv_stream.repres_matr_0,
                                   self.serv_stream.repres_vect)),
                       e_col(self.timer_stream.dim))
        block11 += self.p_hp * kron(np.eye(self.queries_stream.dim_ * self.serv_stream.dim),
                                    self.timer_stream.repres_matr_0)
        return np.bmat([[block00, block01],
                        [block10, block11]])

    def _calc_Q_iiprev(self):
        matrQ_iiprev = []
        for i in range(2, self.N + 1):
            blocks0 = [kron(self.I_W,
                            np.dot(self.serv_stream.repres_matr_0,
                                   self.serv_stream.repres_vect))]
            blocks1 = [kron(self.p_hp * self.I_WM,
                            self.ramatrL[self.N - i + 1][self.N - i])]
            for j in range(1, i - 1):
                blocks0.append(kron(kron(self.I_W,
                                         self.S_0xBeta),
                                    np.eye(ncr(j + self.timer_stream.dim - 1,
                                               self.timer_stream.dim - 1))))
                blocks1.append(kron(self.p_hp * self.I_WM,
                                    self.ramatrL[self.N - i + j + 1][self.N - i]))

            blocks0.append(kron(kron(self.I_W,
                                     self.S_0xBeta),
                                np.eye(ncr(i - 1 + self.timer_stream.dim - 1,
                                           self.timer_stream.dim - 1))))
            last_block1 = kron(kron(self.I_W, self.S_0xBeta),
                               (1 / i) * get_matr_E(self.timer_stream.dim,
                                                    i))
            last_block1 += kron(self.p_hp * self.I_WM,
                                self.ramatrL[self.N][self.N - i])
            blocks1.append(last_block1)

            # Form transposed Quasi-Toeplitz matrix

            temp1 = la.block_diag(*tuple(blocks0))
            temp1 = np.vstack([temp1, np.zeros((last_block1.shape[0], temp1.shape[1]))])

            temp2 = la.block_diag(*tuple(blocks1))
            zero_line2 = np.zeros((blocks0[0].shape[0], temp2.shape[1]))
            temp2 = np.vstack([zero_line2, temp2])

            matrQ_iiprev.append(temp1 + temp2)

        return matrQ_iiprev

    def _calc_Q_ii(self):
        matrQ_ii = []
        for i in range(1, self.N + 1):
            cur_matr = la.block_diag(*(kron(kronsum(self.queries_stream.matrD_0 if i != self.N else self.queries_stream.matrD_1_,
                                                     self.serv_stream.repres_matr),
                                             np.eye(ncr(j + self.timer_stream.dim - 1,
                                                        self.timer_stream.dim - 1))) for j in range(i + 1)))
            second_term_blocks = tuple([kron(self.I_WM, self.ramatrA[self.N - i + j][j]) for j in range(i + 1)])
            second_term_shapes = [block.shape for block in second_term_blocks]
            cur_matr += la.block_diag(*second_term_blocks)

            udiag_matr = np.zeros(cur_matr.shape)
            udiag_blocks = [kron(self.I_WM, self.ramatrL[self.N - i + j][self.N - i]) for j in range(1, i + 1)]

            n_pos = 0
            m_pos = 0
            for udiag_block, shape in zip(udiag_blocks, second_term_shapes):
                m_pos += shape[0]
                copy_matrix_block(udiag_matr, udiag_block, m_pos, n_pos)
                n_pos += shape[1]

            cur_matr += (1 - self.p_hp) * udiag_matr

            delta = np.diag([-np.sum(row) for row in cur_matr])

            cur_matr += delta

            matrQ_ii.append(cur_matr)

        return matrQ_ii

    def __get_ramatrP_mul(self, j, k):
        ramatrP_mul = self.ramatrP[-1][j]
        for i in range(j + 1, j + k):
            ramatrP_mul = np.dot(ramatrP_mul, self.ramatrP[-1][i])

        return ramatrP_mul

    def _calc_matrQ_iik(self):
        matrQ_iik = []
        for i in range(1, self.N):
            matrQ_ii_row = []
            for k in range(1, self.N - i):
                cur_matr = la.block_diag(*(kron(self.queries_stream.transition_matrices[0][k],
                                                np.eye(self.serv_stream.dim * ncr(j + self.timer_stream.dim - 1,
                                                                                  self.timer_stream.dim - 1)))
                                         for j in range(i + 1)))
                zero_matr = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim * np.sum(
                                        [ncr(j + self.timer_stream.dim - 1,
                                             self.timer_stream.dim - 1)
                                         for j in range(i + 1)]),
                                      self.queries_stream.dim_ * self.serv_stream.dim * np.sum(
                                          [ncr(j + self.timer_stream.dim - 1,
                                               self.timer_stream.dim - 1)
                                           for j in range(i + 1, i + k + 1)])
                                      ))
                cur_matr = np.concatenate((cur_matr, zero_matr), axis=1)

                temp_matr = la.block_diag(*(kron(kron(self.queries_stream.transition_matrices[1][k],
                                                      np.eye(self.serv_stream.dim)),
                                                 self.__get_ramatrP_mul(j, k))
                                          for j in range(i + 1)))

                zero_matr = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim * np.sum(
                                        [ncr(j + self.timer_stream.dim - 1,
                                             self.timer_stream.dim - 1)
                                         for j in range(i + 1)]),
                                      self.queries_stream.dim_ * self.serv_stream.dim * np.sum(
                                          [ncr(j + self.timer_stream.dim - 1,
                                               self.timer_stream.dim - 1)
                                           for j in range(k)])
                                      ))
                cur_matr += np.concatenate((zero_matr, temp_matr), axis=1)

                matrQ_ii_row.append(cur_matr)

            matrQ_iik.append(matrQ_ii_row)

        return matrQ_iik

    def _calc_matrQ_iN(self):
        matrQ_iN = []
        for i in range(1, self.N):
            matrD1_sum = self.queries_stream.transition_matrices[0][self.N - i]
            matrD2_sum = self.queries_stream.transition_matrices[1][self.N - i]
            for k in range(self.N - i + 1, self.n + 1):
                matrD1_sum += self.queries_stream.transition_matrices[0][k]
                matrD2_sum += self.queries_stream.transition_matrices[1][k]

            cur_matr = la.block_diag(*(kron(matrD1_sum, np.eye(self.serv_stream.dim * ncr(j + self.timer_stream.dim - 1,
                                                                                          self.timer_stream.dim - 1)))
                                       for j in range(i + 1)))

            zero_matr = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim * np.sum(
                [ncr(j + self.timer_stream.dim - 1,
                     self.timer_stream.dim - 1)
                 for j in range(i + 1)]),
                                  self.queries_stream.dim_ * self.serv_stream.dim * np.sum(
                                      [ncr(j + self.timer_stream.dim - 1,
                                           self.timer_stream.dim - 1)
                                       for j in range(i + 1, self.N + 1)])
                                  ))
            cur_matr = np.concatenate((cur_matr, zero_matr), axis=1)

            temp_matr = la.block_diag(*(kron(kron(matrD2_sum, np.eye(self.serv_stream.dim)),
                                             self.__get_ramatrP_mul(j, self.N - i))
                                        for j in range(i + 1)))

            zero_matr = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim * np.sum(
                                      [ncr(j + self.timer_stream.dim - 1,
                                         self.timer_stream.dim - 1)
                                       for j in range(i + 1)]),
                                  self.queries_stream.dim_ * self.serv_stream.dim * np.sum(
                                      [ncr(j + self.timer_stream.dim - 1,
                                           self.timer_stream.dim - 1)
                                       for j in range(self.N - i)])
                                  ))

            cur_matr += np.concatenate((zero_matr, temp_matr), axis=1)

            matrQ_iN.append(cur_matr)
        return matrQ_iN

    def calc_characteristics(self, verbose=False):
        if verbose:
            print('======= Input BMMAP Parameters =======')
            self.queries_stream.print_characteristics('D')

            print('======= PH service time parameters =======')
            self.serv_stream.print_characteristics('S', 'beta')

            print('======= PH timer parameters =======')
            self.timer_stream.print_characteristics('Г', 'gamma')

        matrQ_0k = self._calc_Q_0k()
        matrQ_10 = self._calc_Q_10()
        matrQ_iiprev = self._calc_Q_iiprev()
        matrQ_ii = self._calc_Q_ii()
        matrQ_iik = self._calc_matrQ_iik()
        matrQ_iN = self._calc_matrQ_iN()

        dd = [matrQ_0k, matrQ_10, matrQ_iiprev, matrQ_ii, matrQ_iik, matrQ_iN]


if __name__ == '__main__':
    qs = TwoPrioritiesQueueingSystem()
    qs.calc_characteristics(True)