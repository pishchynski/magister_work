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
        self._eps_proba = 10 ** (-6)

        self.queries_stream = BMMAPStream(test_matrD_0, test_matrD)
        self.serv_stream = PHStream(test_vect_beta, test_matrS)
        self.timer_stream = PHStream(test_vect_gamma, test_matrGamma)
        self.matr_hat_Gamma = np.array(np.bmat([[np.zeros((1, self.timer_stream.repres_matr_0.shape[1])),
                                                 np.zeros((1, self.timer_stream.repres_matr.shape[1]))],
                                                [self.timer_stream.repres_matr_0, self.timer_stream.repres_matr]]))
        self.I_WM = np.eye(self.queries_stream.dim_ * self.serv_stream.dim)
        self.I_W = np.eye(self.queries_stream.dim_)
        self.I_M = np.eye(self.serv_stream.dim)
        self.S_0xBeta = np.dot(self.serv_stream.repres_matr_0,
                               self.serv_stream.repres_vect)

        self.p_hp = 0.5
        self.n = 3
        self.N = 20
        self.ramatrL, self.ramatrA, self.ramatrP = self._calc_ramaswami_matrices(0, self.N)

        self.generator = None

        self.recalculate_generator()

    def set_BMMAP_queries_stream(self, matrD_0, matrD, q=0.8, n=3, recalculate_generator=False):
        self.queries_stream = BMAPStream(matrD_0, matrD, q, n)
        self.n = n
        if recalculate_generator:
            self.recalculate_generator()

    def set_PH_serv_stream(self, vect, matr, recalculate_generator=False):
        self.serv_stream = PHStream(vect, matr)
        if recalculate_generator:
            self.recalculate_generator()

    def set_PH_timer_stream(self, vect, matr, recalculate_generator=False):
        self.timer_stream = PHStream(vect, matr)
        if recalculate_generator:
            self.recalculate_generator()

    def recalculate_generator(self, verbose=False):
        if verbose:
            print('======= Input BMMAP Parameters =======')
            self.queries_stream.print_characteristics('D')

            print('======= PH service time parameters =======')
            self.serv_stream.print_characteristics('S', 'beta')

            print('======= PH timer parameters =======')
            self.timer_stream.print_characteristics('Г', 'gamma')

        matrQ_0k = self._calc_Q_0k()
        matrQ_iiprev = self._calc_Q_iiprev()
        matrQ_ii = self._calc_Q_ii()
        matrQ_iik = self._calc_matrQ_iik()
        matrQ_iN = self._calc_matrQ_iN()

        self.check_generator(matrQ_0k, matrQ_iiprev, matrQ_ii, matrQ_iik, matrQ_iN)
        self.finalize_generator(matrQ_0k, matrQ_iiprev, matrQ_ii, matrQ_iik, matrQ_iN)

        print("generator checked")

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
        print("Calculating Q_00")

        block00 = copy.deepcopy(self.queries_stream.matrD_0)
        block01 = kron(self.queries_stream.transition_matrices[0][1] + self.queries_stream.transition_matrices[1][1],
                       self.serv_stream.repres_vect)
        block10 = kron(np.eye(self.queries_stream.dim_),
                       self.serv_stream.repres_matr_0)
        block11 = kronsum(self.queries_stream.matrD_0,
                          self.serv_stream.repres_matr)

        matrQ_00 = np.bmat([[block00, block01],
                            [block10, block11]])

        print("Q_00 calculated")

        return np.array(matrQ_00)

    def _calc_Q_0k(self):
        """
        Calculates matrices Q_{0, k}, k = [0, N]

        :return: list of np.arrays with Q_{0, k}
        """
        print("Calculating Q_0k")
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

            temp_block = np.zeros((self.queries_stream.dim_,
                                   self.queries_stream.dim_ * self.serv_stream.dim * sum(
                                       [ncr(j + self.timer_stream.dim - 1,
                                            self.timer_stream.dim - 1)
                                        for j in range(1, k)]
                                   )))
            blocks0k.append(temp_block)

            temp_block = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                                   self.queries_stream.dim_ * self.serv_stream.dim * sum(
                                       [ncr(j + self.timer_stream.dim - 1,
                                            self.timer_stream.dim - 1)
                                        for j in range(1, k)]
                                   )))
            blocks1k.append(temp_block)

            ramatrP_mul = copy.deepcopy(self.ramatrP[-1][0])
            for i in range(1, k):
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

            temp_matr = np.array(np.bmat([blocks0k,
                                          blocks1k]))

            matrQ_0k.append(temp_matr)

        matrQ_0k.append(self._calc_Q_0N())

        print("Q_0k calculated")

        return matrQ_0k

    def _calc_Q_0N(self):
        print("Calculating Q_0N")

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
                           np.eye(self.serv_stream.dim))
        else:
            block10 = kron(self.queries_stream.transition_matrices[0][self.N],
                           np.eye(self.serv_stream.dim))
            for i in range(self.N + 1, self.n + 1):
                block10 += kron(self.queries_stream.transition_matrices[0][i],
                                np.eye(self.serv_stream.dim))

        blocks0k = [block00]
        blocks1k = [block10]

        temp_block = np.zeros((self.queries_stream.dim_,
                               self.queries_stream.dim_ * self.serv_stream.dim * sum(
                                   [ncr(j + self.timer_stream.dim - 1,
                                        self.timer_stream.dim - 1)
                                    for j in range(1, self.N)]
                               )))
        blocks0k.append(temp_block)

        temp_block = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                               self.queries_stream.dim_ * self.serv_stream.dim * sum(
                                   [ncr(j + self.timer_stream.dim - 1,
                                        self.timer_stream.dim - 1)
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

        print("Q_0N calculated")

        return np.array(np.bmat([blocks0k,
                                 blocks1k]))

    def _calc_Q_10(self):
        print("Calculating Q_10")

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

        print("Q_10 calculated")

        return np.array(np.bmat([[block00, block01],
                                 [block10, block11]]))

    def _calc_Q_iiprev(self):
        print("Calculating Q_{i, i - 1}")

        matrQ_iiprev = [None, self._calc_Q_10()]
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

        print("Q_{i, i - 1} calculated")

        return matrQ_iiprev

    def _calc_Q_ii(self):
        """
        Calculates matrices Q_{i,i} including Q_{N, N}

        :return: list of matrices Q_ii
        """
        print("Calculating Q_{i, i}")

        matrQ_ii = [None]
        for i in range(1, self.N + 1):
            cur_matr = la.block_diag(
                *(kron(kronsum(self.queries_stream.matrD_0 if i != self.N else self.queries_stream.matrD_1_,
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

        print("Q_{i, i} calculated")

        return matrQ_ii

    def __get_ramatrP_mul(self, j, k):
        ramatrP_mul = self.ramatrP[-1][j]
        for i in range(j + 1, j + k):
            ramatrP_mul = np.dot(ramatrP_mul, self.ramatrP[-1][i])

        return ramatrP_mul

    def _calc_matrQ_iik(self):
        print("Calculating Q_{i, i + k}")

        matrQ_iik = [None]
        for i in range(1, self.N):

            cur_zero_matr = la.block_diag(*(kron(np.zeros(self.queries_stream.transition_matrices[0][1].shape),
                                            np.eye(self.serv_stream.dim * ncr(j + self.timer_stream.dim - 1,
                                                                              self.timer_stream.dim - 1)))
                                       for j in range(i + 1)))

            matrQ_ii_row = []
            for k in range(1, self.N - i):
                if k <= self.n:
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
                else:


                    zero_matr = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim * np.sum(
                        [ncr(j + self.timer_stream.dim - 1,
                             self.timer_stream.dim - 1)
                         for j in range(i + 1)]),
                                          self.queries_stream.dim_ * self.serv_stream.dim * np.sum(
                                              [ncr(j + self.timer_stream.dim - 1,
                                                   self.timer_stream.dim - 1)
                                               for j in range(i + 1, i + k + 1)])
                                          ))

                    cur_matr = np.concatenate((cur_zero_matr, zero_matr), axis=1)

                    temp_matr = la.block_diag(*(kron(kron(np.zeros(self.queries_stream.transition_matrices[1][1].shape),
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

        print("Q_{i, i + k} calculated")

        return matrQ_iik

    def _calc_matrQ_iN(self):
        """
        Calculates matrices Q_{i, N}m i = [1, N - 1]

        :return: list of np.arrays with Q_{i, N}
        """
        print("Calculating Q_{i, N}")

        matrQ_iN = [None]
        for i in range(1, self.N):
            matrD1_sum = self.queries_stream.transition_matrices[0][-1]
            matrD2_sum = self.queries_stream.transition_matrices[1][-1]

            if self.N - i <= self.n:
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

            print("Q_{i, N} calculated")

        return matrQ_iN

    def check_generator(self, matrQ_0k, matrQ_iiprev, matrQ_ii, matrQ_iik, matrQ_iN):
        """
        Checks first block-row for by-row sum equality to zero
        Adds deltas to Q_{i,i} for generator to satisfy this condition

        :param matrQ_0k:
        :param matrQ_iiprev:
        :param matrQ_ii:
        :param matrQ_iik:
        :param matrQ_iN:
        """
        # First block-row
        print("Generator check started")

        for i in range(self.queries_stream.dim_ + self.queries_stream.dim_ * self.serv_stream.dim):
            temp = 0
            for matr in matrQ_0k:
                temp += np.sum(matr[i])
            if temp > 10 ** (-6):
                print("Line", i, "=", temp)

        # Other block-rows except Nth
        for i in range(1, self.N):
            # Iterating block-rows
            temp = np.sum(matrQ_iiprev[i], axis=1)
            temp += np.sum(matrQ_ii[i], axis=1)
            for block in matrQ_iik[i]:
                temp += np.sum(block, axis=1)
            temp += np.sum(matrQ_iN[i])
            delta = np.diag(-temp)

            matrQ_ii[i] += delta

        # Nth block-row
        temp = np.sum(matrQ_iiprev[self.N], axis=1) + np.sum(matrQ_ii[self.N], axis=1)
        delta = np.diag(-temp)
        matrQ_ii[self.N] += delta
        print("Generator checked")

    def finalize_generator(self, matrQ_0k, matrQ_iiprev, matrQ_ii, matrQ_iik, matrQ_iN):
        print("Finalizaing generator")

        matrQ = [[None for _ in range(self.N + 1)] for _ in range(self.N + 1)]
        matrQ[0] = matrQ_0k
        for i in range(1, self.N + 1):
            matrQ[i][i] = matrQ_ii[i]
            matrQ[i][i - 1] = matrQ_iiprev[i]
            if i < self.N:
                for k, matr in enumerate(matrQ_iik[i]):
                    matrQ[i][i + k + 1] = matr

                matrQ[i][self.N] = matrQ_iN[i]
        self.generator = matrQ
        print("Generator finalized")

    def _calc_matrG(self):
        print("Calculating G")

        matrG = [None for _ in range(self.N)]
        matrQ = self.generator
        for i in range(self.N - 1, -1, -1):
            tempG = -matrQ[i + 1][i + 1]
            tempSum = None
            for j in range(1, self.N - i - 1):
                temp = matrQ[i + 1][i + 1 + j]
                for k in range(i + j, i, -1):
                    temp = np.dot(temp, matrG[k])
                if tempSum is None:
                    tempSum = temp
                else:
                    tempSum = tempSum + temp
            if not tempSum is None:
                tempG = tempG - tempSum
            tempG = la.inv(tempG)
            tempG = np.dot(tempG, matrQ[i + 1][i])
            matrG[i] = tempG

        print("G calculated")

        return matrG

    def _calc_matrQover(self, matrG):
        print("Calculating Q_over")

        matrQ = self.generator
        matrQover = [[None for _ in range(self.N)] + [matrQ[i][self.N]] for i in range(self.N + 1)]
        for k in range(self.N - 1, -1, -1):
            for i in range(k + 1):
                matrQover[i][k] = matrQ[i][k] + np.dot(matrQover[i][k + 1], matrG[k])

        print("Q_over calculated")

        return matrQover

    def _calc_matrF(self, matrQover):
        print("Calculating F")

        matrF = [None]
        for i in range(1, self.N + 1):
            tempF = matrQover[0][i]
            for j in range(1, i):
                tempF = tempF + np.dot(matrF[j], matrQover[j][i])
            tempF = np.dot(tempF, la.inv(-matrQover[i][i]))
            matrF.append(tempF)

        print("F calculated")

        return matrF

    def _calc_p0(self, matrF, matrQover):
        matr_a = matrQover[0][0]
        vect_eaR = e_col(matrF[1].shape[0])
        for i in range(1, self.N + 1):
            vect_e = e_col(matrF[i].shape[1])
            vect_eaR += np.dot(matrF[i], vect_e)

        for i in range(matr_a.shape[0]):
            matr_a[i][0] = vect_eaR[i][0]

        matr_b = np.zeros((matr_a.shape[0], 1))
        matr_b[0][0] = 1.
        matr_a = np.transpose(matr_a)
        p0 = np.transpose(la.solve(matr_a, matr_b))

        return p0

    def calc_stationary_probas(self):
        matrG = self._calc_matrG()
        matrQover = self._calc_matrQover(matrG)
        matrF = self._calc_matrF(matrQover)
        p0 = self._calc_p0(matrF, matrQover)
        stationary_probas = [p0]
        for i in range(1, self.N + 1):
            stationary_probas.append(np.dot(p0, matrF[i]))
        return stationary_probas

    def check_probas(self, stationary_probas):
        sum = 0.0
        for num, proba in enumerate(stationary_probas):
            print("p" + str(num) + ": " + str(proba))
            temp_sum = np.sum(proba)
            sum += temp_sum
            print("sum: " + str(temp_sum) + "\n")

        return 1 - self._eps_proba < sum < 1 + self._eps_proba

    def calc_system_empty_proba(self, stationary_probas):
        r_multiplier = np.array(np.bmat([[e_col(self.queries_stream.dim_)],
                                         [np.zeros((self.queries_stream.dim_ * self.serv_stream.dim, 1))]]))
        return np.dot(stationary_probas[0], r_multiplier)[0][0]

    def calc_system_single_query_proba(self, stationary_probas):
        r_multiplier = np.array(np.bmat([[np.zeros((self.queries_stream.dim_, 1))],
                                         [e_col(self.queries_stream.dim_ * self.serv_stream.dim)]]))
        return np.dot(stationary_probas[0], r_multiplier)[0][0]

    def calc_buffer_i_queries_j_nonprior(self, stationary_probas, i, j):
        mulW_M = self.queries_stream.dim_ * self.serv_stream.dim
        block0_size = int(mulW_M * np.sum([ncr(l + self.timer_stream.dim - 1,
                                               self.timer_stream.dim - 1) for l in range(j)]))
        block1_size = int(mulW_M * ncr(j + self.timer_stream.dim - 1,
                                       self.timer_stream.dim - 1))
        block2_size = int(mulW_M * np.sum([ncr(l + self.timer_stream.dim - 1,
                                               self.timer_stream.dim - 1) for l in range(j + 1, i + 1)]))
        r_multiplier = np.array(np.bmat([[np.zeros((block0_size, 1))],
                                         [e_col(block1_size)],
                                         [np.zeros((block2_size, 1))]]))
        return np.dot(stationary_probas[i],
                      r_multiplier)[0][0]

    def calc_buffer_i_queries(self, stationary_probas, i):
        return np.sum([self.calc_buffer_i_queries_j_nonprior(stationary_probas, i, j) for j in range(i + 1)])

    def calc_avg_buffer_queries_num(self, stationary_probas):
        return np.sum([i * self.calc_buffer_i_queries(stationary_probas, i) for i in range(1, self.N + 1)])

    def calc_avg_buffer_nonprior_queries_num(self, stationary_probas):
        return np.sum(
            [np.sum([j * self.calc_buffer_i_queries_j_nonprior(stationary_probas, i, j) for j in range(1, i + 1)]) for i
             in range(1, self.N + 1)])

    def calc_query_lost_p(self, stationary_probas):
        p_loss = np.dot(stationary_probas[0],
                        np.array(np.bmat([[np.zeros((self.queries_stream.dim_, self.serv_stream.dim))],
                                          [kron(e_col(self.queries_stream.dim_),
                                                self.I_M)]])))
        r_sum = np.dot(stationary_probas[1],
                       kron(kron(e_col(self.queries_stream.dim_),
                                 self.I_M),
                            e_col(np.sum([ncr(j + self.timer_stream.dim - 1,
                                              self.timer_stream.dim - 1) for j in range(2)]))))
        for i in range(2, self.N + 1):
            r_sum += np.dot(stationary_probas[i],
                            kron(kron(e_col(self.queries_stream.dim_),
                                      self.I_M),
                                 e_col(np.sum([ncr(j + self.timer_stream.dim - 1,
                                                   self.timer_stream.dim - 1) for j in range(i + 1)]))))

        p_loss = p_loss + r_sum
        p_loss = np.dot(p_loss, self.serv_stream.repres_matr_0)
        p_loss = 1 - (1 / self.queries_stream.avg_intensity) * p_loss[0][0]

        return p_loss

    def calc_characteristics(self, verbose=False):
        stationary_probas = self.calc_stationary_probas()
        if self.check_probas(stationary_probas):
            print("stationary probas calculated\n")
        else:
            print("stationary probas calculated with error!\n", file=sys.stderr)

        system_empty_proba = self.calc_system_empty_proba(stationary_probas)
        print("p_0 =", system_empty_proba)

        system_single_query_proba = self.calc_system_single_query_proba(stationary_probas)
        print("p_1 =", system_single_query_proba)

        avg_buffer_queries_num = self.calc_avg_buffer_queries_num(stationary_probas)
        print("L_buf =", avg_buffer_queries_num)

        avg_buffer_nonprior_num = self.calc_avg_buffer_nonprior_queries_num(stationary_probas)
        print("q_j =", avg_buffer_nonprior_num)

        p_loss = self.calc_query_lost_p(stationary_probas)
        print("P_loss =", p_loss)


if __name__ == '__main__':
    qs = TwoPrioritiesQueueingSystem()
    qs.calc_characteristics(True)
