import sys

import scipy.sparse as sparse

from matr_E import get_matr_E

sys.path.append("../")
from streams import *
from ramaswami import calc_ramaswami_matrices
import experiments_data.BMMAP3_Poisson_PH_PH as test

np.set_printoptions(threshold=np.inf, suppress=True, formatter={'float': '{: 0.8f}'.format}, linewidth=75)


class TwoPrioritiesQueueingSystem:
    """
    Class describing Heterogeneous Reliable Queueing System with Finite Buffer and
    Two priorities of queries.
    In Kendall designation (informal): BMMAP|PH|TimerPH|N - unchecked(?)
    """

    def __init__(self, experiment_data, name='Default system', verbose=False):
        self.name = name

        self._eps_proba = 10 ** (-6)

        self.queries_stream = BMMAPStream(matrD_0=experiment_data.test_matrD_0,
                                          matrD=experiment_data.test_matrD,
                                          qs=experiment_data.test_qs,
                                          ns=experiment_data.test_ns,
                                          priority_part=experiment_data.priority_part)
        self.serv_stream = PHStream(experiment_data.test_vect_beta, experiment_data.test_matrS)
        self.timer_stream = PHStream(experiment_data.test_vect_gamma, experiment_data.test_matrGamma)
        self.matr_hat_Gamma = np.array(np.bmat([[np.zeros((1, self.timer_stream.repres_matr_0.shape[1])),
                                                 np.zeros((1, self.timer_stream.repres_matr.shape[1]))],
                                                [self.timer_stream.repres_matr_0,
                                                 self.timer_stream.repres_matr]]),
                                       dtype=float)

        self.I_WM = np.eye(self.queries_stream.dim_ * self.serv_stream.dim)
        self.I_W = np.eye(self.queries_stream.dim_)
        self.I_M = np.eye(self.serv_stream.dim)

        self.O_W = np.zeros((self.queries_stream.dim_, self.queries_stream.dim_))
        self.O_WM = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                              self.queries_stream.dim_ * self.serv_stream.dim))

        self.e_WM = e_col(self.queries_stream.dim_ * self.serv_stream.dim)
        self.e_M = e_col(self.serv_stream.dim)

        self.S_0xBeta = np.dot(self.serv_stream.repres_matr_0,
                               self.serv_stream.repres_vect)

        self.p_hp = experiment_data.p_hp  # probability of the query to leave system after the timer's up
        self.ns = experiment_data.test_ns  # number of D_i matrices
        self.n = max(self.ns)

        self.N = experiment_data.test_N  # buffer capacity
        self.ramatrL, self.ramatrA, self.ramatrP = self._calc_ramaswami_matrices(0, self.N) # Ramaswami matrices
        self.calI_1 = [np.array(np.bmat([[kron(kron(self.I_W,
                                                    e_col(self.serv_stream.dim)),
                                               e_col(ncr(self.timer_stream.dim - 1 + j,
                                                         self.timer_stream.dim - 1)))] for j in range(i + 1)]),
                                dtype=float) for i in
                       range(self.N + 1)]
        self.calI_2 = [np.array(np.bmat([[kron(kron(e_col(self.queries_stream.dim_),
                                                    self.I_M),
                                               e_col(ncr(self.timer_stream.dim - 1 + j,
                                                         self.timer_stream.dim - 1)))] for j in range(i + 1)]),
                                dtype=float) for i in
                       range(self.N + 1)]
        self.calI_L = [None] + [np.array(np.bmat([[kron(self.e_WM,
                                                        r_multiply_e(self.ramatrL[self.N - i + j][
                                                                         self.N - i]) if j != 0 else np.array([[0]]))]
                                                  for j in range(i + 1)]))
                                for i in range(1, self.N + 1)]

        self.matrAs = self.calcMatrAs()

        self.W_1_inf = 1.
        self.W_2_inf = 1.

        if verbose:
            print("\n=====RAMASWAMI MATRICES=====\n")
            for block_num, matr in enumerate(self.ramatrL[self.N]):
                print('L_' + str(self.N) + ',' + str(block_num) + '\n', matr)

            for block_num, matr in enumerate(self.ramatrA[self.N]):
                print('A_' + str(self.N) + ',' + str(block_num) + '\n', matr)

            for block_num, matr in enumerate(self.ramatrP[-1]):
                print('P_' + str(block_num) + '\n', matr)

            print("\n=====END RAMASWAMI MATRICES=====\n")

        self.generator = None
        self.sparse_generator = None

        self.recalculate_generator(verbose=verbose)

    def set_BMMAP_queries_stream(self, matrD_0, matrD, qs=(0.8, 0.8), ns=(3, 3), recalculate_generator=False):
        self.queries_stream = BMMAPStream(matrD_0, matrD, qs, ns)
        self.ns = ns
        self.n = max(self.ns)

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
        """
        Recalculates infinitesimal generator with streams set in the system's instance

        :param verbose: whether to print streams' characteristics (default is False)
        :return: None
        """

        if verbose:
            print('======= Input BMMAP Parameters =======')
            self.queries_stream.print_characteristics('D')

            print('======= PH service time parameters =======')
            self.serv_stream.print_characteristics('S', 'beta')

            print('======= PH timer parameters =======')
            self.timer_stream.print_characteristics('Г', 'gamma')

        # print("Generator recalculating")

        matrQ_0k = self._calc_Q_0k()
        matrQ_iiprev = self._calc_Q_iiprev()
        matrQ_ii = self._calc_Q_ii()
        matrQ_iik = self._calc_matrQ_iik()
        matrQ_iN = self._calc_matrQ_iN()

        self.check_generator(matrQ_0k, matrQ_iiprev, matrQ_ii, matrQ_iik, matrQ_iN)
        self.finalize_generator(matrQ_0k, matrQ_iiprev, matrQ_ii, matrQ_iik, matrQ_iN)

        # print("Generator recalculated:")

        if verbose:
            for row_num, block_row in enumerate(self.generator):
                print('Row', row_num)
                for block_num, block in enumerate(block_row):
                    print('Q_' + str(row_num) + ',' + str(block_num), block)

    def _calc_ramaswami_matrices(self, start=0, end=None, verbose=False):
        """
        Calculates Ramaswami matrices L, A and P.

        :param start: not implemented, assume start = 0
        :param end:
        :return: tuple of matrices L, A, P
        """

        if end is None:
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
        """
        Calculates Q_{0,0} block of the infinitesimal generator.

        :return: numpy.array
        """

        # print("Calculating Q_00")

        block00 = copy.deepcopy(self.queries_stream.matrD_0)
        block01 = kron(self.queries_stream.transition_matrices[0][1] + self.queries_stream.transition_matrices[1][1],
                       self.serv_stream.repres_vect)
        block10 = kron(self.I_W,
                       self.serv_stream.repres_matr_0)
        block11 = kronsum(self.queries_stream.matrD_0,
                          self.serv_stream.repres_matr)

        matrQ_00 = np.bmat([[block00, block01],
                            [block10, block11]])

        # print("Q_00 calculated")

        return np.array(matrQ_00, dtype=float)

    def _calc_Q_0k(self):
        """
        Calculates matrices Q_{0, k}, k = [0, N]

        :return: list of np.arrays with Q_{0, k}
        """
        # print("Calculating Q_0k")
        matrQ_0k = [self._calc_Q_00()]

        for k in range(1, self.N):
            if k + 1 > self.n:
                block00 = kron(np.zeros(self.queries_stream.transition_matrices[0][1].shape),
                               # this shape suits because all D_{i}^{1} have same shape
                               self.serv_stream.repres_vect)
            else:
                block00 = kron(self.queries_stream.transition_matrices[0][k + 1],
                               self.serv_stream.repres_vect)

            if k > self.n:
                block10 = kron(np.zeros(self.queries_stream.transition_matrices[0][1].shape),
                               # this shape suits because all D_{i}^{1} have same shape
                               self.I_M)
            else:
                block10 = kron(self.queries_stream.transition_matrices[0][k],
                               self.I_M)

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

            ramatrP_mul = copy.deepcopy(
                self.ramatrP[-1][0])  # first index is '-1' because calc_ramaswami() returns all iterations of matrices
            for i in range(1, k):
                ramatrP_mul = np.dot(ramatrP_mul, self.ramatrP[-1][i])

            if k + 1 > self.n:
                last_block0 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][1].shape),
                                        self.serv_stream.repres_vect),
                                   ramatrP_mul)
            else:
                last_block0 = kron(kron(self.queries_stream.transition_matrices[1][k + 1],
                                        self.serv_stream.repres_vect),
                                   ramatrP_mul)

            if k > self.n:
                last_block1 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][1].shape),
                                        self.I_M),
                                   ramatrP_mul)
            else:
                last_block1 = kron(kron(self.queries_stream.transition_matrices[1][k],
                                        self.I_M),
                                   ramatrP_mul)

            blocks0k.append(last_block0)
            blocks1k.append(last_block1)

            temp_matr = np.array(np.bmat([blocks0k,
                                          blocks1k]),
                                 dtype=float)

            matrQ_0k.append(temp_matr)

        matrQ_0k.append(self._calc_Q_0N())

        # print("Q_0k calculated")

        return matrQ_0k

    def _calc_Q_0N(self):
        """
        Calculates Q_{0,N} block of the infinitesimal generator

        :return:
        """

        # print("Calculating Q_0N")

        if self.N + 1 > self.n:
            block00 = kron(np.zeros(self.queries_stream.transition_matrices[0][1].shape),
                           self.serv_stream.repres_vect)
        else:
            block00 = kron(self.queries_stream.transition_matrices[0][self.N + 1],
                           self.serv_stream.repres_vect)
            for i in range(self.N + 2, self.n + 1):
                block00 += kron(self.queries_stream.transition_matrices[0][i],
                                self.serv_stream.repres_vect)

        if self.N > self.n:
            block10 = kron(np.zeros(self.queries_stream.transition_matrices[0][1].shape),
                           self.I_M)
        else:
            block10 = kron(self.queries_stream.transition_matrices[0][self.N],
                           self.I_M)
            for i in range(self.N + 1, self.n + 1):
                block10 += kron(self.queries_stream.transition_matrices[0][i],
                                self.I_M)

        blocks0k = [block00]
        blocks1k = [block10]

        if self.N > 1:
            temp_block = np.zeros((self.queries_stream.dim_,
                                   self.queries_stream.dim_ * self.serv_stream.dim * sum(
                                       [ncr(j + self.timer_stream.dim - 1,
                                            self.timer_stream.dim - 1)
                                        for j in range(1, self.N)]
                                   )))
            blocks0k.append(copy.deepcopy(temp_block))

            temp_block = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                                   self.queries_stream.dim_ * self.serv_stream.dim * sum(
                                       [ncr(j + self.timer_stream.dim - 1,
                                            self.timer_stream.dim - 1)
                                        for j in range(1, self.N)]
                                   )))
            blocks1k.append(copy.deepcopy(temp_block))

        ramatrP_mul = copy.deepcopy(self.ramatrP[self.N][0])
        for i in range(1, self.N):
            ramatrP_mul = np.dot(ramatrP_mul, self.ramatrP[self.N][i])

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

        if self.N > self.n:
            last_block1 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][1].shape),
                                    self.I_M),
                               ramatrP_mul)
        else:
            last_block1 = kron(kron(self.queries_stream.transition_matrices[1][self.N],
                                    self.I_M),
                               ramatrP_mul)
            for i in range(self.N + 1, self.n + 1):
                last_block1 += kron(kron(self.queries_stream.transition_matrices[1][i],
                                         self.I_M),
                                    ramatrP_mul)

        blocks0k.append(last_block0)
        blocks1k.append(last_block1)

        # print("Q_0N calculated")

        return np.array(np.bmat([blocks0k,
                                 blocks1k]),
                        dtype=float)

    def _calc_Q_10(self):
        """
        Calculates Q_{1,0} block of the infinitesimal generator

        :return:
        """

        # print("Calculating Q_10")

        block00 = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                            self.queries_stream.dim_))
        block01 = kron(self.I_W,
                       self.S_0xBeta)
        block10 = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim * self.timer_stream.dim,
                            self.queries_stream.dim_))
        block11 = kron(kron(self.I_W,
                            self.S_0xBeta),
                       e_col(self.timer_stream.dim))
        block11 += self.p_hp * kron(self.I_WM,
                                    self.timer_stream.repres_matr_0)

        # print("Q_10 calculated")

        return np.array(np.bmat([[block00, block01],
                                 [block10, block11]]),
                        dtype=float)

    def _calc_Q_iiprev(self):
        """
        Calculates Q_{i,i-1} blocks of the infinitesimal generator

        :return:
        """

        # print("Calculating Q_{i, i - 1}")

        matrQ_iiprev = [None, self._calc_Q_10()]
        for i in range(2, self.N + 1):
            blocks0 = [kron(self.I_W,
                            self.S_0xBeta)]
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

        # print("Q_{i, i - 1} calculated")

        return matrQ_iiprev

    def _calc_Q_ii(self):
        """
        Calculates matrices Q_{i,i} including Q_{N, N}

        :return: list of matrices Q_ii
        """
        # print("Calculating Q_{i, i}")

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

            matrQ_ii.append(cur_matr)

        # print("Q_{i, i} calculated")

        return matrQ_ii

    def __get_ramatrP_mul(self, j, k):
        ramatrP_mul = self.ramatrP[-1][j]
        for i in range(j + 1, j + k):
            ramatrP_mul = np.dot(ramatrP_mul, self.ramatrP[-1][i])

        return ramatrP_mul

    def _calc_matrQ_iik(self):
        """
        Calculates Q_{i,i+k} blocks of the infinitesimal generator

        :return:
        """

        # print("Calculating Q_{i, i + k}")

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

        # print("Q_{i, i + k} calculated")

        return matrQ_iik

    def _calc_matrQ_iN(self):
        """
        Calculates matrices Q_{i, N}, i = [1, N - 1]

        :return: list of np.arrays with Q_{i, N}
        """
        # print("Calculating Q_{i, N}")

        matrQ_iN = [None]
        for i in range(1, self.N):
            matrD1_sum = np.zeros(self.queries_stream.transition_matrices[0][-1].shape)
            matrD2_sum = np.zeros(self.queries_stream.transition_matrices[1][-1].shape)

            if self.N - i <= self.n:
                matrD1_sum = copy.deepcopy(self.queries_stream.transition_matrices[0][self.N - i])
                matrD2_sum = copy.deepcopy(self.queries_stream.transition_matrices[1][self.N - i])
            for k in range(self.N - i + 1, self.n + 1):
                matrD1_sum += copy.deepcopy(self.queries_stream.transition_matrices[0][k])
                matrD2_sum += copy.deepcopy(self.queries_stream.transition_matrices[1][k])

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

        # print("Q_{i, N} calculated")

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
        # print("Generator check started")

        for i in range(self.queries_stream.dim_ + self.queries_stream.dim_ * self.serv_stream.dim):
            temp = 0
            for matr in matrQ_0k:
                temp += np.sum(matr[i])
            if abs(temp) > 10 ** (-6):
                print("Line", i, "=", temp)

        # Other block-rows except Nth
        for i in range(1, self.N):
            # Iterating block-rows
            temp = np.sum(matrQ_iiprev[i], axis=1)
            temp += np.sum(matrQ_ii[i], axis=1)
            for block in matrQ_iik[i]:
                temp += np.sum(block, axis=1)
            temp += np.sum(matrQ_iN[i], axis=1)
            delta = np.diag(-temp)

            matrQ_ii[i] += delta

        # Nth block-row
        temp = np.sum(matrQ_iiprev[self.N], axis=1) + np.sum(matrQ_ii[self.N], axis=1)
        delta = np.diag(-temp)
        matrQ_ii[self.N] += delta
        # print("Generator checked")

    def finalize_generator(self, matrQ_0k, matrQ_iiprev, matrQ_ii, matrQ_iik, matrQ_iN):
        # print("Finalizing generator")

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
        # print("Generator finalized")
        # print("Creating sparse generator")

        # Sparse generator was added to find stationary distribution by solving linear system.
        # Not needed for recurrent algorithm
        sparse_matrQ = [[None if matrQ[i][j] is None else sparse.csr_matrix(matrQ[i][j]) for j in range(self.N + 1)] for
                        i in range(self.N + 1)]
        self.sparse_generator = sparse.bmat(sparse_matrQ, "csr")

        # print("Sparse generator created")

    def _calc_matrG(self):
        # print("Calculating G")

        matrG = [None for _ in range(self.N)]
        matrQ = self.generator
        for i in range(self.N - 1, -1, -1):
            tempG = -matrQ[i + 1][i + 1]
            temp_sum = None
            for j in range(1, self.N - i):
                temp = matrQ[i + 1][i + 1 + j]
                for k in range(i + j, i, -1):
                    temp = np.dot(temp, matrG[k])
                if temp_sum is None:
                    temp_sum = temp
                else:
                    temp_sum = temp_sum + temp
            if temp_sum is not None:
                tempG = tempG - temp_sum
            tempG = la.inv(tempG)
            tempG = np.dot(tempG, matrQ[i + 1][i])
            matrG[i] = tempG

        # print("G calculated")

        return matrG

    def _calc_matrQover(self, matrG):
        # print("Calculating Q_over")

        matrQ = self.generator
        matrQover = [[None for _ in range(self.N)] + [copy.deepcopy(matrQ[i][self.N])] for i in range(self.N + 1)]
        for k in range(self.N - 1, -1, -1):
            for i in range(k + 1):
                matrQover[i][k] = matrQ[i][k] + np.dot(matrQover[i][k + 1], matrG[k])

        # print("Q_over calculated")

        return matrQover

    def _calc_matrF(self, matrQover):
        # print("Calculating F")

        matrF = [None]
        for i in range(1, self.N + 1):
            tempF = copy.deepcopy(matrQover[0][i])
            for j in range(1, i):
                tempF = tempF + np.dot(matrF[j], matrQover[j][i])
            tempF = np.dot(tempF, la.inv(-matrQover[i][i]))
            matrF.append(tempF)

        # print("F calculated")

        return matrF

    def _calc_p0(self, matrF, matrQover):
        """
        Calculates p_0 stationary probabilities vector
        """

        matr_a = copy.deepcopy(matrQover[0][0])
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

        # print('\nChecking p_0 * Qover_{0, 0}')
        # print(np.dot(p0, matrQover[0][0]), '\n')

        # print('\nChecking p_0 * matr_a')
        # print(np.dot(p0, np.transpose(matr_a)), '\n')

        return p0

    def calc_stationary_probas(self, verbose=True):
        """
        Calculates stationary distribution

        :param verbose:
        :return:
        """
        matrG = self._calc_matrG()

        if verbose:
            for ind, elem in enumerate(matrG):
                print('G_' + str(ind), elem)

        matrQover = self._calc_matrQover(matrG)

        if verbose:
            for row_num, block_row in enumerate(matrQover):
                print('Row', row_num)
                for block_num, block in enumerate(block_row):
                    print('Q_' + str(row_num) + ',' + str(block_num), block)

        matrF = self._calc_matrF(matrQover)

        if verbose:
            for ind, elem in enumerate(matrF):
                print('F_' + str(ind), elem)

        p0 = self._calc_p0(matrF, matrQover)
        stationary_probas = [p0]
        for i in range(1, self.N + 1):
            stationary_probas.append(np.dot(p0, matrF[i]))

        if self.check_probas(stationary_probas, verbose):
            pass  # print("stationary probas calculated\n")
        else:
            print("stationary probas calculated with error!\n", file=sys.stderr)

        self.W_1_inf = self.func_tilde_W_1(stationary_probas, 10 ** 6)[0][0]
        self.W_2_inf = self.func_tilde_W_2(stationary_probas, 10 ** 6)[0][0]

        return stationary_probas

    def calc_stationary_probas_classic(self):
        """
        Calculates stationary distribution by solving linear system

        :return:
        """

        # print("Calculating probas via pure formula")
        sol = system_solve(self.sparse_generator.toarray())
        ps = [[sol[:self.queries_stream.dim_ * (self.serv_stream.dim + 1)]]]
        prev = self.queries_stream.dim_ * (self.serv_stream.dim + 1)
        for i in range(1, self.N + 1):
            cur = self.queries_stream.dim_ * self.serv_stream.dim * sum((ncr(j + self.timer_stream.dim - 1,
                                                                                self.timer_stream.dim - 1) for j in
                                                                            range(i + 1)))
            ps.append([sol[prev: prev + cur]])
            prev = prev + cur

        if self.check_probas(ps):
            pass  # print("stationary probas calculated\n")
        else:
            print("stationary probas calculated with error!\n", file=sys.stderr)

        return ps

    def check_by_Q_multiplying(self, stationary_probas):
        # print("Checking probas by Q multiplying.")
        # print("Should contain zeros only!")
        ps = np.concatenate(stationary_probas, axis=1)
        # print(np.dot(ps, self.sparse_generator.toarray()))

    def check_probas(self, stationary_probas, verbose=True):
        sum = 0.0
        for num, proba in enumerate(stationary_probas):
            if verbose:
                print("p" + str(num) + ": " + to_latex_table(proba))
            temp_sum = np.sum(proba)
            sum += temp_sum
            if verbose:
                print("sum: " + str(temp_sum) + "\n")
        if verbose:
            print("all_sum:", str(sum))

        return 1 - self._eps_proba < sum < 1 + self._eps_proba

    def calc_system_empty_proba(self, stationary_probas):
        """
        Performance characteristic: probability that system is empty

        :param stationary_probas:
        :return:
        """
        r_multiplier = np.array(np.bmat([[e_col(self.queries_stream.dim_)],
                                         [np.zeros((self.queries_stream.dim_ * self.serv_stream.dim, 1))]]),
                                dtype=float)
        return np.dot(stationary_probas[0], r_multiplier)[0][0]

    def calc_system_single_query_proba(self, stationary_probas):
        """
        Performance characteristic: probability that system has single query and it's serving already

        :param stationary_probas:
        :return:
        """
        r_multiplier = np.array(np.bmat([[np.zeros((self.queries_stream.dim_, 1))],
                                         [e_col(self.queries_stream.dim_ * self.serv_stream.dim)]]),
                                dtype=float)
        return np.dot(stationary_probas[0], r_multiplier)[0][0]

    def calc_buffer_i_queries_j_nonprior(self, stationary_probas, i, j):
        """
        Performance characteristic: probability that buffer contains i queries, j of them are non-priority

        :param stationary_probas:
        :param i:
        :param j:
        :return:
        """

        mulW_M = self.queries_stream.dim_ * self.serv_stream.dim
        block0_size = int(mulW_M * np.sum([ncr(l + self.timer_stream.dim - 1,
                                               self.timer_stream.dim - 1) for l in range(j)]))
        block1_size = int(mulW_M * ncr(j + self.timer_stream.dim - 1,
                                       self.timer_stream.dim - 1))
        block2_size = int(mulW_M * np.sum([ncr(l + self.timer_stream.dim - 1,
                                               self.timer_stream.dim - 1) for l in range(j + 1, i + 1)]))
        r_multiplier = np.array(np.bmat([[np.zeros((block0_size, 1))],
                                         [e_col(block1_size)],
                                         [np.zeros((block2_size, 1))]]),
                                dtype=float)
        return np.dot(stationary_probas[i],
                      r_multiplier)[0][0]

    def calc_buffer_i_queries(self, stationary_probas, i):
        """
        Performance characteristic: probability that buffer contains i queries

        :param stationary_probas:
        :param i:
        :return:
        """
        return np.sum([self.calc_buffer_i_queries_j_nonprior(stationary_probas, i, j) for j in range(i + 1)])

    def calc_avg_buffer_queries_num(self, stationary_probas):
        """
        Performance characteristic: average queries in buffer number

        :param stationary_probas:
        :return:
        """

        return np.sum([i * self.calc_buffer_i_queries(stationary_probas, i) for i in range(1, self.N + 1)])

    def calc_avg_buffer_nonprior_queries_num(self, stationary_probas):
        """
        Performance characteristic: average non-priority queries in buffer number

        :param stationary_probas:
        :return:
        """
        return np.sum(
            [np.sum([j * self.calc_buffer_i_queries_j_nonprior(stationary_probas, i, j) for j in range(1, i + 1)]) for i
             in range(1, self.N + 1)])

    def calc_avg_buffer_prior_queries_num(self, stationary_probas):
        """
        Performance characteristic: average priority queries in buffer number

        :param stationary_probas:
        :return:
        """
        return self.calc_avg_buffer_queries_num(stationary_probas) - self.calc_avg_buffer_nonprior_queries_num(stationary_probas)

    def calc_sd_buffer_queries_num(self, stationary_probas):
        """
        Performance characteristic: standard deviation of queries in buffer number

        :param stationary_probas:
        :return:
        """
        L = self.calc_avg_buffer_queries_num(stationary_probas)

        sum = 0.
        for i in range(1, self.N + 1):
            sum += i * i * self.calc_buffer_i_queries(stationary_probas, i)
        return sqrt(sum - L * L)

    def calc_sd_buffer_prior_queries_num(self, stationary_probas):
        """
        Performance characteristic: standard deviation of priority queries in buffer number

        :param stationary_probas:
        :return:
        """

        L_prior = self.calc_avg_buffer_prior_queries_num(stationary_probas)

        phis = [None]
        for i in range(1, self.N + 1):
            phi = self.calc_buffer_i_queries_j_nonprior(stationary_probas, i, 0)
            for j in range(i + 1, self.N + 1):
                phi += self.calc_buffer_i_queries_j_nonprior(stationary_probas, j, j - i)
            phis.append(phi)

        sum = 0.
        for i in range(1, self.N + 1):
            sum += i * i * phis[i]

        return sqrt(sum - L_prior * L_prior)


    def calc_query_lost_p(self, stationary_probas):
        """
        Performance characteristic: probability that query is lost because of buffer is full

        :param stationary_probas:
        :return:
        """
        p_loss = np.dot(stationary_probas[0],
                        np.array(np.bmat([[np.zeros((self.queries_stream.dim_, self.serv_stream.dim))],
                                          [kron(e_col(self.queries_stream.dim_),
                                                self.I_M)]]),
                                 dtype=float))  # checked
        r_sum = np.dot(stationary_probas[1],
                       self.calI_2[1])
        for i in range(2, self.N + 1):
            r_sum += np.dot(stationary_probas[i],
                            self.calI_2[i])

        p_loss = p_loss + r_sum
        p_loss = np.dot(p_loss, self.serv_stream.repres_matr_0)
        p_loss = 1 - (1 / self.queries_stream.avg_intensity) * p_loss[0][0]

        return p_loss

    def __calc_query_lost_ps_buffer_full_D(self, stationary_probas, matrDs):
        l_sum = (- self.N - 1) * r_multiply_e(self.queries_stream.matrD_0)
        for k in range(1, min(self.N + 1, self.n) + 1):
            l_sum += (k - self.N - 1) * r_multiply_e(matrDs[k])

        l_sum = np.dot(np.dot(stationary_probas[0],
                              np.array(np.bmat([[self.I_W],
                                                [kron(self.O_W,
                                                      e_col(self.serv_stream.dim))]]),
                                       dtype=float)),
                       l_sum)

        c_sum = (- self.N) * r_multiply_e(self.queries_stream.matrD_0)
        for k in range(1, min(self.N, self.n) + 1):
            c_sum += (k - self.N) * r_multiply_e(matrDs[k])

        c_sum = np.dot(np.dot(stationary_probas[0],
                              np.array(np.bmat([[self.O_W],
                                                [kron(self.I_W,
                                                      e_col(self.serv_stream.dim))]]),
                                       dtype=float)),
                       c_sum)
        p_loss = l_sum + c_sum
        if self.N > 1:
            r_sum_1 = np.dot(stationary_probas[1],
                             self.calI_1[1])
            r_sum_2 = (- self.N + 1) * r_multiply_e(self.queries_stream.matrD_0)
            for k in range(1, min(self.N, self.n + 1)):
                r_sum_2 += (k - self.N + 1) * r_multiply_e(matrDs[k])

            r_sum_full = np.dot(r_sum_1, r_sum_2)

            for i in range(2, self.N):
                r_sum_1 = np.dot(stationary_probas[i],
                                 self.calI_1[i])
                r_sum_2 = (- self.N + i) * r_multiply_e(self.queries_stream.matrD_0)
                for k in range(1, min(self.N - i, self.n) + 1):
                    r_sum_2 += (k - self.N + i) * r_multiply_e(matrDs[k])

                r_sum_full += np.dot(r_sum_1, r_sum_2)

            p_loss += r_sum_full

        return p_loss[0][0]

    def __calc_query_lost_ps_buffer_part_D(self, stationary_probas, matrDs):
        sum1 = np.zeros((self.queries_stream.dim_, 1))
        for k in range(1, min(self.N + 1, self.n) + 1):
            sum1 += k * r_multiply_e(matrDs[k])
        for k in range(self.N + 2, self.n + 1):
            sum1 += (self.N + 1) * r_multiply_e(matrDs[k])

        sum1 = np.dot(np.dot(stationary_probas[0],
                             np.array(np.bmat([[self.I_W],
                                               [kron(self.O_W,
                                                     self.e_M)
                                                ]]))),
                      sum1)

        sum2 = np.zeros((self.queries_stream.dim_, 1))
        for k in range(1, min(self.N, self.n) + 1):
            sum2 += k * r_multiply_e(matrDs[k])

        for k in range(self.N + 1, self.n + 1):
            sum2 += self.N * r_multiply_e(matrDs[k])

        sum2 = np.dot(np.dot(stationary_probas[0],
                             np.array(np.bmat([[self.O_W],
                                               [kron(self.I_W,
                                                     self.e_M)
                                                ]]))),
                      sum2)

        sum3 = np.zeros((1, 1))
        for i in range(1, self.N):
            sump3 = np.zeros((self.queries_stream.dim_, 1))
            for k in range(1, min(self.N - i, self.n) + 1):
                sump3 += k * r_multiply_e(matrDs[k])

            for k in range(self.N - i + 1, self.n + 1):
                sump3 += (self.N - i) * r_multiply_e(matrDs[k])

            sum3 += np.dot(np.dot(stationary_probas[i],
                                  self.calI_1[i]),
                           sump3)

        return (sum1 + sum2 + sum3)[0][0]

    def calc_query_lost_ps_buffer_full(self, stationary_probas):
        """
        Performance characteristic: probability that query (priority, non-priority, any) is lost because of buffer is full

        :param stationary_probas:
        :return: ([probability priority query is lost, probability non-priority query is lost], probability any query is lost)
        """
        P_losses = []
        for i in range(2):
            P_losses.append(1 - (1 / self.queries_stream.avg_intensity_t[i]) * self.__calc_query_lost_ps_buffer_part_D(
                stationary_probas,
                self.queries_stream.transition_matrices[i]))

        matrDks = [self.queries_stream.transition_matrices[0][i] + self.queries_stream.transition_matrices[1][i] for i
                   in range(self.n + 1)]

        p_loss_alg = 1 - (1 / self.queries_stream.avg_intensity) * self.__calc_query_lost_ps_buffer_full_D(
            stationary_probas, matrDks)

        return P_losses, p_loss_alg

    def calc_nonprior_query_lost_timer(self, stationary_probas):
        """
        Performance characteristic: probability that non-priority query is lost due to timer click

        :param stationary_probas:
        :return:
        """
        p_loss = np.dot(stationary_probas[1],
                        self.calI_L[1])
        for i in range(2, self.N + 1):
            p_loss += np.dot(stationary_probas[i],
                             self.calI_L[i])
        return (self.p_hp / self.queries_stream.avg_intensity_t[1]) * p_loss[0][0]

    def check_by_theta(self, stationary_probas):
        sum = np.dot(stationary_probas[0], np.bmat([[self.I_W], [kron(self.I_W, e_col(self.serv_stream.dim))]]))

        for i in range(1, self.N + 1):
            sum += np.dot(stationary_probas[i],
                          np.bmat([[kron(kron(self.I_W,
                                              e_col(self.serv_stream.dim)),
                                         e_col(ncr(self.timer_stream.dim - 1 + j,
                                                   self.timer_stream.dim - 1)))] for j in range(i + 1)]))
        return sum

    def calc_pij(self, stationary_probas, i, j):
        return np.dot(stationary_probas[i],
                      np.bmat([[np.zeros((int(self.queries_stream.dim_ * self.serv_stream.dim * np.sum(
                          [ncr(k + self.timer_stream.dim - 1,
                               self.timer_stream.dim - 1)
                           for k in range(j)])),
                                          int(self.queries_stream.dim_ * self.serv_stream.dim * ncr(
                                              j + self.timer_stream.dim - 1,
                                              self.timer_stream.dim - 1))))],
                               [np.eye(self.queries_stream.dim_ * self.serv_stream.dim * ncr(
                                   j + self.timer_stream.dim - 1,
                                   self.timer_stream.dim - 1))],
                               [np.zeros((int(self.queries_stream.dim_ * self.serv_stream.dim * np.sum(
                                   [ncr(k + self.timer_stream.dim - 1,
                                        self.timer_stream.dim - 1)
                                    for k in range(j + 1, i + 1)])),
                                          int(self.queries_stream.dim_ * self.serv_stream.dim * ncr(
                                              j + self.timer_stream.dim - 1,
                                              self.timer_stream.dim - 1))))]]))

    def calcMatrAs(self):
        matrAs = []
        for i in range(self.N):
            S_block = la.block_diag(*(self.serv_stream.repres_matr for _ in range(i + 1)))
            S0b_block = la.block_diag(*(copy.deepcopy(self.S_0xBeta) for _ in range(1, i + 1)))
            matrA = copy.deepcopy(S_block)
            if i > 0:
                S0b_block = np.concatenate(
                    (np.zeros((S0b_block.shape[0], self.serv_stream.repres_matr.shape[1])), S0b_block), axis=1)
                S0b_block = np.concatenate(
                    (S0b_block, np.zeros((self.serv_stream.repres_matr.shape[0], S0b_block.shape[1]))), axis=0)
                matrA += S0b_block
            matrAs.append(matrA)
        return matrAs

    def func_tilde_W_1(self, stationary_probas, t):
        """
        Probability that query came into QS as a priority and its waiting time < t
        """
        matrAs = self.matrAs

        sum1 = r_multiply_e(self.queries_stream.transition_matrices[0][1])
        for k in range(2, self.n + 1):
            rmul1 = 1.
            for j in range(2, min(self.N + 1, k) + 1):
                beta_temp = np.concatenate(
                    (self.serv_stream.repres_vect, np.zeros((1, (j - 2) * self.serv_stream.dim))), axis=1)
                exp_temp = la.expm(matrAs[j - 2] * t)
                rmul1 += np.dot(np.dot(beta_temp,
                                       (np.eye(exp_temp.shape[0]) - exp_temp)),
                                e_col((j - 1) * self.serv_stream.dim))[0][0]
            sum1 += r_multiply_e(self.queries_stream.transition_matrices[0][k]) * rmul1

        sum1 = np.dot(np.dot(stationary_probas[0],
                             np.array(np.bmat([[self.I_W],
                                               [np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                                                          self.queries_stream.dim_))]]))),
                      sum1)

        sum2 = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim, 1))
        for k in range(1, self.n + 1):
            rmul2 = np.zeros((self.serv_stream.dim, 1))
            for j in range(1, min(self.N, k) + 1):
                mul1 = np.concatenate((self.I_M, np.zeros((self.serv_stream.dim, (j - 1) * self.serv_stream.dim))),
                                      axis=1)
                # exp_temp2 = m_exp(matrAs[j - 1], t)
                exp_temp2 = la.expm(matrAs[j - 1] * t)
                rmul2 += np.dot(np.dot(mul1, np.eye(exp_temp2.shape[0]) - exp_temp2),
                                e_col(j * self.serv_stream.dim))
            sum2 += np.dot(kron(r_multiply_e(self.queries_stream.transition_matrices[0][k]), self.I_M), rmul2)

        sum2 = np.dot(np.dot(stationary_probas[0],
                             np.array(np.bmat([[np.zeros((self.queries_stream.dim_,
                                                          self.queries_stream.dim_ * self.serv_stream.dim))],
                                               [self.I_WM]]))),
                      sum2)

        sum3 = np.zeros(sum1.shape)
        for i in range(1, self.N):
            for j in range(i + 1):
                sump3 = 0.
                for k in range(1, self.n + 1):
                    rmul3 = np.zeros((self.serv_stream.dim, 1))
                    for el in range(1, min(self.N - i, k) + 1):
                        mul1 = np.concatenate(
                            (self.I_M, np.zeros((self.serv_stream.dim, (i - j + el - 1) * self.serv_stream.dim))),
                            axis=1)
                        # exp_temp3 = m_exp(matrAs[i - j + el - 1], t)
                        exp_temp3 = la.expm(matrAs[i - j + el - 1] * t)
                        rmul3 += np.dot(np.dot(mul1,
                                               np.eye(exp_temp3.shape[0]) - exp_temp3),
                                        e_col((i - j + el) * self.serv_stream.dim))

                    sump3 += np.dot(kron(kron(r_multiply_e(self.queries_stream.transition_matrices[0][k]),
                                              self.I_M),
                                         e_col(ncr(j + self.timer_stream.dim - 1,
                                                   self.timer_stream.dim - 1))), rmul3)
                sum3 += np.dot(self.calc_pij(stationary_probas, i, j),
                               sump3)

        prob = (1 / self.queries_stream.avg_intensity) * (sum1 + sum2 + sum3)
        return prob

    def func_tilde_W_2(self, stationary_probas, t):
        """
        Probability that query came into QS as a non-priority, became priority and its waiting time < t

        :param stationary_probas:
        :param t:
        :return:
        """
        matrAs = self.matrAs

        sum = 0.
        for i in range(1, self.N + 1):
            for j in range(1, i + 1):
                mul1 = self.calc_pij(stationary_probas, i, j)
                mul2 = kron(kron(e_col(self.queries_stream.dim_),
                                 self.I_M),
                            r_multiply_e(self.ramatrL[self.N - i + j][self.N - i]))
                mul3 = np.concatenate(
                    (self.I_M, np.zeros((self.serv_stream.dim, (i - j) * self.serv_stream.dim))),
                    axis=1)
                # temp_exp = m_exp(matrAs[i - j], t)
                temp_exp = la.expm(matrAs[i - j] * t)

                mul4 = np.eye(temp_exp.shape[0]) - temp_exp

                mul5 = e_col((i - j + 1) * self.serv_stream.dim)

                sum += np.dot(np.dot(np.dot(np.dot(mul1,
                                                   mul2),
                                            mul3),
                                     mul4),
                              mul5)[0][0]

        sumGamma = 0.
        for i in range(1, self.N + 1):
            for j in range(1, i + 1):
                sumGamma += np.dot(self.calc_pij(stationary_probas, i, j),
                                   kron(self.e_WM,
                                        r_multiply_e(self.ramatrL[self.N - i + j][self.N - i])))[0][0]

        sum = (1. - self.p_hp) / sumGamma * sum
        return np.array(sum, dtype=float)

    def func_W_1(self, stationary_probas, t):
        return (self.func_tilde_W_1(stationary_probas, t) / self.W_1_inf)[0][0]

    def func_W_2(self, stationary_probas, t):
        return (self.func_tilde_W_2(stationary_probas, t) / self.W_2_inf)[0][0]

    def avg_wait_time_1(self, stationary_probas):
        """
        Average waiting time for initially priority queries

        :param stationary_probas:
        :return:
        """
        b1 = 1 / self.serv_stream.avg_intensity
        revSe = r_multiply_e(la.inv(-self.serv_stream.repres_matr))

        sum1 = r_multiply_e(self.queries_stream.transition_matrices[0][1])
        for k in range(2, self.n + 1):
            sum1 += r_multiply_e(self.queries_stream.transition_matrices[0][k]) * sum(
                [(j - 1) * b1 for j in range(2, min(self.N + 1, k) + 1)])

        sum1 = np.dot(np.dot(stationary_probas[0],
                             np.array(np.bmat([[self.I_W],
                                               [np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                                                          self.queries_stream.dim_))]]))
                             ),
                      sum1)

        sum2 = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim, 1))
        for k in range(1, self.n + 1):
            sump2 = np.zeros((self.serv_stream.dim, 1))
            for j in range(1, min(self.N, k) + 1):
                sump2 += revSe + self.e_M * (j - 1) * b1
            sum2 += np.dot(kron(r_multiply_e(self.queries_stream.transition_matrices[0][k]),
                                self.I_M),
                           sump2)

        sum2 = np.dot(np.dot(stationary_probas[0],
                             np.array(np.bmat([[np.zeros((self.queries_stream.dim_,
                                                          self.queries_stream.dim_ * self.serv_stream.dim))],
                                               [self.I_WM]]))),
                      sum2)

        sum3 = np.zeros(sum1.shape)
        for i in range(1, self.N):
            for j in range(i + 1):
                sump3 = 0.
                for k in range(1, self.n + 1):
                    rmul3 = np.zeros((self.serv_stream.dim, 1))
                    for el in range(1, min(self.N - i, k) + 1):
                        rmul3 += revSe + self.e_M * (i - j + el - 1) * b1

                    sump3 += np.dot(kron(kron(r_multiply_e(self.queries_stream.transition_matrices[0][k]),
                                              self.I_M),
                                         e_col(ncr(j + self.timer_stream.dim - 1,
                                                   self.timer_stream.dim - 1))), rmul3)
                sum3 += np.dot(self.calc_pij(stationary_probas, i, j),
                               sump3)
        return 1 / (self.queries_stream.avg_intensity * self.W_1_inf) * (sum1 + sum2 + sum3)[0][0]

    def avg_wait_time_2(self, stationary_probas):
        """
        Average waiting time for non-priority queries that became priority

        :param stationary_probas:
        :return:
        """
        revSe = r_multiply_e(la.inv(-self.serv_stream.repres_matr))
        b1 = 1 / self.serv_stream.avg_intensity

        sum = 0.
        for i in range(1, self.N + 1):
            for j in range(1, i + 1):
                mul1 = self.calc_pij(stationary_probas, i, j)
                mul2 = kron(kron(e_col(self.queries_stream.dim_),
                                 self.I_M),
                            r_multiply_e(self.ramatrL[self.N - i + j][self.N - i]))
                mul3 = revSe + self.e_M * (i - j) * b1

                sum += np.dot(np.dot(mul1,
                                     mul2),
                              mul3)[0][0]

        sumGamma = 0.
        for i in range(1, self.N + 1):
            for j in range(1, i + 1):
                sumGamma += np.dot(self.calc_pij(stationary_probas, i, j),
                                   kron(self.e_WM,
                                        r_multiply_e(self.ramatrL[self.N - i + j][self.N - i])))[0][0]

        sum = (1. - self.p_hp) / (sumGamma * self.W_2_inf) * sum
        return np.array(sum, dtype=float)[0][0]

    def calc_characteristics(self, verbose=False):
        stationary_probas = self.calc_stationary_probas(verbose)

        system_empty_proba = self.calc_system_empty_proba(stationary_probas)
        if verbose:
            print("p_0 =", system_empty_proba)

        system_single_query_proba = self.calc_system_single_query_proba(stationary_probas)
        if verbose:
            print("p_1 =", system_single_query_proba)

            for i in range(1, self.N + 1):
                print("p_buf_" + str(i), '=', self.calc_buffer_i_queries(stationary_probas, i))

        avg_buffer_queries_num = self.calc_avg_buffer_queries_num(stationary_probas)
        if verbose:
            print("L_buf =", avg_buffer_queries_num)

        avg_buffer_nonprior_num = self.calc_avg_buffer_nonprior_queries_num(stationary_probas)
        if verbose:
            print("q_j =", avg_buffer_nonprior_num)

        avg_buffer_prior_num = self.calc_avg_buffer_prior_queries_num(stationary_probas)
        if verbose:
            print("L_buf^prior =", avg_buffer_prior_num)

        sd_buffer_queries_num = self.calc_sd_buffer_queries_num(stationary_probas)
        if verbose:
            print(r"\sigma =", sd_buffer_queries_num)

        sd_buffer_prior_queries_num = self.calc_sd_buffer_prior_queries_num(stationary_probas)
        if verbose:
            print(r"\sigma_prior =", sd_buffer_prior_queries_num)

        p_loss = self.calc_query_lost_p(stationary_probas)
        if verbose:
            print("P_loss =", p_loss)

        p_losses, p_loss_alg = self.calc_query_lost_ps_buffer_full(stationary_probas)
        if verbose:
            print("P_losses =", p_losses)
            print("P_loss_alg =", p_loss_alg)

        p_loss_imp = self.calc_nonprior_query_lost_timer(stationary_probas)
        if verbose:
            print("P_loss_imp =", p_loss_imp)

        # p_loss_alg = self.calc_query_lost_p_alg(stationary_probas)
        # print("P_loss_alg =", p_loss_alg)

        check_theta = self.check_by_theta(stationary_probas)
        print("theta =", check_theta, " -- theta_true =", self.queries_stream.theta)

        f1t1 = self.func_W_1(stationary_probas, 0.59)
        f2t1 = self.func_W_2(stationary_probas, 0.59)

        print("F_1(1) =", f1t1)
        print("F_2(1) =", f2t1)

        w1 = self.avg_wait_time_1(stationary_probas)
        print("w1 =", w1)

        w2 = self.avg_wait_time_2(stationary_probas)
        print("w2 =", w2)

        self.check_by_Q_multiplying(stationary_probas)

    def print_generator(self, as_latex=True):
        for row_num, block_row in enumerate(self.generator):
            print('Row', row_num)
            for block_num, block in enumerate(block_row):
                print(block_num, block)


if __name__ == '__main__':
    test_data = test.Bmmap3PoissonPhPh()

    qs = TwoPrioritiesQueueingSystem(test_data, verbose=True)
    qs.queries_stream.print_characteristics()
    qs.serv_stream.print_characteristics(matrix_name='S', vector_name='beta')
    qs.timer_stream.print_characteristics(matrix_name='Г', vector_name='gamma')
    qs.print_generator()
    qs.calc_characteristics(True)
    print(qs.calc_stationary_probas_classic())
