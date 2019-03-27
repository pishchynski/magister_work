import sys

sys.path.append("../")

from streams import *

# import experiments_data.BMMAP3_04_PH_PH as test
import experiments_data.MMAP_Poisson_PH_PH as test

np.set_printoptions(threshold=np.inf, suppress=True, formatter={'float': '{: 0.8f}'.format}, linewidth=75)


class ClassicTwoPrioritiesQueueingSystem:
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
                                          q=experiment_data.test_q,
                                          n=experiment_data.test_n,
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

        self.S_0xBeta = np.dot(self.serv_stream.repres_matr_0,
                               self.serv_stream.repres_vect)

        self.p_hp = experiment_data.p_hp  # probability of the query to leave system after the timer's up
        self.n = experiment_data.test_n  # number of D_i matrices
        self.N = experiment_data.test_N  # buffer capacity

        self.generator = None

        self.recalculate_generator(verbose=verbose)

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

        print("Generator recalculating")

        matrQ_0k = self._calc_Q_0k()
        matrQ_iiprev = self._calc_Q_iiprev()
        matrQ_ii = self._calc_Q_ii()
        matrQ_iik = self._calc_matrQ_iik()
        matrQ_iN = self._calc_matrQ_iN()

        self.check_generator(matrQ_0k, matrQ_iiprev, matrQ_ii, matrQ_iik, matrQ_iN)
        self.finalize_generator(matrQ_0k, matrQ_iiprev, matrQ_ii, matrQ_iik, matrQ_iN)

        print("Generator recalculated:")

        if verbose:
            for row_num, block_row in enumerate(self.generator):
                print('Row', row_num)
                for block_num, block in enumerate(block_row):
                    print('Q_' + str(row_num) + ',' + str(block_num), block)

    def _calc_Q_00(self):
        """
        Calculates Q_{0,0} block of the infinitesimal generator.

        :return: numpy.array
        """

        print("Calculating Q_00")

        block00 = copy.deepcopy(self.queries_stream.matrD_0)
        block01 = kron(self.queries_stream.transition_matrices[0][1] + self.queries_stream.transition_matrices[1][1],
                       self.serv_stream.repres_vect)
        block10 = kron(self.I_W,
                       self.serv_stream.repres_matr_0)
        block11 = kronsum(self.queries_stream.matrD_0,
                          self.serv_stream.repres_matr)

        matrQ_00 = np.bmat([[block00, block01],
                            [block10, block11]])

        print("Q_00 calculated")

        return np.array(matrQ_00, dtype=float)

    def _calc_Q_0k(self):
        """
        Calculates matrices Q_{0, k}, k = [0, N]

        :return: list of np.arrays with Q_{0, k}
        """
        print("Calculating Q_0k")
        matrQ_0k = [self._calc_Q_00()]

        for k in range(1, self.N):
            # block00 = np.zeros(self.queries_stream.transition_matrices[0][1].shape)
            if k + 1 > self.n:
                block00 = kron(np.zeros(self.queries_stream.transition_matrices[0][1].shape),
                               # this shape suits because all D_{i}^{1} have same shape
                               self.serv_stream.repres_vect)
            else:
                block00 = kron(self.queries_stream.transition_matrices[0][k + 1],
                               self.serv_stream.repres_vect)

            # block10 = np.zeros(self.queries_stream.transition_matrices[0][1].shape)
            if k > self.n:
                block10 = kron(np.zeros(self.queries_stream.transition_matrices[0][1].shape),
                               # this shape suits because all D_{i}^{1} have same shape
                               self.I_M)
            else:
                block10 = kron(self.queries_stream.transition_matrices[0][k],
                               self.I_M)

            blocks0k = [block00]
            blocks1k = [block10]

            if k > 1:
                temp_block = np.zeros((self.queries_stream.dim_,
                                       self.queries_stream.dim_ * self.serv_stream.dim * self.timer_stream.dim * (
                                               self.timer_stream.dim ** (k - 1) - 1) / (self.timer_stream.dim - 1)))
                blocks0k.append(temp_block)

                temp_block = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                                       self.queries_stream.dim_ * self.serv_stream.dim * self.timer_stream.dim * (
                                               self.timer_stream.dim ** (k - 1) - 1) / (self.timer_stream.dim - 1)))
                blocks1k.append(temp_block)

            if k + 1 > self.n:
                last_block0 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][1].shape),
                                        self.serv_stream.repres_vect),
                                   kronpow(self.timer_stream.repres_vect, k))
            else:
                last_block0 = kron(kron(self.queries_stream.transition_matrices[1][k + 1],
                                        self.serv_stream.repres_vect),
                                   kronpow(self.timer_stream.repres_vect, k))

            # last_block1 = np.zeros((1, 1))
            if k > self.n:
                last_block1 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][1].shape),
                                        self.I_M),
                                   kronpow(self.timer_stream.repres_vect, k))
            else:
                last_block1 = kron(kron(self.queries_stream.transition_matrices[1][k],
                                        self.I_M),
                                   kronpow(self.timer_stream.repres_vect, k))

            blocks0k.append(last_block0)
            blocks1k.append(last_block1)

            temp_matr = np.array(np.bmat([blocks0k,
                                          blocks1k]),
                                 dtype=float)

            matrQ_0k.append(temp_matr)

        matrQ_0k.append(self._calc_Q_0N())

        print("Q_0k calculated")

        return matrQ_0k

    def _calc_Q_0N(self):
        """
        Calculates Q_{0,N} block of the infinitesimal generator

        :return:
        """

        print("Calculating Q_0N")

        # block00 = np.zeros((1, 1))
        if self.N + 1 > self.n:
            block00 = kron(np.zeros(self.queries_stream.transition_matrices[0][1].shape),
                           self.serv_stream.repres_vect)
        else:
            block00 = kron(self.queries_stream.transition_matrices[0][self.N + 1],
                           self.serv_stream.repres_vect)
            for i in range(self.N + 2, self.n + 1):
                block00 += kron(self.queries_stream.transition_matrices[0][i],
                                self.serv_stream.repres_vect)

        # block10 = np.zeros((1, 1))
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
                                   self.queries_stream.dim_ * self.serv_stream.dim * self.timer_stream.dim * int(
                                       (self.timer_stream.dim ** (self.N - 1) - 1) / (self.timer_stream.dim - 1))))
            blocks0k.append(copy.deepcopy(temp_block))

            temp_block = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                                   self.queries_stream.dim_ * self.serv_stream.dim * self.timer_stream.dim * int(
                                       (self.timer_stream.dim ** (self.N - 1) - 1) / (self.timer_stream.dim - 1))))
            blocks1k.append(copy.deepcopy(temp_block))

        if self.N + 1 > self.n:
            last_block0 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][1].shape),
                                    self.serv_stream.repres_vect),
                               kronpow(self.timer_stream.repres_vect, self.N))
        else:
            last_block0 = kron(kron(self.queries_stream.transition_matrices[1][self.N + 1],
                                    self.serv_stream.repres_vect),
                               kronpow(self.timer_stream.repres_vect, self.N))
            for i in range(self.N + 2, self.n + 1):
                last_block0 += kron(kron(self.queries_stream.transition_matrices[1][i],
                                         self.serv_stream.repres_vect),
                                    kronpow(self.timer_stream.repres_vect, self.N))

        # last_block1 = np.zeros((1, 1))
        if self.N > self.n:
            last_block1 = kron(kron(np.zeros(self.queries_stream.transition_matrices[1][1].shape),
                                    self.I_M),
                               kronpow(self.timer_stream.repres_vect, self.N))
        else:
            last_block1 = kron(kron(self.queries_stream.transition_matrices[1][self.N],
                                    self.I_M),
                               kronpow(self.timer_stream.repres_vect, self.N))
            for i in range(self.N + 1, self.n + 1):
                last_block1 += kron(kron(self.queries_stream.transition_matrices[1][i],
                                         self.I_M),
                                    kronpow(self.timer_stream.repres_vect, self.N))

        blocks0k.append(last_block0)
        blocks1k.append(last_block1)

        print("Q_0N calculated")

        return np.array(np.bmat([blocks0k,
                                 blocks1k]),
                        dtype=float)

    def _calc_Q_10(self):
        """
        Calculates Q_{1,0} block of the infinitesimal generator

        :return:
        """

        print("Calculating Q_10")

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

        print("Q_10 calculated")

        return np.array(np.bmat([[block00, block01],
                                 [block10, block11]]),
                        dtype=float)

    def _calc_Q_iiprev(self):
        """
        Calculates Q_{i,i-1} blocks of the infinitesimal generator

        :return:
        """

        print("Calculating Q_{i, i - 1}")

        matrQ_iiprev = [None, self._calc_Q_10()]
        for i in range(2, self.N + 1):
            blocks0 = [kron(self.I_W,
                            self.S_0xBeta)]
            blocks1 = [kron(self.p_hp * self.I_WM,
                            self.timer_stream.repres_matr_0)]

            for j in range(1, i - 1):
                blocks0.append(kron(kron(self.I_W,
                                         self.S_0xBeta),
                                    np.eye(self.timer_stream.dim ** j)))
                blocks1.append(kron(self.p_hp * self.I_WM,
                                    kronsumpow(self.timer_stream.repres_matr_0, j + 1)))

            blocks0.append(kron(kron(self.I_W,
                                     self.S_0xBeta),
                                np.eye(self.timer_stream.dim ** (i - 1))))

            last_block1 = kron(kron(self.I_W, self.S_0xBeta),
                               (1 / i) * kronsumpow(e_col(self.timer_stream.dim), i))
            last_block1 += kron(self.p_hp * self.I_WM,
                                kronsumpow(self.timer_stream.repres_matr_0, i))
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
            cur_matr_blocks = tuple(
                (kron(kronsum(self.queries_stream.matrD_0 if i != self.N else self.queries_stream.matrD_1_,
                              self.serv_stream.repres_matr),
                      np.eye(self.timer_stream.dim ** j)) + kron(self.I_WM,
                                                                 kronsumpow(self.timer_stream.repres_matr, j)) for j in
                 range(i + 1)))
            block_shapes = [block.shape for block in cur_matr_blocks]
            cur_matr = la.block_diag(*cur_matr_blocks)

            udiag_matr = np.zeros(cur_matr.shape)
            udiag_blocks = [kron(self.I_WM, kronsumpow(self.timer_stream.repres_matr_0, j)) for j in range(1, i + 1)]

            n_pos = 0
            m_pos = 0
            for udiag_block, shape in zip(udiag_blocks, block_shapes):
                m_pos += shape[0]
                copy_matrix_block(udiag_matr, udiag_block, m_pos, n_pos)
                n_pos += shape[1]

            cur_matr += (1 - self.p_hp) * udiag_matr

            matrQ_ii.append(cur_matr)

        print("Q_{i, i} calculated")

        return matrQ_ii

    def _calc_matrQ_iik(self):
        """
        Calculates Q_{i,i+k} blocks of the infinitesimal generator

        :return:
        """

        print("Calculating Q_{i, i + k}")

        matrQ_iik = [None]
        for i in range(1, self.N):

            cur_zero_matr = la.block_diag(*(kron(np.zeros(self.queries_stream.transition_matrices[0][1].shape),
                                                 np.eye(self.serv_stream.dim * (self.timer_stream.dim ** j)))
                                            for j in range(i + 1)))

            matrQ_ii_row = []
            for k in range(1, self.N - i):
                if k <= self.n:
                    cur_matr = la.block_diag(*(kron(self.queries_stream.transition_matrices[0][k],
                                                    np.eye(self.serv_stream.dim * (self.timer_stream.dim ** j)))
                                               for j in range(i + 1)))

                    zero_matr = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim *
                                          (self.timer_stream.dim ** (i + 1) - 1) / (self.timer_stream.dim - 1),
                                          self.queries_stream.dim_ * self.serv_stream.dim * self.timer_stream.dim ** (
                                                  i + 1) * (self.timer_stream.dim ** k - 1) / (
                                                  self.timer_stream.dim - 1)
                                          ))
                    cur_matr = np.concatenate((cur_matr, zero_matr), axis=1)

                    temp_matr = la.block_diag(*(kron(kron(self.queries_stream.transition_matrices[1][k],
                                                          np.eye(self.serv_stream.dim * (self.timer_stream.dim ** j))),
                                                     kronpow(self.timer_stream.repres_vect, k))
                                                for j in range(i + 1)))
                else:

                    zero_matr = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim *
                                          (self.timer_stream.dim ** (i + 1) - 1) / (self.timer_stream.dim - 1),
                                          self.queries_stream.dim_ * self.serv_stream.dim * self.timer_stream.dim ** (
                                                  i + 1) * (self.timer_stream.dim ** k - 1) / (
                                                  self.timer_stream.dim - 1)
                                          ))

                    cur_matr = np.concatenate((cur_zero_matr, zero_matr), axis=1)

                    temp_matr = la.block_diag(*(kron(kron(np.zeros(self.queries_stream.transition_matrices[1][1].shape),
                                                          np.eye(self.serv_stream.dim * (self.timer_stream.dim ** j))),
                                                     kronpow(self.timer_stream.repres_vect, k))
                                                for j in range(i + 1)))

                zero_matr = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim *
                                      (self.timer_stream.dim ** (i + 1) - 1) / (self.timer_stream.dim - 1),
                                      self.queries_stream.dim_ * self.serv_stream.dim * (
                                              self.timer_stream.dim ** k - 1) / (self.timer_stream.dim - 1)
                                      ))
                cur_matr += np.concatenate((zero_matr, temp_matr), axis=1)

                matrQ_ii_row.append(cur_matr)

            matrQ_iik.append(matrQ_ii_row)

        print("Q_{i, i + k} calculated")

        return matrQ_iik

    def _calc_matrQ_iN(self):
        """
        Calculates matrices Q_{i, N}, i = [1, N - 1]

        :return: list of np.arrays with Q_{i, N}
        """
        print("Calculating Q_{i, N}")

        matrQ_iN = [None]
        for i in range(1, self.N):
            matrD1_sum = np.zeros(self.queries_stream.transition_matrices[0][-1].shape)
            matrD2_sum = np.zeros(self.queries_stream.transition_matrices[1][-1].shape)

            if self.N - i <= self.n:
                matrD1_sum = self.queries_stream.transition_matrices[0][self.N - i]
                matrD2_sum = self.queries_stream.transition_matrices[1][self.N - i]
            for k in range(self.N - i + 1, self.n + 1):
                matrD1_sum += self.queries_stream.transition_matrices[0][k]
                matrD2_sum += self.queries_stream.transition_matrices[1][k]

            cur_matr = la.block_diag(*(kron(matrD1_sum, np.eye(self.serv_stream.dim * (self.timer_stream.dim ** j)))
                                       for j in range(i + 1)))

            zero_matr = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim * int(
                (self.timer_stream.dim ** (i + 1) - 1) / (self.timer_stream.dim - 1)),
                                  self.queries_stream.dim_ * self.serv_stream.dim * int(
                                      (self.timer_stream.dim ** (i + 1)) * (
                                              self.timer_stream.dim ** (self.N - i) - 1) / (
                                              self.timer_stream.dim - 1))))

            cur_matr = np.concatenate((cur_matr, zero_matr), axis=1)

            temp_matr = la.block_diag(
                *(kron(kron(matrD2_sum, np.eye(self.serv_stream.dim * self.timer_stream.dim ** j)),
                       kronsumpow(self.timer_stream.repres_vect, self.N - i))
                  for j in range(i + 1)))

            zero_matr = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim * int((
                                                                                                self.timer_stream.dim ** (
                                                                                                i + 1) - 1) / (
                                                                                                self.timer_stream.dim - 1)),
                                  self.queries_stream.dim_ * self.serv_stream.dim * int((
                                                                                                self.timer_stream.dim ** (
                                                                                                self.N - i) - 1) / (
                                                                                                self.timer_stream.dim - 1))))

            cur_matr += np.concatenate((zero_matr, temp_matr), axis=1)

            matrQ_iN.append(cur_matr)

        print("Q_{i, N} calculated")

        return matrQ_iN

    def check_generator(self, matrQ_0k, matrQ_iiprev, matrQ_ii, matrQ_iik, matrQ_iN):
        """
        Checks first block-row for by-row sum equality to zero

        :param matrQ_0k:
        :param matrQ_iiprev:
        :param matrQ_ii:
        :param matrQ_iik:
        :param matrQ_iN:
        """
        # First block-row
        print("Generator check started")

        print("Checking Q_0k")
        for i in range(self.queries_stream.dim_ + self.queries_stream.dim_ * self.serv_stream.dim):
            temp = 0
            for matr in matrQ_0k:
                temp += np.sum(matr[i])
            if abs(temp) > 10 ** (-6):
                print("Line", i, "=", temp)

        print("Checked Q_0k")

        # Other block-rows except Nth
        print("Checking Q_ij")
        for i in range(1, self.N + 1):
            # Iterating block-rows
            temp = np.sum(matrQ_iiprev[i], axis=1)
            temp += np.sum(matrQ_ii[i], axis=1)
            if i != self.N:
                for block in matrQ_iik[i]:
                    temp += np.sum(block, axis=1)
                temp += np.sum(matrQ_iN[i], axis=1)
            for subrow_num, row_sum in enumerate(temp):
                if abs(row_sum) > 10 ** (-6):
                    print("Line", i, "Subline", subrow_num, "=", temp)
        print("Checked Q_ij")

        print("Generator checked")

    def finalize_generator(self, matrQ_0k, matrQ_iiprev, matrQ_ii, matrQ_iik, matrQ_iN):
        print("Finalizing generator")

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

        print("G calculated")

        return matrG

    def _calc_matrQover(self, matrG):
        print("Calculating Q_over")

        matrQ = self.generator
        matrQover = [[None for _ in range(self.N)] + [copy.deepcopy(matrQ[i][self.N])] for i in range(self.N + 1)]
        for k in range(self.N - 1, -1, -1):
            for i in range(k + 1):
                matrQover[i][k] = matrQ[i][k] + np.dot(matrQover[i][k + 1], matrG[k])

        print("Q_over calculated")

        return matrQover

    def _calc_matrF(self, matrQover):
        print("Calculating F")

        matrF = [None]
        for i in range(1, self.N + 1):
            tempF = copy.deepcopy(matrQover[0][i])
            for j in range(1, i):
                tempF = tempF + np.dot(matrF[j], matrQover[j][i])
            tempF = np.dot(tempF, la.inv(-matrQover[i][i]))
            matrF.append(tempF)

        print("F calculated")

        return matrF

    def _calc_p0(self, matrF, matrQover):
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

        print('\nChecking p_0 * Qover_{0, 0}')
        print(np.dot(p0, matrQover[0][0]), '\n')

        print('\nChecking p_0 * matr_a')
        print(np.dot(p0, np.transpose(matr_a)), '\n')

        return p0

    def calc_stationary_probas(self, verbose=True):
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

        if self.check_probas(stationary_probas):
            print("stationary probas calculated\n")
        else:
            print("stationary probas calculated with error!\n", file=sys.stderr)

        return stationary_probas

    def check_probas(self, stationary_probas):
        sum = 0.0
        for num, proba in enumerate(stationary_probas):
            print("p" + str(num) + ": " + to_latex_table(proba))
            temp_sum = np.sum(proba)
            sum += temp_sum
            print("sum: " + str(temp_sum) + "\n")

        print("all_sum:", str(sum))

        return 1 - self._eps_proba < sum < 1 + self._eps_proba

    def calc_system_empty_proba(self, stationary_probas):
        r_multiplier = np.array(np.bmat([[e_col(self.queries_stream.dim_)],
                                         [np.zeros((self.queries_stream.dim_ * self.serv_stream.dim, 1))]]),
                                dtype=float)
        return np.dot(stationary_probas[0], r_multiplier)[0][0]

    def calc_system_single_query_proba(self, stationary_probas):
        r_multiplier = np.array(np.bmat([[np.zeros((self.queries_stream.dim_, 1))],
                                         [e_col(self.queries_stream.dim_ * self.serv_stream.dim)]]),
                                dtype=float)
        return np.dot(stationary_probas[0], r_multiplier)[0][0]

    def calc_buffer_i_queries_j_nonprior(self, stationary_probas, i, j):
        mulW_M = self.queries_stream.dim_ * self.serv_stream.dim
        block0_size = int(mulW_M * (self.timer_stream.dim ** j - 1) / (self.timer_stream.dim - 1))
        block1_size = int(mulW_M * (self.timer_stream.dim ** j))
        block2_size = int(mulW_M * self.timer_stream.dim ** (j + 1) * (self.timer_stream.dim ** (i - j) - 1) / (
                self.timer_stream.dim - 1))
        r_multiplier = np.array(np.bmat([[np.zeros((block0_size, 1))],
                                         [e_col(block1_size)],
                                         [np.zeros((block2_size, 1))]]),
                                dtype=float)
        return np.dot(stationary_probas[i],
                      r_multiplier)[0][0]

    def calc_buffer_i_queries(self, stationary_probas, i):
        return np.sum([self.calc_buffer_i_queries_j_nonprior(stationary_probas, i, j) for j in range(i + 1)])

    def calc_avg_buffer_queries_num(self, stationary_probas):
        return np.sum([i * self.calc_buffer_i_queries(stationary_probas, i) for i in range(1, self.N + 1)])

    def calc_avg_buffer_nonprior_queries_num(self, stationary_probas):
        return np.sum(
            [np.sum([j * self.calc_buffer_i_queries_j_nonprior(stationary_probas, i, j) for j in range(1, i + 1)])
             for i
             in range(1, self.N + 1)])

    def calc_query_lost_p_alg(self, stationary_probas):
        p_loss = np.array(np.dot(stationary_probas[0],
                        np.bmat([[self.I_W],
                                 [kron(np.zeros((self.queries_stream.dim_,
                                                 self.queries_stream.dim_)), e_col(self.serv_stream.dim))]])))

        # for k in range(1, self.N):
        #     p_loss += np.dot(stationary_probas[k],
        #                      kron(kron(self.I_W,
        #                                e_col(self.serv_stream.dim)),
        #                           e_col((self.timer_stream.dim ** (k + 1) - 1) / self.timer_stream.dim - 1)))

        d_sum_1 = (-self.N - 1) * r_multiply_e(self.queries_stream.matrD_0)

        for k in range(1, self.N + 2):
            if k <= self.n:
                d_sum_1 += (k - self.N - 1) * r_multiply_e(
                    self.queries_stream.transition_matrices[0][k] + self.queries_stream.transition_matrices[1][k])

        p_loss = np.dot(p_loss, d_sum_1)  # First addend

        d_sum_2 = (-self.N) * r_multiply_e(self.queries_stream.matrD_0)

        for k in range(1, self.N + 1):
            if k <= self.n:
                d_sum_2 += (k - self.N) * r_multiply_e(
                    self.queries_stream.transition_matrices[0][k] + self.queries_stream.transition_matrices[1][k])

        p_loss += np.array(np.dot(np.dot(stationary_probas[0],
                                         np.bmat([[np.zeros((self.queries_stream.dim_,
                                                             self.queries_stream.dim_))],
                                                  [kron(self.I_W,
                                                        e_col(self.serv_stream.dim))]])),
                                  d_sum_2))

        for i in range(1, self.N):
            r_sum_temp = np.dot(stationary_probas[i],
                                kron(kron(self.I_W,
                                          e_col(self.serv_stream.dim)),
                                     e_col(int((self.timer_stream.dim ** (i + 1) - 1) / (self.timer_stream.dim - 1)))))

            d_sum = (i - self.N) * r_multiply_e(self.queries_stream.matrD_0)
            for k in range(1, self.N - i + 1):
                if k <= self.n:
                    d_sum += (k - self.N + i) * r_multiply_e(self.queries_stream.transition_matrices[0][k] +
                                                             self.queries_stream.transition_matrices[1][k])
            p_loss += np.dot(r_sum_temp, d_sum)

        p_loss = 1 - (1 / self.queries_stream.avg_intensity) * p_loss[0][0]

        return p_loss

    def calc_query_lost_p(self, stationary_probas):
        p_loss = np.array(np.dot(stationary_probas[0],
                                 np.bmat([[np.zeros((self.queries_stream.dim_,
                                                     self.serv_stream.dim))],
                                          [kron(e_col(self.queries_stream.dim_),
                                                self.I_M)]])))

        for i in range(1, self.N):
            p_loss += np.dot(stationary_probas[i],
                             kron(kron(e_col(self.queries_stream.dim_),
                                       self.I_M),
                                  e_col(int((self.timer_stream.dim ** (i + 1) - 1) / (self.timer_stream.dim - 1)))))

        p_loss = np.dot(p_loss, self.serv_stream.repres_matr_0)
        p_loss = 1 - (1 / self.queries_stream.avg_intensity) * p_loss[0][0]

        return p_loss

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

        p_loss = self.calc_query_lost_p(stationary_probas)
        if verbose:
            print("P_loss =", p_loss)

        p_loss_alg = self.calc_query_lost_p_alg(stationary_probas)
        print("P_loss_alg =", p_loss_alg)

    def print_generator(self, as_latex=True):
        for row_num, block_row in enumerate(self.generator):
            print('Row', row_num)
            for block_num, block in enumerate(block_row):
                print(block_num, block)


if __name__ == '__main__':
    test_data = test.MmapPoissonPhPh()

    qs = ClassicTwoPrioritiesQueueingSystem(test_data, verbose=True)
    qs.queries_stream.print_characteristics()
    qs.serv_stream.print_characteristics(matrix_name='S', vector_name='beta')
    qs.timer_stream.print_characteristics(matrix_name='Г', vector_name='gamma')
    qs.print_generator()
    qs.calc_characteristics(True)
