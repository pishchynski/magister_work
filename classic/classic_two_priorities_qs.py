import sys

sys.path.append("../")

from streams import *

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

    def recalculate_generator(self):
        pass

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
                                       self.queries_stream.dim_ * self.serv_stream.dim * self.timer_stream.dim * (self.timer_stream.dim ** (k - 1) - 1) / (self.timer_stream.dim - 1)))
                blocks0k.append(temp_block)

                temp_block = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                                       self.queries_stream.dim_ * self.serv_stream.dim * self.timer_stream.dim * (self.timer_stream.dim ** (k - 1) - 1) / (self.timer_stream.dim - 1)))
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
                                   kronpow(self.timer_stream.repres_vect, k) )

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
                                   self.queries_stream.dim_ * self.serv_stream.dim * self.timer_stream.dim * (self.timer_stream.dim ** (self.N - 1) - 1) / (self.timer_stream.dim - 1)))
            blocks0k.append(copy.deepcopy(temp_block))

            temp_block = np.zeros((self.queries_stream.dim_ * self.serv_stream.dim,
                                   self.queries_stream.dim_ * self.serv_stream.dim * self.timer_stream.dim * (self.timer_stream.dim ** (self.N - 1) - 1) / (self.timer_stream.dim - 1)))
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
                               kronsumpow(e_col(self.timer_stream.dim), i))
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
            cur_matr_blocks = (kron(kronsum(self.queries_stream.matrD_0 if i != self.N else self.queries_stream.matrD_1_,
                               self.serv_stream.repres_matr),
                       np.eye(self.timer_stream.dim ** j) + kron(self.I_WM, kronsumpow(self.timer_stream.repres_matr, j))) for j in range(i + 1))
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
                                          self.queries_stream.dim_ * self.serv_stream.dim * self.timer_stream.dim ** (i + 1) * (self.timer_stream.dim ** k - 1) / (self.timer_stream.dim - 1)
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
                                          self.queries_stream.dim_ * self.serv_stream.dim * (self.timer_stream.dim ** k - 1) / (self.timer_stream.dim - 1)
                                          ))
                cur_matr += np.concatenate((zero_matr, temp_matr), axis=1)

                matrQ_ii_row.append(cur_matr)

            matrQ_iik.append(matrQ_ii_row)

        print("Q_{i, i + k} calculated")

        return matrQ_iik
