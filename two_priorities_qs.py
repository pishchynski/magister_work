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
        self.n = 3
        self.N = 3

    def set_BMMAP_queries_stream(self, matrD_0, matrD, q=0.8, n=3):
        self.queries_stream = BMAPStream(matrD_0, matrD, q, n)
        self.n = n

    def set_PH_serv_stream(self, vect, matr):
        self.serv_stream = PHStream(vect, matr)

    def set_PH_timer_stream(self, vect, matr):
        self.timer_stream = PHStream(vect, matr)

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

        block00 = np.zeros(self.queries_stream.transition_matrices[0][0].shape)
        if self.N + 1 > self.n:
            block00 = kron(np.zeros(self.queries_stream.transition_matrices[0][0].shape),
                           self.serv_stream.repres_vect)    # because of sum from k=N+1 to inf

        block10 = np.zeros(self.queries_stream.transition_matrices[0][0].shape)
        if self.N == self.n:
            block10 = kron(self.queries_stream.transition_matrices[-1],
                           np.eye(self.serv_stream.dim))
        elif self.N > self.n:
            block10 = kron(np.zeros(self.queries_stream.transition_matrices[0][0].shape),
                           np.eye(self.serv_stream.dim))

        blocks0k = [block00]
        blocks1k = [block10]

        for j in range(1, self.N):

            blocks0k.append()
