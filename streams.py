import sys

from math import sqrt

sys.path.append("../")
from utils import *


class MAPStream:
    """
    MAP stream class.
    Contains two transition matrices, stream intensity,
    sum of transition matrices, variation coefficient and correlation coefficient.
    """

    def print_characteristics(self, matrix_name, file=sys.stdout):
        """
        Prints characteristics of MAP stream:
        Average intensity
        Variation coefficient
        Correlation coefficient
        :return: None
        """

        for i, matr in enumerate(self.transition_matrices):
            print(matrix_name + '_' + str(i), ':', file=file)
            matr_print(matr, file=file)

        print('Average intensity:', self.avg_intensity, file=file)
        print('Variation coefficient:', self.c_var, file=file)
        print('Correlation coefficient:', self.c_cor, file=file)
        print('=======END=======', '\n', file=file)

    def __init__(self, transition_matr0, transition_matr1):
        """
        Constructor for MAPStream.
        :param transition_matr0: np.array or list with transition matrix 0
        :param transition_matr1: np.array or list with transition matrix 1
        """

        self.transition_matrices = [np.array(transition_matr0, dtype=float)]
        self.transition_matrices.append(np.array(transition_matr1, dtype=float))
        self.transition_matrices_sum = self.transition_matrices[0] + self.transition_matrices[1]
        gamma = system_solve(self.transition_matrices_sum)
        self.avg_intensity = r_multiply_e(np.dot(gamma, self.transition_matrices[1]))[0]
        self.dim_ = self.transition_matrices[1].shape[0]
        self.dim = self.dim_ - 1
        c_var2 = 2 * self.avg_intensity * r_multiply_e(np.dot(gamma,
                                                              la.inv(-self.transition_matrices[0])))[0] - 1
        self.c_var = sqrt(c_var2)
        self.c_cor = (self.avg_intensity * (r_multiply_e(np.dot(np.dot(np.dot(gamma,
                                                                              la.inv(-self.transition_matrices[0])),
                                                                       self.transition_matrices[1]),
                                                                la.inv(-self.transition_matrices[0])))[0]) - 1) / c_var2


class BMAPStream:
    """
    BMAP stream class.
    Contains list of transition matrices, stream average intensity, stream batches intensity,
    variation coefficient and correlation coefficient.
    """

    def print_characteristics(self, matrix_name, file=sys.stdout):
        """
        Prints characteristics of BMAP stream:
        Matrices
        Average intensity
        Average batch intensity
        Variation coefficient
        Correlation coefficient
        :return: None
        """

        for i, matr in enumerate(self.transition_matrices):
            print(matrix_name + '_' + str(i), ':', file=file)
            matr_print(matr, file=file)

        print('Average intensity:', self.avg_intensity, file=file)
        print('Average batch intensity:', self.batch_intensity, file=file)
        print('Variation coefficient:', self.c_var, file=file)
        print('Correlation coefficient:', self.c_cor, file=file)
        print('=======END=======', '\n', file=file)

    def __init__(self, matrD_0, matrD, q=0.8, n=3):
        """
        Constructor for BMAPStream.
        :param matrD_0: np.array or list with matrix D_0
        :param matrD: np.array or list with matrix that will be used to generate other matrices
        :param q: float coefficient for generating other matrices
        :param n: int number of matrices to be generated (excluding matrix D_0)
        """

        self.q = q
        self.transition_matrices = [np.array(matrD_0, dtype=float)]
        self.matrD = np.array(matrD, dtype=float)
        for k in range(1, n + 1):
            self.transition_matrices.append(self.matrD * (q ** (k - 1)) * (1 - q) / (1 - q ** 3))

        matrD_1_ = np.zeros(self.transition_matrices[0].shape)
        for matr in self.transition_matrices:
            matrD_1_ += matr
        theta = system_solve(matrD_1_)
        matrdD_1_ = np.array(copy.deepcopy(self.transition_matrices[1]), dtype=float)
        for i in range(2, n + 1):
            matrdD_1_ += self.transition_matrices[i] * i
        self.avg_intensity = r_multiply_e(np.dot(theta, matrdD_1_))[0]
        self.batch_intensity = r_multiply_e(np.dot(theta, -self.transition_matrices[0]))[0]
        self.dim_ = self.transition_matrices[0].shape[0]
        self.dim = self.dim_ - 1
        c_var2 = 2 * self.batch_intensity * r_multiply_e(np.dot(theta,
                                                                la.inv(-self.transition_matrices[0])))[0] - 1
        self.c_var = sqrt(c_var2)
        self.c_cor = (self.batch_intensity * (r_multiply_e(np.dot(np.dot(np.dot(theta,
                                                                                la.inv(-self.transition_matrices[0])),
                                                                         matrD_1_ - self.transition_matrices[0]),
                                                                  la.inv(-self.transition_matrices[0])))[
            0]) - 1) / c_var2

    def set_transition_matrices(self, transition_matrices):
        """
        Sets BMAP transition_matrices
        :param transition_matrices: iterable of np.arrays
        """
        self.transition_matrices = np.array(transition_matrices, dtype=float)
        matrD_1_ = np.zeros(self.transition_matrices[0].shape)
        for matr in self.transition_matrices:
            matrD_1_ += matr
        theta = system_solve(matrD_1_)
        matrdD_1_ = np.array(copy.deepcopy(self.transition_matrices[1]), dtype=float)
        for i in range(2, len(self.transition_matrices)):
            matrdD_1_ += self.transition_matrices[i] * i
        self.avg_intensity = r_multiply_e(np.dot(theta, matrdD_1_))[0]
        self.batch_intensity = r_multiply_e(np.dot(theta, -self.transition_matrices[0]))[0]
        self.dim_ = self.transition_matrices[0].shape[0]
        self.dim = self.dim_ - 1
        c_var2 = 2 * self.batch_intensity * r_multiply_e(np.dot(theta,
                                                                la.inv(-self.transition_matrices[0])))[0] - 1
        self.c_var = sqrt(c_var2)
        self.c_cor = (self.batch_intensity * (r_multiply_e(np.dot(np.dot(np.dot(theta,
                                                                                la.inv(
                                                                                    -self.transition_matrices[0])),
                                                                         matrD_1_ - self.transition_matrices[0]),
                                                                  la.inv(-self.transition_matrices[0])))[
            0]) - 1) / c_var2


class BMMAPStream:
    """
    BMMAP stream class.
    Contains list of transition matrices for each type, stream average intensities, stream average intensity,
    stream batches intensity and correlation coefficient.
    """

    def print_characteristics(self, matrix_name='D', file=sys.stdout):
        """
        Prints characteristics of BMMAP stream:
        Transition matrices
        Average intensities
        Average intensity (sum of avg_intensities)
        Average batch intensities
        Correlation coefficient
        :return: None
        """

        for t, type_transition_matrices in enumerate(self.transition_matrices):
            for i, matr in enumerate(type_transition_matrices):
                if matr is not None:
                    print(matrix_name + '^' + str(t + 1) + '_' + str(i), ':', file=file)
                    matr_print(matr, file=file)
            print()

        print("theta =", self.theta)

        print('Average intensities:', file=file)
        for t, intensity in enumerate(self.avg_intensity_t):
            print('avg_intensity_' + str(t + 1) + ':', intensity, file=file)

        print('Average intensity:', self.avg_intensity, file=file)
        print('Average batch intensity:', file=file)
        for t, intensity in enumerate(self.batch_intensity_t):
            print('batch_intensity_' + str(t + 1) + ':', intensity, file=file)

        print('Correlation coefficient:', self.c_cor, file=file)
        print('=======END=======', '\n', file=file)

    def __init__(self, matrD_0, matrD, q=0.8, n=3, t_num=2, priority_part=0.7):
        """
        Constructor for BMMAPStream.
        transition_matrices is a list where each row is transition matrices for one query type
        matrD_l is list of D matrices for each type {0, 1, 2, ...}

        :param matrD_0: np.array or list with matrix D_0
        :param matrD: np.array or list with matrix that will be used to generate other matrices
        :param q: float coefficient for generating other matrices
        :param n: int number of matrices to be generated (excluding matrix D_0)
        :param t_num: int number of queries types
        """

        self.q = q
        self.transition_matrices = [[] for _ in range(t_num)]
        self.matrD_0 = np.array(matrD_0, dtype=float)
        matrD_t = [priority_part * np.array(matrD, dtype=float), (1.0 - priority_part) * np.array(matrD, dtype=float)]
        # matrD_t = [0.99999999999 * np.array(matrD, dtype=float), 0.00000000001 * np.array(matrD, dtype=float)]
        # matrD_t = [0.0000000000000001 * np.array(matrD), 0.9999999999999999 * np.array(matrD)]
        for t in range(t_num):
            if n == 3:
                for k in range(1, n + 1):
                    self.transition_matrices[t].append(matrD_t[t] * (q ** (k - 1)) * (1 - q) / (1 - q ** 3))
            elif n == 1:
                self.transition_matrices[t].append(matrD_t[t])

        self.dim_ = self.transition_matrices[0][0].shape[0]
        self.dim = self.dim_ - 1

        matrD_1_ = copy.deepcopy(self.matrD_0)
        for type_transition_matrices in self.transition_matrices:
            for matr in type_transition_matrices:
                matrD_1_ += matr

        self.matrD_1_ = matrD_1_
        matr_hat_D_k = [[] for _ in range(t_num)]   # Numeration from zero
        for t in range(t_num):
            for k in range(n):
                temp_matr = np.zeros(self.transition_matrices[t][k].shape)
                for i in range(k, n):
                    temp_matr += self.transition_matrices[t][i]
                matr_hat_D_k[t].append(temp_matr)

        matr_cal_D_k = [[] for _ in range(t_num)]   # Numeration from zero
        for t in range(t_num):
            for k in range(n):
                temp_matr = np.zeros(self.transition_matrices[t][k].shape)
                for i in range(k, n):
                    for over_t in filter(lambda x: x != t, range(t_num)):
                        temp_matr += self.transition_matrices[over_t][i]
                matr_cal_D_k[t].append(temp_matr)

        theta = system_solve(self.matrD_1_)
        self.theta = theta

        self.avg_intensity_t = []

        for t in range(t_num):
            temp_matr = np.zeros(self.transition_matrices[t][0].shape)
            for k, matr in enumerate(self.transition_matrices[t]):
                temp_matr += (k + 1) * matr
            self.avg_intensity_t.append(r_multiply_e(np.dot(theta, temp_matr))[0])

        self.avg_intensity = np.sum(self.avg_intensity_t)

        self.batch_intensity_t = [r_multiply_e(np.dot(theta, matr_hat_D[0]))[0] for matr_hat_D in matr_hat_D_k]

        dispersion_t = [(2 * r_multiply_e(
            np.dot(theta, la.inv(- matrD_0 - matr_cal_D[0])))[0])
                        / batch_intensity - 1 / batch_intensity ** 2
                        for matr_cal_D, batch_intensity in zip(matr_cal_D_k, self.batch_intensity_t)]

        self.c_cor = [((r_multiply_e(np.dot(np.dot(np.dot(theta,
                                                          la.inv(- matrD_0 - matr_cal_D[0])),
                                                   matr_hat_D[0]),
                                            la.inv(- matrD_0 - matr_cal_D[0])))[0]) / batch_intensity - (
                                   1 / batch_intensity) ** 2) * (1 / dispersion)
                      for matr_cal_D, matr_hat_D, batch_intensity, dispersion in
                      zip(matr_cal_D_k, matr_hat_D_k, self.batch_intensity_t, dispersion_t)]

        for t in range(t_num):
            self.transition_matrices[t] = [None] + self.transition_matrices[t]


class PHStream:
    """
    PH stream class.
    Contains representation vector, representation matrix, representation matrix_0,
    stream control Markov chain dimensions, stream intensity,
    variation coefficient and correlation coefficient.
    """

    def print_characteristics(self, matrix_name, vector_name, file=sys.stdout):
        """
        Prints characteristics of PH stream:
        Matrix
        Vector
        Average intensity
        Variation coefficient
        Correlation coefficient
        :return: None
        """

        print(matrix_name, ':', file=file)
        matr_print(self.repres_matr, file=file)
        print(vector_name, ':', file=file)
        print(self.repres_vect[0], file=file)

        print('Average intensity:', self.avg_intensity, file=file)
        print('Variation coefficient:', self.c_var, file=file)
        print('=======END=======', '\n', file=file)

    def __init__(self, repres_vect, repres_matr):
        """
        Constructor for PHStream
        :param repres_vect: np.array or list with representation vector
        :param repres_matr: np.array or list with representation matrix
        """

        self.repres_vect = np.array(repres_vect, dtype=float)
        self.repres_matr = np.array(repres_matr, dtype=float)
        self.repres_matr_0 = -r_multiply_e(self.repres_matr)
        self.avg_intensity = -la.inv(r_multiply_e(np.dot(self.repres_vect,
                                                         la.inv(self.repres_matr))))[0, 0]
        self.dim = self.repres_matr.shape[0]
        self.dim_ = self.dim + 1
        b1 = r_multiply_e(np.dot(self.repres_vect,
                                 la.inv(-self.repres_matr)))[0]
        b2 = 2 * r_multiply_e(np.dot(self.repres_vect,
                                     np.linalg.matrix_power(-self.repres_matr, -2)))[0]
        c_var2 = (b2 - b1 ** 2) / b1 ** 2
        self.c_var = sqrt(c_var2)
