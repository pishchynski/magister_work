import copy
from math import sqrt
import sys

sys.path.append("../")
from utils import *


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

    def __init__(self, matrD_0, matrD, q, n):
        """
        Constructor for BMAPStream.
        :param matrD_0: np.array or list with matrix D_0
        :param matrD: np.array or list with matrix that will be used to generate other matrices
        :param q: float coefficient for generating other matrices
        :param n: int number of matrices to be generated (excluding matrix D_0)
        """

        self.q = q
        self.transition_matrices = [np.array(matrD_0)]
        self.matrD = np.array(matrD)
        for k in range(1, n + 1):
            self.transition_matrices.append(self.matrD * (q ** (k - 1)) * (1 - q) / (1 - q ** 3))

        matrD_1_ = np.zeros(self.transition_matrices[0].shape)
        for matr in self.transition_matrices:
            matrD_1_ += matr
        theta = system_solve(matrD_1_)
        matrdD_1_ = np.array(copy.deepcopy(self.transition_matrices[1]))
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
        self.transition_matrices = np.array(transition_matrices)
        matrD_1_ = np.zeros(self.transition_matrices[0].shape)
        for matr in self.transition_matrices:
            matrD_1_ += matr
        theta = system_solve(matrD_1_)
        matrdD_1_ = np.array(copy.deepcopy(self.transition_matrices[1]))
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
    def __init__(self, matrD_0, matrD, q=0.8, n=3):
        """
        Constructor for BMMAPStream.
        :param matrD_0: np.array or list with matrix D_0
        :param matrD: np.array or list with matrix that will be used to generate other matrices
        :param q: float coefficient for generating other matrices
        :param n: int number of matrices to be generated (excluding matrix D_0)
        """

        self.q = q
        self.transition_matrices_1 = []
        self.transition_matrices_2 = []
        self.matrD_0 = np.array(matrD_0)
        matrD_1 = 0.7 * np.array(matrD)
        matrD_2 = 0.3 * np.array(matrD)
        for k in range(1, n + 1):
            self.transition_matrices_1.append(matrD_1 * (q ** (k - 1)) * (1 - q) / (1 - q ** 3))
            self.transition_matrices_2.append(matrD_2 * (q ** (k - 1)) * (1 - q) / (1 - q ** 3))

        matrD_1_ = np.zeros(self.matrD_0.shape)
        


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

        self.repres_vect = np.array(repres_vect)
        self.repres_matr = np.array(repres_matr)
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
