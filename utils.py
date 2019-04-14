import copy
import operator as op
import sys
from functools import reduce
from sys import stderr
from time import sleep

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

characteristics_loc = ["Загруженность системы",
                       "Пропускная способность системы",
                       "Среднее число запросов в системе",
                       "Дисперсия числа запросов в системе",
                       "Вероятность того, что прибор 1 исправен и обслуживает запрос",
                       """Вероятность того, что прибор 1 в неисправном состоянии, а прибор 2
                          обслуживает запрос""",
                       """Вероятность того, что в системе есть запросы, прибор1 в неисправном
                          состоянии и идет переключение с этого прибора на прибор 2 (при этом оба
                          прибора не обслуживают заявки)""",
                       """Вероятность того, что в системе есть запросы, прибор 1 в исправном
                          состоянии и идет переключение с прибора 2 на прибор 1 (при этом прибор 2
                          продолжает обслуживать запросы)""",
                       """Вероятность того, что прибор 1 доступен (средняя доля времени, в течение которого
                          прибор 1 доступен)""",
                       """Вероятность того, что прибор 1 недоступен, а прибор 2 доступен (средняя доля времени,
                          в течение которого прибор 2 доступен)""",
                       """Вероятность того, что оба прибора недоступны, т.е. идет переключение с прибора 1
                          на прибор 2 (средняя доля времени, в течение которого оба прибора недоступны)""",
                       """Среднее число переключений с прибора 1 на прибор 2 в единицу времени""",
                       """Среднее число переключений с прибора 2 на прибор 1 в единицу времени""",
                       "Среднее время нахождения заявки в системе",
                       "Среднее совокупное число переключений в единицу времени"]

characteristics_names = ['system_load',
                         'system_capacity',
                         'avg_queries_num',
                         'queries_num_dispersion',
                         'prob_1_work_serves',
                         'prob_1_broken_2_serves',
                         'prob_1_broken_switch_1_2',
                         'prob_1_work_switch_2_1',
                         'prob_1_available',
                         'prob_1_unavail_2_avail',
                         'prob_1_2_unavail',
                         'avg_switch_1_2_num',
                         'avg_switch_2_1_num',
                         'avg_service_time',
                         'avg_switch_num']


def kron(A: object, B: object) -> object:
    """
    Just np.linalg.kron() function.
    :param A: np.array
    :param B: np.array
    :return: np.array as a result of kronecker multiplication of A and B
    """
    return la.kron(A, B)


def kronsum(A, B):
    """
    Kronecker sum function.
    :param A: np.array
    :param B: np.array
    :return: np.array as a result of kronecker sum of A and B
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError('A is not square')

    if B.shape[0] != B.shape[1]:
        raise ValueError('B is not square')

    L = kron(A, np.eye(B.shape[0]))
    R = kron(np.eye(A.shape[0]), B)

    return L + R


def kronpow(A, pow):
    """
    Kronecker power function.
    :param A: np.array
    :param pow: int
    :return: np.array as a result of kronecker power A^pow
    """
    if pow == 0:
        return np.array([[1]])

    temp = copy.deepcopy(A)

    for _ in range(1, pow):
        temp = kron(temp, A)

    return temp


def kronsumpow(A, pow):
    """
    Kronecker "sum power" function.
    :param A: np.array
    :param pow: int
    :return: np.array
    """
    if pow < 0:
        raise ValueError(str.format("pow must be >= 0, but {} given", pow))

    if pow == 0:
        return np.array([[0]])

    n = A.shape[0]

    temp = kron(kron(np.eye(1), A), np.eye(n ** (pow - 1)))

    for m in range(1, pow):
        temp += kron(kron(np.eye(n ** m), A), np.eye(n ** (pow - m - 1)))

    return temp


def system_solve(matr):
    """
    Solves system of type (vect * matr = 0) & (vect * e = 1).
    :param matr: np.array
    :return: np.array with vect
    """
    matr_a = np.array(matr, dtype=float)

    for i in range(matr_a.shape[0]):
        matr_a[i][0] = 1

    matr_b = np.zeros((matr_a.shape[0], 1))
    matr_b[0][0] = 1
    matr_a = np.transpose(matr_a)

    result = np.transpose(la.solve(matr_a, matr_b))

    return result[0]


def sparse_system_solve(sparse_matr):
    """
    Solves system of type (vect * matr = 0) & (vect * e = 1).
    :param sparse_matr: scipy sparse array
    :return: np.array with vect
    """
    matr_a = sparse_matr.copy()
    for i in range(matr_a.shape[0]):
        matr_a[i][0] = 1

    matr_b = np.zeros((matr_a.shape[0], 1))
    matr_b[0][0] = 1

    result = spla.spsolve(matr_a, matr_b).transpose()

    return result


def r_multiply_e(matr):
    """
    Multiplies matrix matr on unitary vector of matching shape from the right: matr * e.
    :param matr: np.array
    :return: any type with the result of multiplication
    """
    if len(matr.shape) > 1:
        vect_e = np.array([[1] for _ in range(matr.shape[1])], dtype=float)
    elif len(matr.shape) > 0:
        vect_e = np.array([[1] for _ in range(matr.shape[0])], dtype=float)
    else:
        vect_e = 1.
    result = np.dot(matr, vect_e)

    return result


def e_col(dim):
    """
    Generates unitary vector-column of given dimension.
    :param dim: int dimension
    :return: np.array with unitary vector-column
    """
    return np.array([[1] for _ in range(dim)], dtype=float)


def matr_print(matr, file=sys.stdout):
    """
    Pretty prints the given matrix.
    :param matr: iterable with matrix to print
    :return: None
    """

    s = [[str(e) for e in row] for row in matr]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table), file=file)


def linux_check_cpu_temperature(notify=True):
    try:
        sleep_flag = False
        while True:
            with open('/sys/class/thermal/thermal_zone0/temp', mode='r') as temp_file:
                cpu_temp = float(temp_file.readline()) / 1000
            if cpu_temp >= 90.:
                if notify:
                    print('Your CPU is too hot to proceed!', file=stderr)
                    print('Please wait for 60 seconds to cool CPU...', file=stderr)
                    sleep_flag = True
                sleep(60)
            else:
                if sleep_flag:
                    print('CPU temp is Ok. Proceeding...', file=stderr)
                return
    except:
        return


def copy_matrix_block(dest, src, m_pos, n_pos):
    m_shift = src.shape[0]
    n_shift = src.shape[1]
    dest[m_pos: m_pos + m_shift, :][:, n_pos: n_pos + n_shift] = src


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def limdiv(numer, denom):
    """
    Divides more 'safely'. e.g. limdiv(0, 0) == 1

    :param numer: int numerator
    :param denom: int denominator
    :return: int the result of division
    """

    if numer == denom == 0:
        return 1
    else:
        return numer // denom


def to_latex_table(elements):
    latex_table = ''
    for i, element in enumerate(elements[0]):
        if i % 6 == 0:
            latex_table += '& '
        latex_table += str(round(element, 8)) + ', '
        if (i + 1) % 6 == 0:
            latex_table += '&\\\\\n'

    return latex_table
