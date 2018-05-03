import numpy as np
from itertools import product
from utils import ncr

def get_matr_E(R: int, j: int):
    all_comb = list(reversed([np.array(x) for x in product([i for i in range(j + 1)], repeat=R)]))
    rows_comb = list(filter(lambda x: np.sum(x) == j, all_comb))
    cols_comb = list(filter(lambda x: np.sum(x) == j - 1, all_comb))
    deltas = np.array([x - y for x in rows_comb for y in cols_comb])
    matr_E = np.reshape([0 if np.min(delta) < 0 else rows_comb[ind // R][np.argmax(delta)] for (ind, delta) in enumerate(deltas)],
                        (ncr(j + R - 1, R - 1), ncr(j + R - 2, R - 1)))

    print(matr_E)


if __name__ == '__main__':
    get_matr_E(2, 2)
