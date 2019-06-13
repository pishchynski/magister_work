import datetime

import matplotlib.pyplot as plt
import numpy as np

import experiments_data.BMMAP3_02_PH_PH as cor02
import experiments_data.BMMAP3_04_PH_PH as cor04
import experiments_data.BMMAP3_Poisson_PH_PH as poisson
import two_priorities_qs as qs


def main():
    test_data_Poisson_initial = poisson.Bmmap3PoissonPhPh()
    test_data_Poisson = poisson.Bmmap3PoissonPhPh()

    test_data_02_initial = cor02.Bmmap302PhPh()
    test_data_02 = cor02.Bmmap302PhPh()

    test_data_04_initial = cor04.Bmmap304PhPh()
    test_data_04 = cor04.Bmmap304PhPh()

    qs1 = qs.TwoPrioritiesQueueingSystem(test_data_Poisson_initial)
    qs2 = qs.TwoPrioritiesQueueingSystem(test_data_02_initial)
    qs3 = qs.TwoPrioritiesQueueingSystem(test_data_04_initial)

    W1s = {"poisson": [], "cor02": [], "cor04": []}
    W2s = {"poisson": [], "cor02": [], "cor04": []}

    coef = 0.25 # as we have \lambda = 32 and want it to be 8
    ts = []

    for t in np.linspace(0.0001, 1., 200):
        test_data_Poisson.test_matrD = test_data_Poisson_initial.test_matrD * coef
        test_data_Poisson.test_matrD_0 = test_data_Poisson_initial.test_matrD_0 * coef

        test_data_02.test_matrD = test_data_02_initial.test_matrD * coef
        test_data_02.test_matrD_0 = test_data_02_initial.test_matrD_0 * coef

        test_data_04.test_matrD = test_data_04_initial.test_matrD * coef
        test_data_04.test_matrD_0 = test_data_04_initial.test_matrD_0 * coef

        qs1 = qs.TwoPrioritiesQueueingSystem(test_data_Poisson)
        stationary_probas1 = qs1.calc_stationary_probas(False)

        qs2 = qs.TwoPrioritiesQueueingSystem(test_data_02)
        stationary_probas2 = qs2.calc_stationary_probas(False)

        qs3 = qs.TwoPrioritiesQueueingSystem(test_data_04)
        stationary_probas3 = qs3.calc_stationary_probas(False)

        ts.append(t)
        W1s["poisson"].append(qs1.func_W_1(stationary_probas1, t))
        W2s["poisson"].append(qs1.func_W_2(stationary_probas1, t))

        W1s["cor02"].append(qs2.func_W_1(stationary_probas2, t))
        W2s["cor02"].append(qs2.func_W_2(stationary_probas2, t))

        W1s["cor04"].append(qs3.func_W_1(stationary_probas3, t))
        W2s["cor04"].append(qs3.func_W_2(stationary_probas3, t))

    plt.plot(ts, W1s["poisson"])
    plt.plot(ts, W1s["cor02"])
    plt.plot(ts, W1s["cor04"])
    plt.ylabel('W_1')
    plt.xlabel('t')
    plt.legend((str.format("c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs1.queries_stream.c_cor[0], 3),
                           round(qs1.queries_stream.c_cor[1], 3)),
                str.format("c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs2.queries_stream.c_cor[0], 3),
                           round(qs2.queries_stream.c_cor[1], 3)),
                str.format("c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs3.queries_stream.c_cor[0], 3),
                           round(qs3.queries_stream.c_cor[1], 3)),
                ),
               loc=0)
    plt.title('Зависимость W_1 от t при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP3_W_1_t_cor_0_02_04_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()

    plt.plot(ts, W2s["poisson"])
    plt.plot(ts, W2s["cor02"])
    plt.plot(ts, W2s["cor04"])
    plt.ylabel('W_2')
    plt.xlabel('t')
    plt.legend((str.format("c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs1.queries_stream.c_cor[0], 3),
                           round(qs1.queries_stream.c_cor[1], 3)),
                str.format("c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs2.queries_stream.c_cor[0], 3),
                           round(qs2.queries_stream.c_cor[1], 3)),
                str.format("c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs3.queries_stream.c_cor[0], 3),
                           round(qs3.queries_stream.c_cor[1], 3)),
                ),
               loc=0)
    plt.title('Зависимость W_2 от t при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP3_W_2_t_cor_0_02_04_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()

