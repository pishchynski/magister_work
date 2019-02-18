import datetime

import matplotlib.pyplot as plt
import numpy as np

import experiments_data.BMMAP3_02_PH_PH as cor02
import experiments_data.BMMAP3_04_PH_PH as cor04
import experiments_data.BMMAP3_Poisson_PH_PH as poisson
import experiments_data.BMMAP3_Poisson_PH_Poisson_noRemoval as poisson_noremoval
import two_priorities_qs as qs


def main():
    test_data_Poisson_initial = poisson.Bmmap3PoissonPhPh()
    test_data_Poisson = poisson.Bmmap3PoissonPhPh()

    test_data_02_initial = cor02.Bmmap302PhPh()
    test_data_02 = cor02.Bmmap302PhPh()

    test_data_04_initial = cor04.Bmmap304PhPh()
    test_data_04 = cor04.Bmmap304PhPh()

    test_data_Poisson_noRemoval_initial = poisson_noremoval.Bmmap3PoissonPhPoissonNoRemoval()
    test_data_Poisson_noRemoval = poisson_noremoval.Bmmap3PoissonPhPoissonNoRemoval()

    qs1 = qs.TwoPrioritiesQueueingSystem(test_data_Poisson_initial)
    qs2 = qs.TwoPrioritiesQueueingSystem(test_data_02_initial)
    qs3 = qs.TwoPrioritiesQueueingSystem(test_data_04_initial)
    qs4 = qs.TwoPrioritiesQueueingSystem(test_data_Poisson_noRemoval_initial)

    lambdas = {"poisson": [], "cor02": [], "cor04": [], "poisson_NoRemoval": []}
    Ls = {"poisson": [], "cor02": [], "cor04": [], "poisson_NoRemoval": []}
    Ps = {"poisson": [], "cor02": [], "cor04": [], "poisson_NoRemoval": []}

    for coef in np.linspace(0.01, 3, 200):
        test_data_Poisson.test_matrD = test_data_Poisson_initial.test_matrD * coef
        test_data_Poisson.test_matrD_0 = test_data_Poisson_initial.test_matrD_0 * coef

        test_data_02.test_matrD = test_data_02_initial.test_matrD * coef
        test_data_02.test_matrD_0 = test_data_02_initial.test_matrD_0 * coef

        test_data_04.test_matrD = test_data_04_initial.test_matrD * coef
        test_data_04.test_matrD_0 = test_data_04_initial.test_matrD_0 * coef

        test_data_Poisson_noRemoval.test_matrD = test_data_Poisson_noRemoval_initial.test_matrD * coef
        test_data_Poisson_noRemoval.test_matrD_0 = test_data_Poisson_noRemoval_initial.test_matrD_0 * coef

        qs1 = qs.TwoPrioritiesQueueingSystem(test_data_Poisson)
        stationary_probas1 = qs1.calc_stationary_probas(False)

        qs2 = qs.TwoPrioritiesQueueingSystem(test_data_02)
        stationary_probas2 = qs2.calc_stationary_probas(False)

        qs3 = qs.TwoPrioritiesQueueingSystem(test_data_04)
        stationary_probas3 = qs3.calc_stationary_probas(False)

        qs4 = qs.TwoPrioritiesQueueingSystem(test_data_Poisson_noRemoval)
        stationary_probas4 = qs4.calc_stationary_probas(False)

        lambdas["poisson"].append(qs1.queries_stream.avg_intensity)
        Ls["poisson"].append(qs1.calc_avg_buffer_queries_num(stationary_probas1))
        Ps["poisson"].append(qs1.calc_query_lost_p(stationary_probas1))

        lambdas["cor02"].append(qs2.queries_stream.avg_intensity)
        Ls["cor02"].append(qs2.calc_avg_buffer_queries_num(stationary_probas2))
        Ps["cor02"].append(qs2.calc_query_lost_p(stationary_probas2))

        lambdas["cor04"].append(qs3.queries_stream.avg_intensity)
        Ls["cor04"].append(qs3.calc_avg_buffer_queries_num(stationary_probas3))
        Ps["cor04"].append(qs3.calc_query_lost_p(stationary_probas3))

        lambdas["poisson_NoRemoval"].append(qs4.queries_stream.avg_intensity)
        Ls["poisson_NoRemoval"].append(qs4.calc_avg_buffer_queries_num(stationary_probas4))
        Ps["poisson_NoRemoval"].append(qs4.calc_query_lost_p(stationary_probas4))

    plt.plot(lambdas["poisson"], Ls["poisson"])
    plt.plot(lambdas["cor02"], Ls["cor02"])
    plt.plot(lambdas["cor04"], Ls["cor04"])
    plt.plot(lambdas["poisson_NoRemoval"], Ls["poisson_NoRemoval"])
    plt.ylabel('L')
    plt.xlabel('λ')
    plt.legend((str.format("c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs1.queries_stream.c_cor[0], 3),
                           round(qs1.queries_stream.c_cor[1], 3)),
                str.format("c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs2.queries_stream.c_cor[0], 3),
                           round(qs2.queries_stream.c_cor[1], 3)),
                str.format("c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs3.queries_stream.c_cor[0], 3),
                           round(qs3.queries_stream.c_cor[1], 3)),
                str.format("no removal c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs4.queries_stream.c_cor[0], 3),
                           round(qs4.queries_stream.c_cor[1], 3))
                ),
               loc=0)
    plt.title('Зависимость L от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP3_L_lambda_cor_0_02_04_NoRemoval_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()

    plt.plot(lambdas["poisson"], Ps["poisson"])
    plt.plot(lambdas["cor02"], Ps["cor02"])
    plt.plot(lambdas["cor04"], Ps["cor04"])
    plt.plot(lambdas["poisson_NoRemoval"], Ps["poisson_NoRemoval"])
    plt.ylabel('P_loss')
    plt.xlabel('λ')
    plt.legend((str.format("c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs1.queries_stream.c_cor[0], 3),
                           round(qs1.queries_stream.c_cor[1], 3)),
                str.format("c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs2.queries_stream.c_cor[0], 3),
                           round(qs2.queries_stream.c_cor[1], 3)),
                str.format("c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs3.queries_stream.c_cor[0], 3),
                           round(qs3.queries_stream.c_cor[1], 3)),
                str.format("no removal c_cor^(1)={0}, c_cor^(2)={1}",
                           round(qs4.queries_stream.c_cor[0], 3),
                           round(qs4.queries_stream.c_cor[1], 3))
                ),
               loc=0)
    plt.title('Зависимость P_loss от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP3_P_loss_lambda_cor_0_02_04_NoRemoval_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()