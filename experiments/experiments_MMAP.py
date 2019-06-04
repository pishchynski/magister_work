import datetime

import matplotlib.pyplot as plt
import numpy as np

import experiments_data.MMAP_02_PH_PH as cor02
import experiments_data.MMAP_04_PH_PH as cor04
import experiments_data.MMAP_Poisson_PH_PH as poisson
import two_priorities_qs as qs


def main():
    test_data_Poisson_initial = poisson.MmapPoissonPhPh()
    test_data_Poisson = poisson.MmapPoissonPhPh()

    test_data_02_initial = cor02.Mmap02PhPh()
    test_data_02 = cor02.Mmap02PhPh()

    test_data_04_initial = cor04.Mmap04PhPh()
    test_data_04 = cor04.Mmap04PhPh()

    qs1 = qs.TwoPrioritiesQueueingSystem(test_data_Poisson_initial)
    qs2 = qs.TwoPrioritiesQueueingSystem(test_data_02_initial)
    qs3 = qs.TwoPrioritiesQueueingSystem(test_data_04_initial)

    lambdas = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    Ls = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    Ps = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    Ps_alg = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    P1s = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    P2s = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}

    for coef in np.linspace(0.01, 3, 100):
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

        lambdas["poisson"].append(qs1.queries_stream.avg_intensity)
        Ls["poisson"].append(qs1.calc_avg_buffer_queries_num(stationary_probas1))
        Ps["poisson"].append(qs1.calc_query_lost_p(stationary_probas1))
        Ps_alg["poisson"].append(qs1.calc_query_lost_ps_buffer_full(stationary_probas1)[1])
        P1s["poisson"].append(qs1.calc_query_lost_ps_buffer_full(stationary_probas1)[0][0])
        P2s["poisson"].append(qs1.calc_query_lost_ps_buffer_full(stationary_probas1)[0][1])

        lambdas["cor02"].append(qs2.queries_stream.avg_intensity)
        Ls["cor02"].append(qs2.calc_avg_buffer_queries_num(stationary_probas2))
        Ps["cor02"].append(qs2.calc_query_lost_p(stationary_probas2))
        Ps_alg["cor02"].append(qs2.calc_query_lost_ps_buffer_full(stationary_probas2)[1])
        P1s["cor02"].append(qs2.calc_query_lost_ps_buffer_full(stationary_probas2)[0][0])
        P2s["cor02"].append(qs2.calc_query_lost_ps_buffer_full(stationary_probas2)[0][1])

        lambdas["cor04"].append(qs3.queries_stream.avg_intensity)
        Ls["cor04"].append(qs3.calc_avg_buffer_queries_num(stationary_probas3))
        Ps["cor04"].append(qs3.calc_query_lost_p(stationary_probas3))
        Ps_alg["cor04"].append(qs3.calc_query_lost_ps_buffer_full(stationary_probas3)[1])
        P1s["cor04"].append(qs3.calc_query_lost_ps_buffer_full(stationary_probas3)[0][0])
        P2s["cor04"].append(qs3.calc_query_lost_ps_buffer_full(stationary_probas3)[0][1])

    plt.plot(lambdas["poisson"], Ls["poisson"])
    plt.plot(lambdas["cor02"], Ls["cor02"])
    plt.plot(lambdas["cor04"], Ls["cor04"])
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
                ),
               loc=0)
    plt.title('Зависимость L от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/MMAP_L_lambda_cor_0_02_04_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()

    plt.plot(lambdas["poisson"], Ps["poisson"])
    plt.plot(lambdas["cor02"], Ps["cor02"])
    plt.plot(lambdas["cor04"], Ps["cor04"])
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
                ),
               loc=0)
    plt.title('Зависимость P_loss от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/MMAP_P_loss_lambda_cor_0_02_04_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()

    plt.plot(lambdas["poisson"], Ps_alg["poisson"])
    plt.plot(lambdas["cor02"], Ps_alg["cor02"])
    plt.plot(lambdas["cor04"], Ps_alg["cor04"])
    plt.ylabel('P_loss_alg')
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
                ),
               loc=0)
    plt.title('Зависимость P_loss_alg от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/MMAP_P_loss_alg_lambda_cor_0_02_04_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()

    plt.plot(lambdas["poisson"], P1s["poisson"])
    plt.plot(lambdas["cor02"], P1s["cor02"])
    plt.plot(lambdas["cor04"], P1s["cor04"])
    plt.ylabel('P1_loss')
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
                ),
               loc=0)
    plt.title('Зависимость P1_loss от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/MMAP_P1_loss_lambda_cor_0_02_04_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()

    plt.plot(lambdas["poisson"], P2s["poisson"])
    plt.plot(lambdas["cor02"], P2s["cor02"])
    plt.plot(lambdas["cor04"], P2s["cor04"])
    plt.ylabel('P2_loss')
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
                ),
               loc=0)
    plt.title('Зависимость P2_loss от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/MMAP_P2_loss_lambda_cor_0_02_04_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
