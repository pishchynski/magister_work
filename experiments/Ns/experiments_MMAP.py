import datetime

import matplotlib.pyplot as plt

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

    Ns = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    Ls = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    Ps = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    p_0s = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}

    for newN in range(1, 25):
        test_data_Poisson.test_N = newN

        test_data_02.test_N = newN

        test_data_04.test_N = newN

        qs1 = qs.TwoPrioritiesQueueingSystem(test_data_Poisson)
        stationary_probas1 = qs1.calc_stationary_probas(False)

        qs2 = qs.TwoPrioritiesQueueingSystem(test_data_02)
        stationary_probas2 = qs2.calc_stationary_probas(False)

        qs3 = qs.TwoPrioritiesQueueingSystem(test_data_04)
        stationary_probas3 = qs3.calc_stationary_probas(False)

        Ns["poisson"].append(newN)
        Ls["poisson"].append(qs1.calc_avg_buffer_queries_num(stationary_probas1))
        Ps["poisson"].append(qs1.calc_query_lost_p(stationary_probas1))
        p_0s["poisson"].append(qs1.calc_system_empty_proba(stationary_probas1))

        Ns["cor02"].append(newN)
        Ls["cor02"].append(qs2.calc_avg_buffer_queries_num(stationary_probas2))
        Ps["cor02"].append(qs2.calc_query_lost_p(stationary_probas2))
        p_0s["cor02"].append(qs2.calc_system_empty_proba(stationary_probas2))


        Ns["cor04"].append(newN)
        Ls["cor04"].append(qs3.calc_avg_buffer_queries_num(stationary_probas3))
        Ps["cor04"].append(qs3.calc_query_lost_p(stationary_probas3))
        p_0s["cor04"].append(qs3.calc_system_empty_proba(stationary_probas3))


    plt.plot(Ns["poisson"], Ls["poisson"])
    plt.plot(Ns["cor02"], Ls["cor02"])
    plt.plot(Ns["cor04"], Ls["cor04"])

    plt.ylabel('L')
    plt.xlabel('N')
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
    plt.title('Зависимость L от N при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/MMAP_L_N_cor_0_02_04_NoRemoval_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()

    plt.plot(Ns["poisson"], Ps["poisson"])
    plt.plot(Ns["cor02"], Ps["cor02"])
    plt.plot(Ns["cor04"], Ps["cor04"])

    plt.ylabel('P_loss')
    plt.xlabel('N')
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
    plt.title('Зависимость P_loss от N при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/MMAP_P_loss_N_cor_0_02_04_NoRemoval_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()

    plt.plot(Ns["poisson"], p_0s["poisson"])
    plt.plot(Ns["cor02"], p_0s["cor02"])
    plt.plot(Ns["cor04"], p_0s["cor04"])

    plt.ylabel('p_0')
    plt.xlabel('N')
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
    plt.title('Зависимость p_0 от N при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/MMAP_p_0_N_cor_0_02_04_NoRemoval_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
