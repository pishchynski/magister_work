import datetime
import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

import experiments.lambdas.data.BMMAP5_02_PH_PH as cor02
import experiments.lambdas.data.BMMAP5_04_PH_PH as cor04
import experiments.lambdas.data.BMMAP5_Poisson_PH_PH as poisson
import two_priorities_qs as qs

from tqdm import tqdm


def main():
    PRINT_TITLE = False

    nowdate = datetime.datetime.now()

    if not os.path.exists("plots"):
        os.makedirs("plots")

    if not os.path.exists("tables"):
        os.makedirs("tables")

    lines = ["-k", "--k", "-.k"]
    linecycler = cycle(lines)

    test_data_Poisson_initial = poisson.Bmmap5PoissonPhPh()
    test_data_Poisson = poisson.Bmmap5PoissonPhPh()

    test_data_02_initial = cor02.Bmmap502PhPh()
    test_data_02 = cor02.Bmmap502PhPh()

    test_data_04_initial = cor04.Bmmap504PhPh()
    test_data_04 = cor04.Bmmap504PhPh()

    qs1 = qs.TwoPrioritiesQueueingSystem(test_data_Poisson_initial)
    qs2 = qs.TwoPrioritiesQueueingSystem(test_data_02_initial)
    qs3 = qs.TwoPrioritiesQueueingSystem(test_data_04_initial)

    lambdas = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    Ls = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    LsPrior = {"poisson": [], "cor02": [], "cor04": []}
    sigmas = {"poisson": [], "cor02": [], "cor04": []}
    sigmas_prior = {"poisson": [], "cor02": [], "cor04": []}
    Ps = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    Ps_alg = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    P1s = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    P2s = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    Ps_imp = {"poisson": [], "cor02": [], "cor04": [], "poisson_PH_poisson_NoRemoval": [], "poisson_NoRemoval": []}
    w1s = {"poisson": [], "cor02": [], "cor04": []}
    w2s = {"poisson": [], "cor02": [], "cor04": []}

    table_poisson = []
    table_cor_02 = []
    table_cor_04 = []

    for coef in tqdm(np.linspace(1, 150, 150)):
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
        LsPrior["poisson"].append(qs1.calc_avg_buffer_prior_queries_num(stationary_probas1))
        sigmas["poisson"].append(qs1.calc_sd_buffer_queries_num(stationary_probas1))
        sigmas_prior["poisson"].append(qs1.calc_sd_buffer_prior_queries_num(stationary_probas1))
        Ps["poisson"].append(qs1.calc_query_lost_p(stationary_probas1))
        Ps_alg["poisson"].append(qs1.calc_query_lost_ps_buffer_full(stationary_probas1)[1])
        P1s["poisson"].append(qs1.calc_query_lost_ps_buffer_full(stationary_probas1)[0][0])
        P2s["poisson"].append(qs1.calc_query_lost_ps_buffer_full(stationary_probas1)[0][1])
        Ps_imp["poisson"].append(qs1.calc_nonprior_query_lost_timer(stationary_probas1))
        w1s["poisson"].append(qs1.avg_wait_time_1(stationary_probas1))
        w2s["poisson"].append(qs1.avg_wait_time_2(stationary_probas1))

        lambdas["cor02"].append(qs2.queries_stream.avg_intensity)
        Ls["cor02"].append(qs2.calc_avg_buffer_queries_num(stationary_probas2))
        LsPrior["cor02"].append(qs2.calc_avg_buffer_prior_queries_num(stationary_probas2))
        sigmas["cor02"].append(qs2.calc_sd_buffer_queries_num(stationary_probas2))
        sigmas_prior["cor02"].append(qs2.calc_sd_buffer_prior_queries_num(stationary_probas2))
        Ps["cor02"].append(qs2.calc_query_lost_p(stationary_probas2))
        Ps_alg["cor02"].append(qs2.calc_query_lost_ps_buffer_full(stationary_probas2)[1])
        P1s["cor02"].append(qs2.calc_query_lost_ps_buffer_full(stationary_probas2)[0][0])
        P2s["cor02"].append(qs2.calc_query_lost_ps_buffer_full(stationary_probas2)[0][1])
        Ps_imp["cor02"].append(qs2.calc_nonprior_query_lost_timer(stationary_probas2))
        w1s["cor02"].append(qs2.avg_wait_time_1(stationary_probas2))
        w2s["cor02"].append(qs2.avg_wait_time_2(stationary_probas2))

        lambdas["cor04"].append(qs3.queries_stream.avg_intensity)
        Ls["cor04"].append(qs3.calc_avg_buffer_queries_num(stationary_probas3))
        LsPrior["cor04"].append(qs3.calc_avg_buffer_prior_queries_num(stationary_probas3))
        sigmas["cor04"].append(qs3.calc_sd_buffer_queries_num(stationary_probas3))
        sigmas_prior["cor04"].append(qs3.calc_sd_buffer_prior_queries_num(stationary_probas3))
        Ps["cor04"].append(qs3.calc_query_lost_p(stationary_probas3))
        Ps_alg["cor04"].append(qs3.calc_query_lost_ps_buffer_full(stationary_probas3)[1])
        P1s["cor04"].append(qs3.calc_query_lost_ps_buffer_full(stationary_probas3)[0][0])
        P2s["cor04"].append(qs3.calc_query_lost_ps_buffer_full(stationary_probas3)[0][1])
        Ps_imp["cor04"].append(qs3.calc_nonprior_query_lost_timer(stationary_probas3))
        w1s["cor04"].append(qs3.avg_wait_time_1(stationary_probas3))
        w2s["cor04"].append(qs3.avg_wait_time_2(stationary_probas3))

        table_poisson.append(
            [lambdas["poisson"][-1], P1s["poisson"][-1], P2s["poisson"][-1], Ps_alg["poisson"][-1],
             Ps_imp["poisson"][-1], w1s["poisson"][-1], w2s["poisson"][-1]])
        table_cor_02.append(
            [lambdas["cor02"][-1], P1s["cor02"][-1], P2s["cor02"][-1], Ps_alg["cor02"][-1], Ps_imp["cor02"][-1],
             w1s["cor02"][-1], w2s["cor02"][-1]])
        table_cor_04.append(
            [lambdas["cor04"][-1], P1s["cor04"][-1], P2s["cor04"][-1], Ps_alg["cor04"][-1], Ps_imp["cor04"][-1],
             w1s["cor04"][-1], w2s["cor04"][-1]])

    plt.plot(lambdas["poisson"], Ls["poisson"], next(linecycler))
    plt.plot(lambdas["cor02"], Ls["cor02"], next(linecycler))
    plt.plot(lambdas["cor04"], Ls["cor04"], next(linecycler))
    plt.ylabel(r'$L$')
    plt.xlabel(r'$\lambda$')
    plt.legend((r'$c_{cor}^{(1)}$ =' + str(abs(round(qs1.queries_stream.c_cor[0], 3))) + r', $c_{cor}^{(2)}$=' + str(
        abs(round(qs1.queries_stream.c_cor[1], 3))),
                r'$c_{cor}^{(1)}$ =' + str(round(qs2.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs2.queries_stream.c_cor[1], 3)),
                r'$c_{cor}^{(1)}$ =' + str(round(qs3.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs3.queries_stream.c_cor[1], 3))
                ),
               loc=0)
    if PRINT_TITLE:
        plt.title('Зависимость L от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP5_L_lambda_cor_0_02_04_{}.png", nowdate),
                dpi=400,
                bbox_inches='tight')
    plt.close()

    plt.plot(lambdas["poisson"], LsPrior["poisson"], next(linecycler))
    plt.plot(lambdas["cor02"], LsPrior["cor02"], next(linecycler))
    plt.plot(lambdas["cor04"], LsPrior["cor04"], next(linecycler))
    plt.ylabel(r'$L^{(prior)}$')
    plt.xlabel(r'$\lambda$')
    plt.legend((r'$c_{cor}^{(1)}$ =' + str(abs(round(qs1.queries_stream.c_cor[0], 3))) + r', $c_{cor}^{(2)}$=' + str(
        abs(round(qs1.queries_stream.c_cor[1], 3))),
                r'$c_{cor}^{(1)}$ =' + str(round(qs2.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs2.queries_stream.c_cor[1], 3)),
                r'$c_{cor}^{(1)}$ =' + str(round(qs3.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs3.queries_stream.c_cor[1], 3))
                ),
               loc=0)
    if PRINT_TITLE:
        plt.title('Зависимость L_prior от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP5_L_prior_lambda_cor_0_02_04_{}.png", nowdate),
                dpi=400,
                bbox_inches='tight')
    plt.close()

    plt.plot(lambdas["poisson"], sigmas["poisson"], next(linecycler))
    plt.plot(lambdas["cor02"], sigmas["cor02"], next(linecycler))
    plt.plot(lambdas["cor04"], sigmas["cor04"], next(linecycler))
    plt.ylabel(r'$\sigma$')
    plt.xlabel(r'$\lambda$')
    plt.legend((r'$c_{cor}^{(1)}$ =' + str(abs(round(qs1.queries_stream.c_cor[0], 3))) + r', $c_{cor}^{(2)}$=' + str(
        abs(round(qs1.queries_stream.c_cor[1], 3))),
                r'$c_{cor}^{(1)}$ =' + str(round(qs2.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs2.queries_stream.c_cor[1], 3)),
                r'$c_{cor}^{(1)}$ =' + str(round(qs3.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs3.queries_stream.c_cor[1], 3))
                ),
               loc=0)
    if PRINT_TITLE:
        plt.title('Зависимость sigma от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP5_sigma_lambda_cor_0_02_04_{}.png", nowdate),
                dpi=400,
                bbox_inches='tight')
    plt.close()

    plt.plot(lambdas["poisson"], sigmas_prior["poisson"], next(linecycler))
    plt.plot(lambdas["cor02"], sigmas_prior["cor02"], next(linecycler))
    plt.plot(lambdas["cor04"], sigmas_prior["cor04"], next(linecycler))
    plt.ylabel(r'$\sigma^{(prior})$')
    plt.xlabel(r'$\lambda$')
    plt.legend((r'$c_{cor}^{(1)}$ =' + str(abs(round(qs1.queries_stream.c_cor[0], 3))) + r', $c_{cor}^{(2)}$=' + str(
        abs(round(qs1.queries_stream.c_cor[1], 3))),
                r'$c_{cor}^{(1)}$ =' + str(round(qs2.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs2.queries_stream.c_cor[1], 3)),
                r'$c_{cor}^{(1)}$ =' + str(round(qs3.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs3.queries_stream.c_cor[1], 3))
                ),
               loc=0)
    if PRINT_TITLE:
        plt.title('Зависимость sigma^{(prior}) от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP5_sigma_prior_lambda_cor_0_02_04_{}.png", nowdate),
                dpi=400,
                bbox_inches='tight')
    plt.close()

    # DEPRECATED
    # plt.plot(lambdas["poisson"], Ps["poisson"])
    # plt.plot(lambdas["cor02"], Ps["cor02"])
    # plt.plot(lambdas["cor04"], Ps["cor04"])
    # plt.ylabel(r'$P_{loss}$')
    # plt.xlabel(r'$\lambda$')
    # plt.legend((r'$c_{cor}^{(1)}$ =' + str(abs(round(qs1.queries_stream.c_cor[0], 3))) + r', $c_{cor}^{(2)}$=' + str(
    #     abs(round(qs1.queries_stream.c_cor[1], 3))),
    #             r'$c_{cor}^{(1)}$ =' + str(round(qs2.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
    #                 round(qs2.queries_stream.c_cor[1], 3)),
    #             r'$c_{cor}^{(1)}$ =' + str(round(qs3.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
    #                 round(qs3.queries_stream.c_cor[1], 3))
    #             ),
    #            loc=0)
    # plt.title('Зависимость P_loss от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')
    #
    # plt.savefig(str.format("plots/BMMAP5_P_loss_lambda_cor_0_02_04_{}.png", nowdate),
    #             bbox_inches='tight')
    # plt.close()

    plt.plot(lambdas["poisson"], Ps_alg["poisson"], next(linecycler))
    plt.plot(lambdas["cor02"], Ps_alg["cor02"], next(linecycler))
    plt.plot(lambdas["cor04"], Ps_alg["cor04"], next(linecycler))
    plt.ylabel(r'$P_{loss}$')
    plt.xlabel(r'$\lambda$')
    plt.legend((r'$c_{cor}^{(1)}$ =' + str(abs(round(qs1.queries_stream.c_cor[0], 3))) + r', $c_{cor}^{(2)}$=' + str(
        abs(round(qs1.queries_stream.c_cor[1], 3))),
                r'$c_{cor}^{(1)}$ =' + str(round(qs2.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs2.queries_stream.c_cor[1], 3)),
                r'$c_{cor}^{(1)}$ =' + str(round(qs3.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs3.queries_stream.c_cor[1], 3))
                ),
               loc=0)
    if PRINT_TITLE:
        plt.title('Зависимость P_loss_alg от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP5_P_loss_alg_lambda_cor_0_02_04_{}.png", nowdate),
                dpi=400,
                bbox_inches='tight')
    plt.close()

    plt.plot(lambdas["poisson"], P1s["poisson"], next(linecycler))
    plt.plot(lambdas["cor02"], P1s["cor02"], next(linecycler))
    plt.plot(lambdas["cor04"], P1s["cor04"], next(linecycler))
    plt.ylabel(r'$P_{loss}^{(1)}$')
    plt.xlabel(r'$\lambda$')
    plt.legend((r'$c_{cor}^{(1)}$ =' + str(abs(round(qs1.queries_stream.c_cor[0], 3))) + r', $c_{cor}^{(2)}$=' + str(
        abs(round(qs1.queries_stream.c_cor[1], 3))),
                r'$c_{cor}^{(1)}$ =' + str(round(qs2.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs2.queries_stream.c_cor[1], 3)),
                r'$c_{cor}^{(1)}$ =' + str(round(qs3.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs3.queries_stream.c_cor[1], 3))
                ),
               loc=0)
    if PRINT_TITLE:
        plt.title('Зависимость P1_loss от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP5_P1_loss_lambda_cor_0_02_04_{}.png", nowdate),
                dpi=400,
                bbox_inches='tight')
    plt.close()

    plt.plot(lambdas["poisson"], P2s["poisson"], next(linecycler))
    plt.plot(lambdas["cor02"], P2s["cor02"], next(linecycler))
    plt.plot(lambdas["cor04"], P2s["cor04"], next(linecycler))
    plt.ylabel(r'$P_{loss}^{(2)}$')
    plt.xlabel(r'$\lambda$')
    plt.legend((r'$c_{cor}^{(1)}$ =' + str(abs(round(qs1.queries_stream.c_cor[0], 3))) + r', $c_{cor}^{(2)}$=' + str(
        abs(round(qs1.queries_stream.c_cor[1], 3))),
                r'$c_{cor}^{(1)}$ =' + str(round(qs2.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs2.queries_stream.c_cor[1], 3)),
                r'$c_{cor}^{(1)}$ =' + str(round(qs3.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs3.queries_stream.c_cor[1], 3))
                ),
               loc=0)
    if PRINT_TITLE:
        plt.title('Зависимость P2_loss от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP5_P2_loss_lambda_cor_0_02_04_{}.png", nowdate),
                dpi=400,
                bbox_inches='tight')
    plt.close()

    plt.plot(lambdas["poisson"], Ps_imp["poisson"], next(linecycler))
    plt.plot(lambdas["cor02"], Ps_imp["cor02"], next(linecycler))
    plt.plot(lambdas["cor04"], Ps_imp["cor04"], next(linecycler))
    plt.ylabel(r'$P_{loss}^{(imp)}$')
    plt.xlabel(r'$\lambda$')
    plt.legend((r'$c_{cor}^{(1)}$ =' + str(abs(round(qs1.queries_stream.c_cor[0], 3))) + r', $c_{cor}^{(2)}$=' + str(
        abs(round(qs1.queries_stream.c_cor[1], 3))),
                r'$c_{cor}^{(1)}$ =' + str(round(qs2.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs2.queries_stream.c_cor[1], 3)),
                r'$c_{cor}^{(1)}$ =' + str(round(qs3.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs3.queries_stream.c_cor[1], 3))
                ),
               loc=0)
    if PRINT_TITLE:
        plt.title('Зависимость P_loss_imp от λ при различных\n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP5_P_loss_imp_lambda_cor_0_02_04_{}.png", nowdate),
                dpi=400,
                bbox_inches='tight')
    plt.close()

    plt.plot(lambdas["poisson"], w1s["poisson"], next(linecycler))
    plt.plot(lambdas["cor02"], w1s["cor02"], next(linecycler))
    plt.plot(lambdas["cor04"], w1s["cor04"], next(linecycler))
    plt.ylabel(r'$\bar{w}_1$')
    plt.xlabel(r'$\lambda$')
    plt.legend((r'$c_{cor}^{(1)}$ =' + str(abs(round(qs1.queries_stream.c_cor[0], 3))) + r', $c_{cor}^{(2)}$=' + str(
        abs(round(qs1.queries_stream.c_cor[1], 3))),
                r'$c_{cor}^{(1)}$ =' + str(round(qs2.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs2.queries_stream.c_cor[1], 3)),
                r'$c_{cor}^{(1)}$ =' + str(round(qs3.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs3.queries_stream.c_cor[1], 3))
                ),
               loc=0)
    if PRINT_TITLE:
        plt.title(
            'Зависимость $bar{w}_1$ от $lambda$ при различных \n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP5_w1_lambda_cor_{}.png", nowdate),
                dpi=400,
                bbox_inches='tight')
    plt.close()

    plt.plot(lambdas["poisson"], w2s["poisson"], next(linecycler))
    plt.plot(lambdas["cor02"], w2s["cor02"], next(linecycler))
    plt.plot(lambdas["cor04"], w2s["cor04"], next(linecycler))
    plt.ylabel(r'$\bar{w}_2$')
    plt.xlabel(r'$\lambda$')
    plt.legend((r'$c_{cor}^{(1)}$ =' + str(abs(round(qs1.queries_stream.c_cor[0], 3))) + r', $c_{cor}^{(2)}$=' + str(
        abs(round(qs1.queries_stream.c_cor[1], 3))),
                r'$c_{cor}^{(1)}$ =' + str(round(qs2.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs2.queries_stream.c_cor[1], 3)),
                r'$c_{cor}^{(1)}$ =' + str(round(qs3.queries_stream.c_cor[0], 3)) + r', $c_{cor}^{(2)}$=' + str(
                    round(qs3.queries_stream.c_cor[1], 3))
                ),
               loc=0)
    if PRINT_TITLE:
        plt.title(
            'Зависимость $bar{w}_2$ от $lambda$ при различных \n коэффициентах корреляции длин двух соседних интервалов')

    plt.savefig(str.format("plots/BMMAP5_w2_lambda_cor_{}.png", nowdate),
                dpi=400,
                bbox_inches='tight')
    plt.close()

    table_poisson = np.asarray(table_poisson)
    table_cor_02 = np.asarray(table_cor_02)
    table_cor_04 = np.asarray(table_cor_04)

    np.savetxt(str.format("tables/BMMAP5_P_losses_lambda_cor_poisson_{}.csv", nowdate), table_poisson,
               header=r'$\lambda$, $P_{loss}^{(1)}$, $P_{loss}^{(2)}$, $P_{loss}, P_{loss}^{(imp)}$, $\bar{w}_1$, $\bar{w}_2$',
               delimiter=',',
               fmt='%.5f', comments='')
    np.savetxt(str.format("tables/BMMAP5_P_losses_lambda_cor_02_{}.csv", nowdate), table_cor_02,
               header=r'$\lambda$, $P_{loss}^{(1)}$, $P_{loss}^{(2)}$, $P_{loss}, P_{loss}^{(imp)}$, $\bar{w}_1$, $\bar{w}_2$',
               delimiter=',',
               fmt='%.5f', comments='')
    np.savetxt(str.format("tables/BMMAP5_P_losses_lambda_cor_04_{}.csv", nowdate), table_cor_04,
               header=r'$\lambda$, $P_{loss}^{(1)}$, $P_{loss}^{(2)}$, $P_{loss}, P_{loss}^{(imp)}$, $\bar{w}_1$, $\bar{w}_2$',
               delimiter=',',
               fmt='%.5f', comments='')


if __name__ == '__main__':
    main()
