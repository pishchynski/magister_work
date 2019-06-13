import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

import experiments.ts.mus.data.BMMAP5_02_PH_PH as cor02
import two_priorities_qs as qs


def main():
    if not os.path.exists("plots"):
        os.makedirs("plots")

    test_data_less_mu_initial = cor02.Bmmap502PhPh(matrS_elem=8.)
    test_data_less_mu = cor02.Bmmap502PhPh(matrS_elem=8.)

    test_data_equal_mu_initial = cor02.Bmmap502PhPh(matrS_elem=16.)
    test_data_equal_mu = cor02.Bmmap502PhPh(matrS_elem=16.)

    test_data_larger_mu_initial = cor02.Bmmap502PhPh(matrS_elem=24.)
    test_data_larger_mu = cor02.Bmmap502PhPh(matrS_elem=24.)

    qs1 = qs.TwoPrioritiesQueueingSystem(test_data_less_mu_initial)
    qs2 = qs.TwoPrioritiesQueueingSystem(test_data_equal_mu_initial)
    qs3 = qs.TwoPrioritiesQueueingSystem(test_data_larger_mu_initial)

    W1s = {"less_mu": [], "equal_mu": [], "larger_mu": []}
    W2s = {"less_mu": [], "equal_mu": [], "larger_mu": []}

    coef = 0.25 # as we have \lambda = 32 and want it to be 8
    ts = []

    for t in np.linspace(0.01, 3., 100):
        test_data_less_mu.test_matrD = test_data_less_mu_initial.test_matrD * coef
        test_data_less_mu.test_matrD_0 = test_data_less_mu_initial.test_matrD_0 * coef

        test_data_equal_mu.test_matrD = test_data_equal_mu_initial.test_matrD * coef
        test_data_equal_mu.test_matrD_0 = test_data_equal_mu_initial.test_matrD_0 * coef

        test_data_larger_mu.test_matrD = test_data_larger_mu_initial.test_matrD * coef
        test_data_larger_mu.test_matrD_0 = test_data_larger_mu_initial.test_matrD_0 * coef

        qs1 = qs.TwoPrioritiesQueueingSystem(test_data_less_mu)
        stationary_probas1 = qs1.calc_stationary_probas(False)

        qs2 = qs.TwoPrioritiesQueueingSystem(test_data_equal_mu)
        stationary_probas2 = qs2.calc_stationary_probas(False)

        qs3 = qs.TwoPrioritiesQueueingSystem(test_data_larger_mu)
        stationary_probas3 = qs3.calc_stationary_probas(False)

        ts.append(t)
        W1s["less_mu"].append(qs1.func_W_1(stationary_probas1, t))
        W2s["less_mu"].append(qs1.func_W_2(stationary_probas1, t))

        W1s["equal_mu"].append(qs2.func_W_1(stationary_probas2, t))
        W2s["equal_mu"].append(qs2.func_W_2(stationary_probas2, t))

        W1s["larger_mu"].append(qs3.func_W_1(stationary_probas3, t))
        W2s["larger_mu"].append(qs3.func_W_2(stationary_probas3, t))

    plt.plot(ts, W1s["less_mu"])
    plt.plot(ts, W1s["equal_mu"])
    plt.plot(ts, W1s["larger_mu"])
    plt.ylabel('W_1')
    plt.xlabel('t')
    plt.legend((str.format("µ={0}",
                           round(qs1.serv_stream.avg_intensity, 3)),
                str.format("µ={0}",
                           round(qs2.serv_stream.avg_intensity, 3)),
                str.format("µ={0}",
                           round(qs3.serv_stream.avg_intensity, 3)),
                ),
               loc=0)
    plt.title('Зависимость W_1 от t при различных\n средних интенсивностях обслуживания')

    plt.savefig(str.format("plots/BMMAP5_W_1_t_mu_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()

    plt.plot(ts, W2s["less_mu"])
    plt.plot(ts, W2s["equal_mu"])
    plt.plot(ts, W2s["larger_mu"])
    plt.ylabel('W_2')
    plt.xlabel('t')
    plt.legend((str.format("µ={0}",
                           round(qs1.serv_stream.avg_intensity, 3)),
                str.format("µ={0}",
                           round(qs2.serv_stream.avg_intensity, 3)),
                str.format("µ={0}",
                           round(qs3.serv_stream.avg_intensity, 3)),
                ),
               loc=0)
    plt.title('Зависимость W_2 от t при различных\n средних интенсивностях обслуживания')

    plt.savefig(str.format("plots/BMMAP5_W_2_t_mu_{}.png", datetime.datetime.now()),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
