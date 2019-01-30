import debugging.MMAP_Poisson_PH_Poisson_NoRemoval as poisson_PH_noremoval
import two_priorities_qs as qs

if __name__ == '__main__':
    test_data_Poisson_PH_noRemoval = poisson_PH_noremoval.MmapPoissonPhPoissonNoRemoval()

    qs1 = qs.TwoPrioritiesQueueingSystem(test_data_Poisson_PH_noRemoval, verbose=True)

    qs1.calc_characteristics(True)

    # test_data_Poisson_noRemoval = poisson_noremoval.MmapPoissonNoRemoval()
    #
    # print("\nPoisson system:\n")
    #
    # qs2 = qs.TwoPrioritiesQueueingSystem(test_data_Poisson_noRemoval, verbose=True)
    #
    # qs2.calc_characteristics(True)


