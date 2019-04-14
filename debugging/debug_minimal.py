import debugging.MMAP_cor04 as cor04
import two_priorities_qs as qs4

if __name__ == '__main__':
    test_data = cor04.Mmap04PhPh()
    # test_data = cor02.Mmap02PhPh()
    # test_data = cor0.MmapPoissonPh()

    qs1 = qs4.TwoPrioritiesQueueingSystem(test_data, verbose=True)

    qs1.calc_characteristics(True)
