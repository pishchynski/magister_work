import debugging.MMAP_cor02 as cor02
import two_priorities_qs as qs

if __name__ == '__main__':
    # test_data = cor04.Mmap04PhPh()
    test_data = cor02.Mmap02PhPh()

    qs1 = qs.TwoPrioritiesQueueingSystem(test_data, verbose=True)

    qs1.calc_characteristics(True)
