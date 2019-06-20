from streams import *


if __name__ == '__main__':
    matrD_0 = np.array([[-86, 0.01],
                        [0.02, -2.76]])

    matrD = np.array([[85, 0.99],
                      [0.2, 2.54]])

    test_bmmap = BMMAPStream(matrD_0, matrD)
    test_bmmap.print_characteristics()

    test_bmap = BMAPStream(matrD_0, matrD)
    test_bmap.print_characteristics('D')