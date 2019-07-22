# Queueing System with two Priorities characteristics calculator

__Tested Python version: 3.6__

This set of scripts was created to calculate QS performance characteristics depending on initial parameters.

It consists of several "main" scripts and lots of another, which serve to experiments.

## Main scripts

   1. `streams.py` - Here are classes that are used to describe QS streams. Here you can find 
   `MAP`, `BMAP`, `BMMAP`, `PH` streams.
   
        Only `BMMAP` and `PH` are used in this project. Other streams were just copied from earlier work.
        
        Some of main stream's characteristics are calculated here and can be accessed later.
        
        Unfortunately there is no single interface for these streams. But there are some docstrings that should help.
   
   2. `utils.py` - Some useful util functions are located here:
        * `kron(A, B)` - Kronecker product
        * `kronsum(A, B)` - Kronecker sum
        * `kronpow(A, pow)` - Kronecker power
        * `kronsumpow(A, pow)` - Kronecker "sum power" (as I don't know true name of this operator)
        * `system_solve(matr)` - Function that is used to solve systems such as `(vect * matr = 0) & (vect * e = 1)`
        * `r_multiply_e(matr)` - Right multiplication by unitary vector column
        * `e_col(dim)` - Function that generates unitary vector column of given dimension
        * `copy_matrix_block(dest, src, m_pos, n_pos)` - Function that copies matrix block
        * `ncr(n, r)` - $C_{n}^{r}$
        * `m_exp(matr, t)` - Matrix exponential
        * etc.
      
        They are documented pretty well in the code.
   3. `ramaswami.py` - Ramaswami matrices are calculating here.
   
        `calc_ramaswami_matrices(matr_S, matr_tilde_S, vect_beta, N, verbose=False)` is the main function here.
        
        This code was adapted from Java code written by the one who knows this algorithm and stays some kind of black box for me.
        
   4. `matr_E.py` - Special E_{i, j} matrices are calculating here. (_Description to be added_)
   5. `two_priorities_qs.py` (_runnable_) - All the magic is here.
   
   ## Additional scripts
   Packages will be described here along with common files format for experiments.
* `experiments_data` - Single experiment data files are here.
    
   File contents should look like the following (`BMMAP3_02_PH_PH.py`):
    
```python
import numpy as np


class Bmmap302PhPh:
    """
    Experiment with BMMAP queries stream with max group of 3
    and max(c_cor) ~ 0.2
    with PH serve time
    and PH timer   
    """
    
    def __init__(self, matrS_elem=20):
        # BMMAP
        self.test_matrD_0 = np.array([[-5.408, 0.],
                                      [0., -0.1755]]) * 4.319328938619707

        self.test_matrD = np.array([[5.372, 0.036],
                                    [0.09772, 0.07778]]) * 4.319328938619707

        self.test_qs = (0.8, 0.2)
        self.test_ns = (5, 2)

        self.priority_part = 0.9

        self.test_N = 10
        self.p_hp = 0.4

        # Server PH
        self.test_vect_beta = np.array([[1., 0.]])
        self.test_matrS = np.array([[-matrS_elem, matrS_elem],
                                    [0., -matrS_elem]])

        # Timer PH
        self.test_vect_gamma = np.array([[1., 0.]])
        self.test_matrGamma = np.array([[-10., 10.],
                                        [0., -10.]])
```
To launch this single experiment you should modify `twopriorities_qs.py`:
1. replace any imported as test with `import experiments_data.<file_name> as test`
2. under `if __name__ == '__main__':` section at the bottom of the file replace
 `test_data = test.Bmmap3PoissonPhPh()` with
 `test_data = test.<class_name_from_imported_file>`
3. run `two_priorities_qs.py`

This way you'll get characteristics of the QS with parameters described in experiments script calculated and printed out.
