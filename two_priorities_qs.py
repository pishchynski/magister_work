import gc
import sys

sys.path.append("../")
from utils import *

np.set_printoptions(threshold=np.inf, suppress=True, formatter={'float': '{: 0.8f}'.format}, linewidth=75)


class TwoPrioritiesQueueingSystem:


    def __init__(self, name='Default system', p_max_num=100):
        self.name = name
        self._p_max_num = p_max_num