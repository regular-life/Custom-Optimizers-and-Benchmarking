from .multiple_local_minima import *
from .bowl_shaped import *
from .plate_shaped import *
from .valley_shaped import *
from .steep_ridges import *
from .others import *

__all__ = [
    # multiple_local_minima
    'ackley', 'bukin_n6', 'cross_in_tray', 'drop_wave', 'eggholder',
    'gramacy_lee', 'griewank', 'holder_table', 'langermann',
    'levy', 'levy_n13', 'rastrigin', 'schaffer_n2',
    'schaffer_n4', 'schwefel', 'shubert',
    # bowl_shaped
    'bohachevsky', 'perm_0', 'rotated_hyper_ellipsoid',
    'sphere', 'sum_different_powers', 'sum_squares', 'trid',
    # plate_shaped
    'booth', 'matyas', 'mccormick', 'power_sum', 'zakharov',
    # valley_shaped
    'three_hump_camel', 'six_hump_camel', 'dixon_price', 'rosenbrock',
    # steep_ridges
    'de_jong_n5', 'easom', 'michalewicz',
    # others
    'beale', 'branin', 'colville', 'forrester', 'goldstein_price',
    'hartmann3d', 'hartmann4d', 'hartmann6d',
    'perm', 'powell', 'shekel', 'styblinski_tang',
]
