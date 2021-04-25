import numpy as np
import math
from pair_potential.pair_potential import fast_pair_potential
from pair_potential.lj_potential import vectorised_lennard_jones_potential
from local_optimisation.newton_raphson import newton_raphson_multi
from calculus.difference.difference import first_central_multi, second_central_multi
from local_optimisation.bfgs import bfgs

x = np.array([[1.0132226417, 0.3329955686, 0.1812866397],
              [0.7255989775, -0.7660449415, 0.2388625373],
              [0.7293356067, -0.2309436666, -0.7649239428],
              [0.3513618941, 0.8291166557, -0.5995702064],
              [0.3453146118, -0.0366957540, 1.0245903005],
              [0.1140240770, 0.9491685999, 0.5064104273],
              [-1.0132240213, -0.3329960305, -0.1812867552],
              [-0.1140234764, -0.9491689127, -0.5064103454],
              [-0.3513615244, -0.8291170821, 0.5995701458],
              [-0.3453152548, 0.0366956843, -1.0245902691],
              [-0.7255983925, 0.7660457628, -0.2388624662],
              [-0.7293359733, 0.2309438428, 0.7649237858],
              [0.0000008339, 0.0000002733, 0.0000001488]])


def first_diff(r):
    return first_central_multi(lj_potential, r, 0.00001)


def second_diff(r):
    return second_central_multi(lj_potential, r, 0.00001)


def lj_potential(x):
    return fast_pair_potential(x, potential=vectorised_lennard_jones_potential, args=())





if __name__ == '__main__':
    rarray = bfgs(lj_potential, x, first_diff, 1e-5)
    print(rarray)

    def test_function(x):
        y = x * x
        return float(y.sum())

    def first_diff_test(r):
        return first_central_multi(test_function, r, 0.00001)


    def second_diff_test(r):
        return second_central_multi(test_function, r, 0.00001)

    original_x = np.array([[0,1],[2,3]],dtype=float)
    result=bfgs(test_function,original_x,first_diff_test,1e-7)
    print(result)
