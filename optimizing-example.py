import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'code','pydif'))
from pydif.pydif import autodiff
from pydif.optimize.optimize import Optimize


def f1(x, y):
    return np.array(
        [((x**4 + y**4)**(1/4)) - 1,
         ((((3*x) - (y))**4 + (x)**4)**(1/4)) - 1])


x_opt = Optimize(f1)

min_pos = x_opt.newton(np.array([1,1]))
print(min_pos)
min_pos = x_opt.newton(np.array([-1,0]))
print(min_pos)
min_pos = x_opt.newton(np.array([0,1]))
print(min_pos)
min_pos = x_opt.newton(np.array([0,-1]))
print(min_pos)
