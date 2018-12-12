import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'code','pydif'))
from pydif.pydif import autodiff
import scipy.optimize
from pydif.optimize.optimize import Optimize


def f1(x, y):
    return np.array(
        [((x**4 + y**4)**(1/4)) - 1,
         ((((3*x) - (y))**4 + (x)**4)**(1/4)) - 1])

def f(x,y):
	return (1-x)**2 + (y-x**2)**2


x_opt = Optimize(f1)
x1_opt = Optimize(f)

min_pos = x_opt.newton(np.array([1,1]))
print(min_pos)
min_pos = x_opt.newton(np.array([-1,0]))
print(min_pos)
min_pos = x_opt.newton(np.array([0,1]))
print(min_pos)
min_pos = x1_opt.gradient_descent((1,-1))
print(min_pos)
