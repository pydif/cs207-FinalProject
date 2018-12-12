import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'code','pydif'))
from pydif.pydif import autodiff
from pydif.optimize.optimize import Optimize


def f(x, y):
	return ((1-x)**2 + 100*(y-x**2)**2)


x_opt = Optimize(f)

min_pos = x_opt.plot_optimization(x_opt.gradient_descent, (-1, 1))

print(min_pos)
