import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'code','pydif'))
from pydif.pydif import autodiff
from pydif.optimize.optimize import Optimize


def f(x, y):
	return x**2 + y**2


x_opt = Optimize(f)

min_pos = x_opt.gradient_descent((7, 8))

print(min_pos)