import sys
import os
import numpy as np
from inspect import signature
sys.path.append(os.path.join(os.getcwd(),'pydif'))
from pydif.pydif import autodiff

class optimize():
    def __init__(self, func):
        self.func = func

    def gradient_descent(self, init_pos, step_size=0.1, max_iters=100, precision=0.001):

        num_params = len(signature(self.func).parameters)
        badDimentionsMsg = 'poorly formatted initial position. should be of length {}.'.format(num_params)
        if num_params != len(init_pos):
            raise ValueError(badDimentionsMsg)

        cur_pos = init_pos
        iters = 0 
        dfdx = autodiff(self.func)
        val = dfdx.get_val(init_pos)
        prev_step_size = 100 + precision
        while (prev_step_size > precision and iters < max_iters):
            jac = dfdx.get_der(cur_pos, jacobian=True)
            prev_pos = cur_pos
            cur_pos = cur_pos - step_size * jac
            prev_step_size = np.linalg.norm(abs(cur_pos - prev_pos))
            iters += 1

        return cur_pos
