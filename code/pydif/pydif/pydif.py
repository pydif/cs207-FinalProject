"""pydif.py
This file implements the autodiff class. The autodiff class is initialized with a function that accepts single variables of the form f(x,y,z,...).
The autodiff object then allows for the evaluation of the value, and derivatives of a function at a specified position.

Args:
    param1 (function object): The function whose value and derivatives are to be evaluated.

Returns:
    autodiff Object: Returns a new autodiff object for evaluation of the specified function.

"""

from inspect import signature
#from dual import dual
from dualDEPRECATED import Dual
import numpy as np
import collections

class autodiff():

    #initialize autodiff class with a function
    def __init__(self, func):
        self.func = func
        self.num_params = len(signature(func).parameters)

    #make function parameters dual numbers and evaluate function at a position
    def _eval(self, pos, jacobian = False):
        if isinstance(pos, collections.Iterable):
            if jacobian:
                params = []
                for cursor, pos_i in enumerate(pos):
                    der_partials = np.zeros(self.num_params)
                    der_partials[cursor] = 1
                    params.append(Dual(pos_i, der_partials))
            else:
                params = [Dual(pos_i,1) for pos_i in pos]
            return self.func(*params)
        else:
            if jacobian:
                params = Dual(pos,[1])
            else:
                params = Dual(pos,1)
            return self.func(params)

    #check that the specified position is the same shape as the function (specified at object creation) input
    def _check_pos(self, pos):
        if isinstance(pos, collections.Iterable):
            if len(pos) != self.num_params:
                raise ValueError('poorly formatted position. should be of length {}.'.format(self.num_params))

    #evaluate the value of the function at a specified position
    def get_val(self, pos):
        self._check_pos(pos)
        res = self._eval(pos)
        if isinstance(res, collections.Iterable):
            return [i.val for i in res]
        else:
            return res.val

    #evaluate the derivatives of the function at a specified position
    def get_der(self, pos, jacobian = False, dir=None):
        if dir is None:
            self._check_pos(pos)
            res = self._eval(pos, jacobian)

            if isinstance(res, collections.Iterable):
                return [i.der for i in res]
            else:
                return res.der
        else:
            #seed vector/directional derivatives
            raise NotImplementedError


if __name__ == '__main__':
    pass