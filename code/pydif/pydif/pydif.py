from inspect import signature
import numpy as np
import collections
from dual import dual

class autodiff():

    def __init__(self, func):
        self.func = func
        self.num_params = len(signature(func).parameters)

    def _eval(self, pos, jacobian = False):
        #print(pos)
        if isinstance(pos, collections.Iterable):
            if jacobian:
                params = []
                for cursor, pos_i in enumerate(pos):
                    der_clean = np.zeros(self.num_params)
                    der_clean[cursor] = 1
                    params.append(dual.Dual(pos_i, der_clean))
                #print(params)
                #print(*params)
                return self.func(*params)
            else:
                params = [dual.Dual(pos_i,1) for pos_i in pos]
                return self.func(*params)
        else:
            if jacobian:
                params = dual.Dual(pos,[1])
                return self.func(params)
            else:
                params = dual.Dual(pos,1)
                return self.func(params)                

    def _check_pos(self, pos):
        if isinstance(pos, collections.Iterable):
            if len(pos) != self.num_params:
                raise ValueError('poorly formatted position. should be of length {}.'.format(self.num_params))

    def get_val(self, pos):
        self._check_pos(pos)
        res = self._eval(pos)
        if isinstance(res, collections.Iterable):
            return [i.val for i in res]
        else:
            return res.val
        # if isinstance(pos, collections.Iterable):
        #     return [i.val for i in self._eval(pos)]        
        # else:
        #     return self._eval(pos).val

    def get_der(self, pos, jacobian = False, dir=None):
        if dir is None:
            self._check_pos(pos)
            res = self._eval(pos, jacobian)

            if isinstance(res, collections.Iterable):
                return [i.der for i in res]
            else:
                return res.der
            # if jacobian:
            #     if isinstance(res, collections.Iterable):
            #         return [i.der for i in res]
            #     else:
            #         return res.der
            # else:
            #     if isinstance(res, collections.Iterable):
            #         return [i.der for i in res]
            #     else:
            #         return res.der

            # if isinstance(pos, collections.Iterable):
            #     return [i.der for i in self._eval(pos)]        
            # else:
            #     return self._eval(pos).der
        else:
            #seed vector
            raise NotImplementedError


if __name__ == '__main__':
    func = lambda x: x+1
    test = autodiff(func)