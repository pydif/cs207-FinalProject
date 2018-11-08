from inspect import signature
import collections
from dual import Dual

class autodiff():

    def __init__(self, func):
        self.func = func
        self.num_params = len(signature(func).parameters)

    def _eval(self, pos):
        if isinstance(pos, collections.Iterable):
            params = [Dual(pos_i,1) for pos_i in pos]
        else:
            params = Dual(pos,1)
        return self.func(params)

    def _check_pos(self, pos):
        if isinstance(pos, collections.Iterable):
            if len(pos) != self.num_params:
                raise ValueError('poorly formatted position. should be of length {}.'.format(self.num_params))

    def get_val(self, pos):
        self._check_pos(pos)
        if isinstance(pos, collections.Iterable):
            return [i.val for i in self._eval(pos)]        
        else:
            return self._eval(pos).val

    def get_der(self, pos, dir=None):
        if dir is None:
            self._check_pos(pos)
            if isinstance(pos, collections.Iterable):
                return [i.der for i in self._eval(pos)]        
            else:
                return self._eval(pos).der
        else:
            #seed vector
            raise NotImplementedError


if __name__ == '__main__':
    func = lambda x: x+1
    test = autodiff(func)