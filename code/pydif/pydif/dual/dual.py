import numpy as np
class Dual():

    def __init__(self, val , der):
        self.val = val
        self.der = np.array(der)

    def __add__(self, x):
        try:
            return Dual(self.val +  x.val, self.der +  x.der)
        except AttributeError:
            return Dual(self.val +  x, self.der)


    def __sub__(self, x):
        try:
            return Dual(self.val -  x.val, self.der -  x.der)
        except AttributeError:
            return Dual(self.val -  x, self.der)

    def __radd__(self, x):
        return self.__add__(x)

    def __mul__(self, x):
        try:
            return Dual(self.val * x.val, self.der * x.val + self.val * x.der)
        except AttributeError:
            return Dual(self.val *  x, self.der *  x)

    def __rmul__(self, x):
        return self.__mul__(x)

    def __truediv__(self, x):
        try:
            return Dual(self.val/ x.val, (self.der* x.val - self.val * x.der)/(x.val)**2)
        except AttributeError:
            return Dual(self.val /x, self.der / x)

    def __rtruediv__(self, x):
        return x * self**-1

    def __pow__(self, x):
        try:
            return Dual(self.val**x.val, self.val**x.val*(self.der*(x.val/self.val)+x.der*np.log(self.val)))
        except AttributeError:
            print('this loop')

            return Dual(self.val**x, self.val**x*(self.der*(x)/(self.val)))

    def __rpow__(self, x):
        try:
            return Dual(self.val**x, self.val**x*x.der*np.log(self.val))
            #raise AttributeError
        except AttributeError:
            print('that loop')
            print('x: {}'.format(x))
            return Dual(x**self.val, x**self.val*(self.der*np.log(x)))

    def __neg__(self):
        try:
            return Dual(-self.val, -self.der)
        except AttributeError:
            return -self

    def __repr__(self):
        return '[{0},{1}]'.format(self.val, self.der)
