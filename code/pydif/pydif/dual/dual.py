"""dual.py
This file overloads operators dunder methods to allow for dual numbers to be used. The functions first try to work
with x as a dual number and falls back to treating x as a normal numerical type
Args:
    param1 (float): The value of a function at a point.
    param2 (array or array-like): An array of partial derivatives at a point

Returns:
    Dual Number Object: Returns a new Dual number object with the above parameters.

"""

import numpy as np
class Dual():

    #initialize derivative as a numpy array to handle partial derivatives
    def __init__(self, val , der):
        self.val = val
        self.der = np.array(der)

    #overload the addition method
    def __add__(self, x):
        try:
            return Dual(self.val +  x.val, self.der +  x.der)
        except AttributeError:
            return Dual(self.val +  x, self.der)

    #overload radd by calling addition on the Dual number
    def __radd__(self, x):
        return self.__add__(x)

    #overload the subtraction method
    def __sub__(self, x):
        try:
            return Dual(self.val -  x.val, self.der -  x.der)
        except AttributeError:
            return Dual(self.val -  x, self.der)

    #overload rsub
    def __rsub__(self, x):
        try:
            return Dual(self.val -  x.val, self.der -  x.der)
        except AttributeError:
            return Dual(x - self.val, self.der)

    #overload multiplication
    def __mul__(self, x):
        try:
            return Dual(self.val * x.val, self.der * x.val + self.val * x.der)
        except AttributeError:
            return Dual(self.val *  x, self.der *  x)

    #overload rmul by calling multiplication on the Dual number
    def __rmul__(self, x):
        return self.__mul__(x)

    #overload division
    def __truediv__(self, x):
        try:
            return Dual(self.val/ x.val, (self.der* x.val - self.val * x.der)/(x.val)**2)
        except AttributeError:
            return Dual(self.val /x, self.der / x)

    #overload rdivision by multiplying by the value to the negative first power
    def __rtruediv__(self, x):
        return x * self**-1

    #overload power operator using formula for derivative of a function raised to a function if both are dual numbers
    def __pow__(self, x):
        try:
            return Dual(self.val**x.val, self.val**x.val*(self.der*(x.val/self.val)+x.der*np.log(self.val)))
        except AttributeError:
            return Dual(self.val**x, self.val**x*(self.der*(x)/(self.val)))

    #overload rpow similarly to above
    def __rpow__(self, x):
        try:
            return Dual(self.val**x, self.val**x*x.der*np.log(self.val))
            #raise AttributeError
        except AttributeError:
            return Dual(x**self.val, x**self.val*(self.der*np.log(x)))

    #overload negation
    def __neg__(self):
        try:
            return Dual(-self.val, -self.der)
        except AttributeError:
            return -self

    #overload repr by displaying as a list where the first value is the value and the second is a derivative
    def __repr__(self):
        return '[{0},{1}]'.format(self.val, self.der)
