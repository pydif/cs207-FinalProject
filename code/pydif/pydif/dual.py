class Dual():

    def __init__(self, val , der):
        self.val = val
        self.der = der

    def __add__(self, x):
        self.val +=  x.val
        self.der +=  x.der
        return self

    def __sub__(self, x):
        self.val -=  x.val
        self.der -=  x.der
        return self

    def __radd__(self, x):
        return self.__add__(x)

    def __mul__(self, x):
        self.val *= x.val
        self.der = self.der* x.val + self.val * x.der
        return self

    def __rmul__(self, x):
        return self.__mul__(x)

    def __truediv__(self, x):
        self.val /= x.val
        self.der = (self.der* x.val - self.val * x.der)/(x.val)**2
        return self

    def __pow__(self, x):
        self.val = self.val**x.val
        self.der = x.der*self.val**x.val
        return self

    def __repr__(self):
        return '[{0},{1}]'.format(self.val, self.der)
