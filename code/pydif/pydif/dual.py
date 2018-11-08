class Dual():

    def __init__(self, val , der):
        self.val = val
        self.der = der

    def __add__(self, x):
        try:
            self.val +=  x.val
            self.der +=  x.der
        except AttributeError:
            self.val += x
            self.der += x
        return self

    def __sub__(self, x):
        try:
            self.val -=  x.val
            self.der -=  x.der
        except AttributeError:
            self.val -= x
            self.der -= x
        return self

    def __radd__(self, x):
        return self.__add__(x)

    def __mul__(self, x):
        try:
            self.der = self.der * x.val + self.val * x.der
            self.val *= x.val
        except AttributeError:
            self.val *= x
            self.der *= x
        return self

    def __rmul__(self, x):
        return self.__mul__(x)

    def __truediv__(self, x):
        try:
            self.der = (self.der* x.val - self.val * x.der)/(x.val)**2
            self.val /= x.val
        except AttributeError:
            self.val /= x
            self.der /= x
        return self

    def __pow__(self, x):
        try:
            self.der = x.der*self.val**x.val
            self.val = self.val**x.val
        except AttributeError:
            self.val = self.val**x
            self.der = self.der**x
        return self

    def __repr__(self):
        return '[{0},{1}]'.format(self.val, self.der)
