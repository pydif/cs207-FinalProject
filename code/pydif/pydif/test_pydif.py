import pytest
import numpy as np
#import pydif
from . import pydif
from . import elementaryDEPRECATED as ele
#import .elementaryDEPRECATED as ele


def test_multiply_add_simple():
    alpha = 2
    pos = 2

    def f1(x):
        return alpha*x+3
    def f2(x):
        return x*alpha+3
    def f3(x):
        return 3+x*alpha
    def f4(x):
        return 3+alpha*x

    ad1 = pydif.autodiff(f1)
    ad2 = pydif.autodiff(f2)
    ad3 = pydif.autodiff(f3)
    ad4 = pydif.autodiff(f4)

    assert(ad1.get_val(pos) == 7)
    assert(ad2.get_val(pos) == 7)
    assert(ad3.get_val(pos) == 7)
    assert(ad4.get_val(pos) == 7)

    assert(ad1.get_der(pos) == 2)
    assert(ad2.get_der(pos) == 2)
    assert(ad3.get_der(pos) == 2)
    assert(ad4.get_der(pos) == 2)


def test_divide_add_simple():
    alpha = 2
    pos = 2

    def f1(x):
        return alpha/x+3
    def f2(x):
        return 3+alpha/x
    def f3(x):
        return x/alpha+3
    def f4(x):
        return 3+x/alpha

    ad1 = pydif.autodiff(f1)
    ad2 = pydif.autodiff(f2)
    ad3 = pydif.autodiff(f3)
    ad4 = pydif.autodiff(f4)

    assert(ad1.get_val(pos) == 4)
    assert(ad2.get_val(pos) == 4)
    assert(ad3.get_val(pos) == 4)
    assert(ad4.get_val(pos) == 4)

    assert(ad1.get_der(pos) == -0.5)
    assert(ad2.get_der(pos) == -0.5)
    assert(ad3.get_der(pos) == 0.5)
    assert(ad4.get_der(pos) == 0.5)

def test_composite():
    pos = (1,2,3)

    def f1(x,y,z):
        return (1/(x*y*z)) + ele.sin((1/x) + (1/y) + (1/z))
    def f2(x,y,z):
        return (1/(x*y*z))

    ad1 = pydif.autodiff(f1)
    ad2 = pydif.autodiff(f2)

    expected_val = (1/6) + np.sin(11/6)

    assert(ad1.get_val(pos) == expected_val)

    der1 = -(1/6) - np.cos(11/6) #d/dx
    der2 = -(1/12) - (np.cos(11/6)/4) #d/dy
    der3 = -(1/18) - (np.cos(11/6)/9) #d/dz

    assert(np.allclose(ad1.get_der(pos,jacobian=True),[der1,der2,der3]))

    assert(ad2.get_val(pos) == 1/(1*2*3))
    #http://www.wolframalpha.com/input/?i=d%2Fda+(1%2F(x(a)*y(a)*z(a))
    expected_der = -((2*3 + 1*3 + 1*2)/((1**2) * (2**2) * (3**2)))
    assert(np.isclose(float(ad2.get_der(pos)), expected_der))