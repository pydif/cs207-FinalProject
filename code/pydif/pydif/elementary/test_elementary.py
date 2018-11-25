from pydif.elementary import elementary as el
import pytest
# from ..dual.dual import Dual 
from pydif.dual.dual import Dual

def test_cos():
    assert(el.cos(5) == pytest.approx(0.2836, 0.001))
    x = Dual(5, 8)
    cosx = el.cos(x)
    assert(cosx.val == pytest.approx(0.2836, 0.001)) 
    assert(cosx.der == pytest.approx(7.671, 0.001))

def test_sin():
    assert(el.sin(3) == pytest.approx(0.1411, 0.001))
    x = Dual(3, 3)
    sinx = el.sin(x)
    assert(sinx.val == pytest.approx(0.1411, 0.001)) 
    assert(sinx.der == pytest.approx(-2.9699, 0.001))

def test_tan():
    assert(el.tan(2) == pytest.approx(-2.185, 0.001))
    x = Dual(2, 3)
    tanx = el.tan(x)
    assert(tanx.val == pytest.approx(-2.185, 0.001))
    assert(tanx.der == pytest.approx(17.32319, 0.001))

def test_exp():
    assert(el.exp(0) == pytest.approx(1))
    assert(el.exp(5)) == pytest.approx(148.413159, 0.0001)
    x = Dual(3, 4)
    ex = el.exp(x)
    assert(ex.val == pytest.approx(20.0855369, 0.0001))
    assert(ex.der == pytest.approx(80.3421476, 0.0001))

def test_arcsin():
    assert(el.arcsin(1) == pytest.approx(1.570, 0.001))
    x = Dual(0.5, 2)
    arcsinx = el.arcsin(x)
    assert(arcsinx.val == pytest.approx(0.5235, 0.001))
    assert(arcsinx.der == pytest.approx(1.154*2, 0.001))

def test_arccos():
    assert(el.arccos(0.9) == pytest.approx(0.451, 0.001))
    x = Dual(0.5, 2)
    arccosx = el.arccos(x)
    assert(arccosx.val == pytest.approx(1.047, 0.001))
    assert(arccosx.der == pytest.approx( -2 *1.1547, 0.001))

def test_arctan():
    assert(el.arctan(0.75) == pytest.approx(0.6435, 0.001))
    x = Dual(0.5, 2)
    arctanx = el.arctan(x)
    assert(arctanx.val == pytest.approx(0.4636, 0.001))
    assert(arctanx.der == pytest.approx(2 *0.8, 0.001))

def tesh_sinh():
    assert(el.sinh(0.7) == pytest.approx(0.7585, 0.001))
    x = Dual(0.5, 2)
    sinhx = el.sinh(x)
    assert(sinhx.val == pytest.approx(0.5210, 0.001))
    assert(sinhx.der == pytest.approx(2 * 1.127, 0.001))

def test_cosh():
    assert(el.cosh(0.6) == pytest.approx(1.185, 0.001))
    x = Dual(0.5, 2)
    coshx = el.cosh(x)
    assert(coshx.val == pytest.approx(1.127, 0.001))
    assert(coshx.der == pytest.approx(2 * 0.52109, 0.001))

def test_tanh():
    assert(el.tanh(0.5) == pytest.approx(0.462, 0.001))
    x = Dual(0.5, 2)
    tanhx = el.tanh(x)
    assert(tanhx.val == pytest.approx(0.462, 0.001))
    assert(tanhx.der == pytest.approx(2*0.7864, 0.001))

def test_exp2():
    assert(el.exp2(0) == pytest.approx(1))
    assert(el.exp2(4) == pytest.approx(16))
    x = Dual(4, 4)
    x2 = el.exp2(x)
    assert(x2.val == pytest.approx(16))
    assert(x2.der == pytest.approx(44.361, 0.001))

def test_log():
    assert(el.log(27) == pytest.approx(3.2958, 0.001))
    x = Dual(3, 5)
    lnx = el.log(x)
    assert(lnx.val == pytest.approx(1.098, 0.001))
    assert(lnx.der == pytest.approx(1.6666, 0.001))

def test_log2():
    assert(el.log2(8) == pytest.approx(3))
    x = Dual(4, 8)
    log2x = el.log2(x)
    assert(log2x.val == pytest.approx(2))
    assert(log2x.der == pytest.approx(2.8853, 0.001))

def test_log10():
    assert(el.log10(100) == pytest.approx(2))
    x = Dual(10, 30)
    log10x = el.log10(x)
    assert(log10x.val == pytest.approx(1))
    assert(log10x.der == pytest.approx(1.30288, 0.001))