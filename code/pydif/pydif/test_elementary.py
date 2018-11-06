import elementary as el
import pytest
from dual import dual 

def test_cos();
	assert(el.cos(5) == pytest.approx(0.2836, 0.001))
	x = Dual(5, 8)
	sinx = el.cos(d)
	assert(sinx.val == pytest.approx(0.2836, 0.001)) 
	assert(sinx.der == pytest.approx(7.671, 0.001))

def test_sin():
	assert(el.sin(3) == pytest.approx(0.1411, 0.001))
	x = Dual(3, 3)
	cosx = el.cos(x)
	assert(cosx.val == pytest.approx(0.1411, 0.001)) 
	assert(cosx.der == pytest.approx(-2.9699, 0.001))

def test_tan():
	assert(el.tan(2) == pytest.approx(-2.185, 0.001))
	x = Dual(1, 3)
	tanx = el.tan(x)
	assert(tanx.val == pytest.approx(-2.185, 0.001))
	assert(tanx.der == pytest.approx(10.2765565, 0.001))

def test_exp():
	assert(el.exp(0) == pytest.approx(0))
	assert(el.exp(5)) == pytest.approx(148.413159, 0.0001)
	x = Dual(3, 4)
	ex = el.exp(x)
	assert(ex.val == pytest.approx(20.0855369, 0.0001))
	assert(ex.der = pytest.approx(80.3421476, 0.0001))


def test_exp2():
	assert(el.exp2(0) == pytest.approx(0))
	assert(el.exp2(4) == pytest.approx(16))
	x = Duel(4, 4)
	2x = el.exp2(x)
	assert(2x.val == pytest.approx(16))
	assert(2x.der == pytest.approx(44.361, 0.001))

def test_log():
	assert(el.log(27) == pytest.approx(3.2958, 0.0001))
	x = Duel(3, 5)
	lnx = el.log(x)
	assert(lnx.val == pytest.approx(1.098, 0.001))
	assert(lnx.der == pytest.approx(4.551, 0.001))

def test_log2():
	assert(el.log2(8) == pytest.approx(3))
	x = Duel(4, 8)
	log2x = el.log2(x)
	assert(log2x.val == pytest.approx(2))
	assert(log2x.der == pytest.approx(2.8853, 0.001))

def test_log10():
	assert(el.log10(100) == pytest.approx(10))
	x = Duel(10, 30)
	log10x = el.log10(x)
	assert(log10x.val == pytest.approx(1))
	assert(log10x.der == pytest.approx(1.30288))