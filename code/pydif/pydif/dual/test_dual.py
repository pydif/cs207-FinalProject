import pytest
import numpy as np
from dual import Dual

def test_add():

    x = Dual(2, [1, 0])
    y = Dual(3, [0, 1])
    z = x + y

    assert(z.val == 5)
    assert(all(z.der == [1,1]))