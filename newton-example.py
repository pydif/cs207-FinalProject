import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'code','pydif'))
from pydif.pydif import autodiff
from pydif.elementary import elementary as el

#adapted from lecture 9 slides from CS207
def f(x):
    return x - el.exp(-2.0 * el.sin(4.0*x) * el.sin(4.0*x))

dfdx = autodiff(f)

# Start Newton algorithm
xk = 1.0 # Initial guess
tol = 1.0e-08 # Some tolerance
max_it = 100 # Just stop if a root isn't found after 100 iterations

root = None # Initialize root
for k in range(max_it):
    delta_xk = -f(xk) / dfdx.get_der(xk, jacobian=True)[0] #there is only 1 partial derivative that we care about
    if (abs(delta_xk) <= tol): # Stop iteration if solution found
        root = xk + delta_xk
        print("Found root at x = {0:17.16f} after {1} iteratons.".format(root, k+1))
        break
    print("At iteration {0}, Delta x = {1:17.16f}".format(k+1, delta_xk))
    xk += delta_xk # Update xk

print('The calculated root is close to 0: {}.'.format(np.isclose(0.0,f(xk))))