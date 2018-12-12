import numpy as np
import sys
import matplotlib.pyplot as plt
import os
sys.path.append(os.path.join(os.getcwd(),'code','pydif'))
from pydif.pydif import autodiff
from pydif.optimize.optimize import Optimize


def f(x, y):
	return ((1-x)**2 + 100*(y-x**2)**2)

x_opt = Optimize(f)

#function that allows for numerous initial conditions to be specified and plotted
def plot_optimization(optimizer, func, initial_cond, name):

    #define initial conditions and plot results
    min_val, hist = optimizer(initial_cond, return_hist = True) #call optimization function
    hist = np.array(hist)

    #format plot
    xs = np.linspace(-5,5,1000)
    ys = np.linspace(-5,5,1000)
    X, Y = np.meshgrid(xs, ys)
    Z = func(X, Y) # TODO f is the function we  are optimizing
    plt.contour(X, Y, Z, 100)
    plt.plot(hist[:, 0], hist[:, 1])
    plt.ylim((ys[0],ys[-1]))
    plt.xlim((xs[0],xs[-1]))
    plt.xlabel('x')
    plt.title(name)
    plt.ylabel('y')
    plt.savefig(name)
    plt.close()

plot_optimization(x_opt.gradient_descent, x_opt.func, (0, 1), 'gradient_descent')
plot_optimization(x_opt.BFGS, x_opt.func, (0, 1), 'BFGS')
plot_optimization(x_opt.steepest, x_opt.func, (0, 1), 'steepest')
