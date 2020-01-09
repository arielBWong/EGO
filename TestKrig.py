from smt.surrogate_models import KRG
import numpy as np

from numpy import genfromtxt

x = genfromtxt('x.csv', delimiter=',')
y = genfromtxt('y.csv', delimiter=',')
x = np.atleast_2d(x)
y = np.atleast_2d(y).reshape(-1, 1)

sm = KRG(theta0=[1e-2], print_global=False)
sm.set_training_values(x, y)
sm.train()



