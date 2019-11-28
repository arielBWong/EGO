import numpy as np
from sklearn.utils.validation import check_array



def Branin(x):

    # problem definition
    # https://www.sfu.ca/~ssurjano/branin.html

    x = check_array(x)
    x1 = x[:, 0]
    x2 = x[:, 1]

    a = 1.0
    b = 5.1/(4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0/(8.0 * np.pi)

    part1 = a * (x2 - b * x1**2 + c * x1 - 6.0)**2.0
    part2 = s * (1 - t) * np.cos(x1)
    part3 = s

    f = part1 + part2 + part3

    return f

