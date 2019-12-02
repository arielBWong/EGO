import numpy as np
from sklearn.utils.validation import check_array
import pyDOE
# import matplotlib.pyplot as plt


def Branin_mesh(x1, x2):
    # in ths function, we suppose that x and y is in the right  range
    # the returned value f should be 2d mesh compatible values
    a = 1.0
    b = 5.1/(4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0/(8.0 * np.pi)

    part1 = a * (x2 - b * x1 ** 2 + c * x1 - 6.0) ** 2.0
    part2 = s * (1 - t) * np.cos(x1)
    part3 = s

    f = part1 + part2 + part3
    return f



def Branin(x):

    # problem definition
    # https://www.sfu.ca/~ssurjano/branin.html

    x = check_array(x)
    x1 = np.atleast_2d(x[:, 0]).reshape(-1, 1)
    x2 = np.atleast_2d(x[:, 1]).reshape(-1, 1)

    # assume that values in x is in range [0, 1]
    if np.any(x > 1) or np.any(x < 0):
        raise Exception('Branin input should be in range [0, 1]')
        exit(1)

    x1 = -5 + (10 - (-5)) * x1
    x2 = 0 + (15 - 0) * x2

    a = 1.0
    b = 5.1/(4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0/(8.0 * np.pi)

    part1 = a * (x2 - b * x1**2 + c * x1 - r)**2.0
    part2 = s * (1 - t) * np.cos(x1)
    part3 = s

    f = part1 + part2 + part3
    x[:, 0] = -5 + (10 - (-5)) * x[:, 0]
    x[:, 1] = 0 + (15 - 0) * x[:, 1]

    return x, f


def Branin_after_init(x):

    x = check_array(x)
    x1 = np.atleast_2d(x[:, 0]).reshape(-1, 1)
    x2 = np.atleast_2d(x[:, 1]).reshape(-1, 1)

    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    part1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2.0
    part2 = s * (1 - t) * np.cos(x1)
    part3 = s

    f = part1 + part2 + part3

    return x, f




if __name__ == "__main__":
    lhs_x = pyDOE.lhs(2, 100)
    z = Branin(lhs_x)
    x = -5 + (10 - (-5)) * lhs_x[:, 0]
    y = 0 + (15 - 0) * lhs_x[:, 1]

    # this main is used to test whether Branin function is created right
    x = np.atleast_2d(np.linspace(-5, 10, 500)).reshape(-1, 1)
    y = np.atleast_2d(np.linspace(0, 15, 500)).reshape(-1, 1)

    x_mesh, y_mesh = np.meshgrid(x, y)
    z = Branin_mesh(x_mesh, y_mesh)



    levels = np.linspace(np.amin(z), np.amax(z), 50)
    plt.contour(x_mesh, y_mesh, z, levels=levels)
    plt.title('Branin function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()




