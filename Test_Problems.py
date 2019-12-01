import numpy as np
from sklearn.utils.validation import check_array
import pyDOE
import matplotlib.pyplot as plt


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

    part1 = a * (x2 - b * x1**2 + c * x1 - 6.0)**2.0
    part2 = s * (1 - t) * np.cos(x1)
    part3 = s

    f = part1 + part2 + part3

    return f


if __name__ == "__main__":
    lhs_x = pyDOE.lhs(2, 5)
    z = Branin(lhs_x)
    x = -5 + (10 - (-5)) * lhs_x[:, 0]
    y = 0 + (15 - 0) * lhs_x[:, 1]

    x_mesh, y_mesh = np.meshgrid(x, y)


    fig = plt.figure(figsize=(6, 5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    cp = ax.contour(x_mesh, y_mesh, z)
    ax.clabel(cp, inline=True,
              fontsize=10)
    ax.set_title('Contour Plot')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    plt.show()




