import numpy as np
from sklearn.utils.validation import check_array
import pyDOE
import matplotlib.pyplot as plt
from joblib import dump, load


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


def branin(params, a=1., b=5.1 / (4. * np.pi**2), c=5. / np.pi, r=6., s=10., t=1. / (8. * np.pi)):
    x, y = params['x']['samples'][0], params['y']['samples'][0]
    result = a * (y - b * x**2 + c*x - r)**2 + s * (1 - t) * np.cos(x) + s
    params['branin'] = result
    return params


if __name__ == "__main__":
    '''
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
    '''

    gpr = load('Branin.joblib')

    x_sample = gpr.X_train_
    para_m = load('normal_p.joblib')

    x_mean = para_m['mean_x']
    x_std = para_m['std_x']

    z_mean = para_m['mean_y']
    z_std = para_m['std_y']

    x_t = np.linspace(-5, 10, 500)
    y_t = np.linspace(0,  15, 500)
    X, Y = np.meshgrid(x_t, y_t)
    Z = np.zeros((len(x_t), len(y_t)))

    for x_index, x in enumerate(x_t):
        for y_index, y in enumerate(y_t):
            input_gpr = np.atleast_2d([x, y])
            input_gpr = (input_gpr - x_mean)/x_std

            Z[x_index, y_index] = gpr.predict(input_gpr)

    # denomalize Z
    Z = Z * z_std + z_mean
    x_sample = x_sample * x_std + x_mean
    levels = np.linspace(np.amin(Z), np.amax(Z), 50)
    plt.contour(X, Y, Z, levels=levels)
    plt.scatter(x_sample[:, 0], x_sample[:, 1], c="red", marker="x")
    plt.title('Branin function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()





