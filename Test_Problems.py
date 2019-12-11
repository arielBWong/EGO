import numpy as np
import pyDOE
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.validation import check_array
from scipy.stats import norm, zscore


def Branin_mesh(x1, x2):
    # in ths function, we suppose that x and y is in the right  range
    # the returned value f should be 2d mesh compatible values
    a = 1.0
    b = 5.1/(4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0/(8.0 * np.pi)

    part1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2.0
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

# banzhuan version
def branin(params, a=1., b=5.1 / (4. * np.pi**2), c=5. / np.pi, r=6., s=10., t=1. / (8. * np.pi)):
    x, y = params['x']['samples'][0], params['y']['samples'][0]
    result = a * (y - b * x**2 + c*x - r)**2 + s * (1 - t) * np.cos(x) + s
    params['branin'] = result
    return params

def mean_std_save(data):
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)
    return m, s

def train_data_norm(train_x, train_y):
    mean_train_x, std_train_x = mean_std_save(train_x)
    mean_train_y, std_train_y = mean_std_save(train_y)
    #
    norm_train_x = zscore(train_x, axis=0)
    norm_train_y = zscore(train_y, axis=0)

    return mean_train_x, mean_train_y, std_train_x, std_train_y, norm_train_x, norm_train_y

if __name__ == "__main__":

    x = np.atleast_2d(np.linspace(-5, 10, 500)).reshape(-1, 1)
    y = np.atleast_2d(np.linspace(0, 15, 500)).reshape(-1, 1)

    x_mesh, y_mesh = np.meshgrid(x, y)
    z =  -(x_mesh - 10.) ** 2 - (y_mesh - 15.) ** 2

    x_sample = load('train_x.joblib')
    plt.scatter(x_sample[:, 0], x_sample[:, 1], c="red", marker="x")

    levels = np.linspace(np.amin(z), np.amax(z), 50)
    plt.contour(x_mesh, y_mesh, z, levels=levels)
    plt.title('new Branin g<=5 function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()



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


    # for testing  purpose
    x_test = np.array([[-5., -4.2, -3.5, -3., -2, -1.5, -0.5, 1, 1.5, 2.5, 3, 3.5, 4.5, 5, 6.1, 7, 7.5, 8.5, 9, 10],
                       [7.5, 2.2,  11.5,  15, 5.5,  9,   1.8, 4, 8.5, 0.5, 14.5,11,  4, 7, 0, 13.5, 9.5, 3,  6, 12]]).T
    x_test, z_test = Branin_after_init(x_test)

    # keep the mean and std of training data
    mean_train_x, mean_train_y, std_train_x, std_train_y, norm_train_x, norm_train_y = \
        train_data_norm(x_test, z_test)

    # fit GPR
    # kernal initialization should also use external configuration
    kernel = RBF(1, (np.exp(-1), np.exp(3)))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0)
    gpr.fit(norm_train_x, norm_train_y)

    '''
    '''
    levels = np.linspace(np.amin(z), np.amax(z), 50)
    plt.contour(x_mesh, y_mesh, z, levels=levels)
    plt.title('Branin function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


   
    gpr = load('Branin.joblib')

    x_sample = gpr.X_train_
    para_m = load('normal_p.joblib')

    x_mean = para_m['mean_x']
    x_std = para_m['std_x']

    z_mean = para_m['mean_y']
    z_std = para_m['std_y']
    
    '''

    '''
    x_mean = mean_train_x
    x_std = std_train_x

    z_mean = mean_train_y
    z_std = std_train_y

    x_sample = norm_train_x

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
    '''

    '''
    Z = Z * z_std + z_mean
    x_sample = x_sample * x_std + x_mean
    levels = np.linspace(np.amin(Z), np.amax(Z), 150)
    fig = plt.figure()
    ax1 = plt.Axes(fig, [0.2, 0.2, 0.4, 0.4])
    # ** there needs some fix  on colour bar
    plt.contour(X, Y, Z, levels=levels)
    plt.scatter(x_sample[:, 0], x_sample[:, 1], c="red", marker="x")
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.title('Branin estimate')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    '''
