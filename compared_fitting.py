from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import numpy as np
import matplotlib.pyplot as plt
from unitFromGPR import external_test_of_loglikelihood

np.random.seed(20)

def y(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
    # y = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    return y.tolist()


train_X = np.array([3, 1, 4, 5, 9]).reshape(-1, 1)
train_y = y(train_X, noise_sigma=0)
test_X = np.arange(0, 10, 0.1).reshape(-1, 1)


# fit GPR
# kernel = ConstantKernel(constant_value=1, constant_value_bounds=(1, 1)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
kernel = RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
print(gpr.get_params())

gpr.fit(train_X, train_y)

print(gpr.kernel_.theta)
print(gpr.kernel_.length_scale)

mu, cov = gpr.predict(test_X, return_cov=True)
test_y = mu.ravel()
uncertainty = 1.96 * np.sqrt(np.diag(cov))





# plotting
plt.figure()
# plt.title("l=%.1f sigma_f=%.1f" % (gpr.kernel_.k2.length_scale, gpr.kernel_.k1.constant_value))
plt.title("l=%.1f" % (gpr.kernel_.length_scale))
plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
plt.plot(test_X, test_y, label="predict")
plt.scatter(train_X, train_y, label="train", c="red", marker="x")
plt.legend()
plt.show()


external_test_of_loglikelihood(gpr)



