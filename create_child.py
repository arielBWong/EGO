import numpy as np


def create_child(dimensions, bounds, popsize, crossp, mut, pop):
    min_b, max_b = np.asarray(bounds).T
    child_x = np.zeros((popsize, dimensions))
    for j in range(popsize):
        idxs = [idx for idx in range(popsize) if idx != j]
        a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
        mutant = np.clip(a + mut * (b - c), 0, 1)
        cross_points = np.random.rand(dimensions) < crossp
        if not np.any(cross_points):
            cross_points[np.random.randint(0, dimensions)] = True
        trial = np.where(cross_points, mutant, pop[j])
        child_x[j, :] = trial
    return (child_x)