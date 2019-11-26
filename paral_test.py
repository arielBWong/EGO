import numpy as np
from EI_problem import expected_improvement
from deap import base
from deap import creator
from deap import tools
from EA_optimizer_for_EI import function_m

def EI_valuation(individual, **kwargs):
    f = expected_improvement(X=kwargs['X'],
                             X_sample=kwargs['X_sample'],
                             Y_sample=kwargs['Y_sample'],
                             gpr=kwargs['gpr'])
    return -f


creator.create('EI_fitness_min', base.Fitness, weight=(-1.0,))
creator.create('Individual', np.array, fitness=creator.EI_fitness_min)

toolbox = base.Toolbox()
toolbox.register("attr_conti", np.random.uniform, 0, 1)
toolbox.register('individual',
                 tools.initRepeat,
                 creator.Individual,
                 toolbox.attr_conti,
                 1)
toolbox.register('population', tools.initRepeat, np.array, toolbox.individual)

train_x = np.atleast_2d([[0, 0.2, 0.6, 1], [1, 0.2, 0.4, 0.1]]).T
train_y = function_m(train_x)


#toolbox.register('evaluation', EI_valuation, **kwargs)




