import operator
import math
import random
import time
import sys
import numpy as np

from deap.algorithms import varOr, eaMuPlusLambda
from deap import base
from deap import creator
from deap import tools
from deap import gp

from gp_fitness import coh_fitness
from regplots import plot_results
from mockdata import starting_data
from mockdata.mock_noise import my_func as f

# Define new functions
def safeDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 0

def safeExp(arg):
    try:
        return math.exp(arg)
    except OverflowError:
        return 0

# second argument: number of witness channels to use
pset = gp.PrimitiveSet("MAIN", 2)
# set of possible operations to use in expression tree
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(safeExp, 1)
pset.addPrimitive(np.arctan, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='z1')
pset.renameArguments(ARG1='z2')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


times, tar, y1, y2, data = starting_data(sys.argv[1:], include_filt=True, scat_model=True)
toolbox.register("evaluate", coh_fitness, times, tar, y1, y2, toolbox)

toolbox.register("select",
                 tools.selDoubleTournament,
                 fitness_size=3,
                 parsimony_size=1.3,
                 fitness_first=False)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# limit total tree height
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=8))



def main():
    # random.seed(time.time())
    random.seed(300)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(3)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # From the deap documentation:
    # eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen
    #               [, stats, halloffame, verbose])
    # mu: The number of individuals to select for the next generation.
    # lambda_: The number of children to produce at each generation.
    # cxpb: The probability that an offspring is produced by crossover.
    # mutpb: The probability that an offspring is produced by mutation.
    # ngen: The number of generations.

    pop, log = eaMuPlusLambda(pop, toolbox, 50, 50, 0.5, 0.5, 5,
                              stats=mstats, halloffame=hof, verbose=True)
    print hof[0]
    f_guess = toolbox.compile(expr=hof[0])
    f_guess = np.vectorize(f_guess)
    f_guess_vals = f_guess(y1,y2)
    args = sys.argv[1:]
    plot_results(tar, y1, y2, f_guess_vals, data,
                 file_end = "_prelim", plot_all=True)
    pop, log = eaMuPlusLambda(pop, toolbox, 30, 30, 0.3, 0.7, 30,
                              stats=mstats, halloffame=hof, verbose=True)

    # print log
    print hof[0]
    print hof[1]
    print hof[2]
    f_guess = toolbox.compile(expr=hof[0])
    f_guess = np.vectorize(f_guess)
    f_guess_vals = f_guess(y1,y2)
    plot_results(tar, y1, y2, f_guess_vals, data, file_end="_final")


    # plot with test data also: use same channels at a different time
    # or different randomly generated data
    if(len(args) >= 3):
        args[2] = int(args[2]) - SEC
    global times, tar, y1, y2, data
    times, tar, y1, y2, data = starting_data(args)
    f_vals = f(y1,y2)
    tar_orig = data + apply_filt(f_vals)
    wit_guess = f_guess(y1,y2)
    plot_results(tar, y1, y2, wit_guess, data, file_end="_test", plot_all=False)

    return pop, log, hof

if __name__ == "__main__":
    main()
