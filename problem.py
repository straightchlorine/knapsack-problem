from knapsack.dataset import DataInterface
from knapsack.evaluators.fitness import (
    FitnessEvaluator,
    ScalingFitnessEvaluator,
)
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.operators.fixed_point_crossover import FixedPointCrossover
from knapsack.operators.multi_point_crossover import MultiPointCrossover
from knapsack.operators.uniform_crossover import UniformCrossover
from knapsack.selectors.elitism_selector import ElitismSelector
from knapsack.selectors.random_selector import RandomSelector

# create DataInterface object based on csv file
dataset = DataInterface.from_csv("datasets/dataset.csv")

# get the random problem out of the dataset
problem = dataset.random_problem()

# pick problems from the available datasets
# problem = dataset.chromosome_datasets[0]

# FitnessEvaluator:
# evaluator = FitnessEvaluator(problem)
# Very basic evaluator, that evaluates chromosome to 0, if weight exceeds the
# capacity.

# ScalingFitnessEvaluator:
evaluator = ScalingFitnessEvaluator(problem)
# Similar to FitnessEvaluator, but the punishments scales, based on how much
# weight exceeds the capacity, multiplied by the amount of elements in the
# weights array. Thanks to that, bad chromosomes are receiving more severe
# punishment, which combined with Elitisim selector offers quicker convergence
# of the algorithm, at the cost of the diversity of the population.

# RandomSelector:
# selector = RandomSelector()
# Using RandomSelector is quite hard to get best solution out of the algorithm.
# Its doing decent, when it comes to finding solutions when multiple weights
# are involved, but fails miserably, when it comes to datasets, that contain
# only 1 weight (ex. dataset.chromosome_datasets[0]).
# In such case it requires at least population size of 60 and 10 generations
# or
# for population size 10, it requres 10^6 generations to get to the best
# solution, which is not something usable in any enviornment.

# ElitismSelector:
selector = ElitismSelector(evaluator)
# (ElitismSelector + FitnessEvaluator)
# Using ElitismSelector, it is possible to get to the best solution within
# 10 generations with population size of 10. Significant difference, compared
# to the random. For this case with only 5 weight:value pairs it works well,
# but for wider cases it might constrict the algorithm too much. )

# display debug information
dev = False

# operator = FixedPointCrossover(dev)

# amount of points defines how many points will be picked for crossover
# operator = MultiPointCrossover(points=2, dev)

operator = UniformCrossover(dev)


# create genetic algorithm object
alg = GeneticAlgorithm(
    problem,
    evaluator,
    selector,
    operator,
    population_size=10,
    num_generations=10,
    mutation_rate=0.01,
)

# start the algorithm
alg.evolve()

# get the best solutions
alg.get_best_solution(5)
