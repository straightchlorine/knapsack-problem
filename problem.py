#!/usr/bin/env python

from knapsack import population
from knapsack.analyze.analyze_combined_parameters import (
    analyze_combined_parameters,
    combined_analysis,
)
from knapsack.analyze.analyze_crossover import (
    analyze_crossover_methods,
    crossover_operator_analysis,
    plot_crossover_analysis,
)
from knapsack.analyze.analyze_mutation import (
    _measure_metrics,
    mutation_metric_impact_analysis,
    plot_mutation_impact,
)
from knapsack.analyze.analyze_population_size import (
    analyze_population_size,
    plot_population_analysis,
    population_analysis,
)
from knapsack.analyze.analyze_selectors import (
    analyze_selection_methods,
    plot_selection_analysis,
    selection_analysis,
)
from knapsack.analyze.analyze_tournament_to_generations import (
    analyze_tournament_selection,
    plot_tournament_selector_analysis,
    tournament_selector_analysis,
)
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
from knapsack.selectors.roulette_selector import RouletteSelector
from knapsack.selectors.tournament_selector import TournamentSelector

# testing imports
from tests.test_execution_time import (
    measure_execution_time,
    plot_execution_speed,
)
from tests.test_mutation_impact import (
    plot_mutation_diversity,
    test_mutation_impact,
)
from tests.test_mutation_impact_by_selector import (
    plot_mutation_diversity_by_selector,
    test_mutation_diversity_by_selector,
)
from tests.test_selection_methods import (
    analyze_results,
    plot_test_selection,
    test_selection_methods,
)

dataset = DataInterface.from_csv("datasets/dataset.csv")

# random problem/specific problem from the dataset
problem = dataset.random_problem()
problem = dataset.chromosome_datasets[100]

# FitnessEvaluator:
fitness_evaluator = FitnessEvaluator(problem)
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
random_selector = RandomSelector()
# Using RandomSelector is quite hard to get best solution out of the algorithm.
# Its doing decent, when it comes to finding solutions when multiple weights
# are involved, but fails miserably, when it comes to datasets, that contain
# only 1 weight (ex. dataset.chromosome_datasets[0]).
# In such case it requires at least population size of 60 and 10 generations
# or
# for population size 10, it requres 10^6 generations to get to the best
# solution, which is not something usable in any enviornment.

# ElitismSelector:
elitism_selector = ElitismSelector(evaluator)
# (ElitismSelector + FitnessEvaluator)
# Using ElitismSelector, it is possible to get to the best solution within
# 10 generations with population size of 10. Significant difference, compared
# to the random. For this case with only 5 weight:value pairs it works well,
# but for wider cases it might constrict the algorithm too much. )

roulette_selector = RouletteSelector(evaluator)
tournament_selector = TournamentSelector(evaluator)

# display debug information
dev = False

# create genetic algorithm object
alg = GeneticAlgorithm(
    problem,
    evaluator,
    elitism_selector,
    # FixedPointCrossover(dev, fixed_point=3),
    MultiPointCrossover(points=[2, 3], dev=dev),
    # UniformCrossover(dev),
    population_size=10,
    num_generations=5,
    mutation_rate=0.01,
)

# start the algorithm
# alg.evolve()

# get the best solutions
# alg.get_best_solution(5)

selectors = [
    random_selector,
    elitism_selector,
    roulette_selector,
    tournament_selector,
]


# List 3
# -------------------------------------------------------------
population_sizes = [2, 6, 10, 50, 200]
mutation_rate = 0.01
generations = 5
evaluator = ScalingFitnessEvaluator(problem)
selector = RouletteSelector(evaluator)
crossover_operator = MultiPointCrossover(points=[2, 3], dev=dev)

list3 = False
if list3:
# -------------------------------------------------------------
