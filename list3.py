#!/usr/bin/env python
from knapsack.analyze.mutation_impact import mutation_impact
from knapsack.dataset import DataInterface
from knapsack.evaluators.fitness import ScalingFitnessEvaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.operators.multi_point_crossover import MultiPointCrossover
from knapsack.performance.selector_execution_time import (
    selector_time_efficiency,
)
from knapsack.performance.selector_mutation_impact import (
    selector_diversity_impact,
)
from knapsack.performance.selector_performance import selector_effectiveness
from knapsack.selectors.elitism_selector import ElitismSelector
from knapsack.selectors.random_selector import RandomSelector
from knapsack.selectors.roulette_selector import RouletteSelector
from knapsack.selectors.tournament_selector import TournamentSelector

dev = False

dataset = DataInterface.from_csv("datasets/dataset.csv")
problem = dataset.chromosome_datasets[100]

# ---- basic ----
population_size = 10
num_generations = 5
mutation_rate = 0.01
# ---- basic ----

evaluator = ScalingFitnessEvaluator(problem)
selector = RouletteSelector(evaluator)
crossover = MultiPointCrossover(points=[2, 3], dev=dev)

mutation_rates = [0.01, 0.05, 0.1]
selectors = [
    RandomSelector(),
    RouletteSelector(evaluator),
    TournamentSelector(evaluator, tournament_size=3),
    ElitismSelector(evaluator),
]

alg = GeneticAlgorithm(
    problem,
    evaluator,
    selector,
    crossover,
    population_size=population_size,
    num_generations=num_generations,
    mutation_rate=mutation_rate,
)

selector_effectiveness(alg, selectors, iterations=20)
selector_time_efficiency(alg, selectors, iterations=20)
mutation_impact(alg, mutation_rates, iterations=20)
selector_diversity_impact(alg, selectors, mutation_rates, iterations=20)
