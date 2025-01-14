#!/usr/bin/env python
from knapsack.analyze.combined_parameters_impact import combined_params_impact
from knapsack.analyze.crossover_operator_impact import (
    crossover_operator_impact_analysis,
)
from knapsack.analyze.mutation_impact import mutation_impact
from knapsack.analyze.population_size_impact import population_impact_analysis
from knapsack.analyze.selector_impact import selector_impact_analysis
from knapsack.analyze.tournament_selector import (
    tournament_selector_params_impact_analysis,
)
from knapsack.dataset import DataInterface
from knapsack.evaluators.fitness import ScalingFitnessEvaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.operators.fixed_point_crossover import FixedPointCrossover
from knapsack.operators.multi_point_crossover import MultiPointCrossover
from knapsack.operators.uniform_crossover import UniformCrossover
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

population_sizes = [6, 10, 50]
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
alg.dev = dev

# List 4
# -------------------------------------------------------------
population_impact_analysis(
    GeneticAlgorithm,
    problem,
    evaluator,
    selector,
    crossover,
    mutation_rate,
    num_generations,
    population_sizes,
)
# -------------------------------------------------------------
evaluator = ScalingFitnessEvaluator(problem)
selectors = [
    RandomSelector(),
    ElitismSelector(evaluator),
    RouletteSelector(evaluator),
    TournamentSelector(evaluator),
]
selector_impact_analysis(alg, selectors)
# -------------------------------------------------------------
crossover_operators = [
    FixedPointCrossover(fixed_point=3, dev=dev),
    MultiPointCrossover(points=[2, 3], dev=dev),
    UniformCrossover(dev=dev),
]
crossover_operator_impact_analysis(alg, crossover_operators)
# -------------------------------------------------------------
mutation_rates = [0.01, 0.05, 0.1, 0.2, 0.8]
mutation_impact(alg, mutation_rates, iterations=20)
# -------------------------------------------------------------
selector = TournamentSelector(evaluator)
tournament_selector_params_impact_analysis(alg, selector, [2, 5, 10], [2, 3, 4])
# -------------------------------------------------------------
population_sizes = [6, 10, 50]
mutation_rates = [0.01, 0.05, 0.1]
evaluators = [ScalingFitnessEvaluator(problem)]
selectors = [
    RandomSelector(),
    ElitismSelector(evaluators[0]),
    TournamentSelector(evaluators[0]),
]
generations = [5]
crossover_operators = [MultiPointCrossover(points=[2, 3], dev=dev)]
combined_params_impact(
    GeneticAlgorithm,
    problem,
    population_sizes,
    mutation_rates,
    selectors,
    generations,
    crossover_operators,
    evaluators,
)
# -------------------------------------------------------------
