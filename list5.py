#!/usr/bin/env python3

from knapsack.analyze.compare_mutation_impact import compare_mutation_impact
from knapsack.analyze.selector_impact import selector_impact_analysis
from knapsack.dataset import DataInterface
from knapsack.evaluators.fitness import ScalingFitnessEvaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.mutations.bitflip_mutation import BitFlipMutation
from knapsack.mutations.dynamic_mutation import DynamicMutation
from knapsack.mutations.gaussian_mutation import GaussianMutation
from knapsack.operators.arithmetic_crossover import ArithmeticCrossover
from knapsack.operators.blend_crossover import BlendCrossover
from knapsack.operators.multi_point_crossover import MultiPointCrossover
from knapsack.operators.fixed_point_crossover import FixedPointCrossover
from knapsack.operators.simulated_binary_crossover import SimulatedBinaryCrossover
from knapsack.performance.crossover_performance import (
    _measure_crossover_effectiveness,
    crossover_efectiveness,
)
from knapsack.performance.initialization_performance import initialization_effectiveness
from knapsack.performance.selector_performance import selection_performance_metrics
from knapsack.selectors.adaptive_selector import AdaptiveRouletteSelector
from knapsack.selectors.ranking_selector import RankingSelector
from knapsack.selectors.roulette_selector import RouletteSelector
from knapsack.selectors.tournament_selector import TournamentSelector
from knapsack.selectors.variable_tournament_selector import VariableTournamentSelector

dev = False

dataset = DataInterface.from_csv("datasets/dataset.csv")
problem = dataset.chromosome_datasets[100]

# ---- basic ----
population_size = 10
num_generations = 5
mutation_rate = 0.01
# ---- basic ----

# List 5
# -------------------------------------------------------------
evaluator = ScalingFitnessEvaluator(problem)
selector = TournamentSelector(evaluator)
crossover = MultiPointCrossover(points=[2, 3], dev=dev)
mutation_operator = BitFlipMutation(mutation_rate)

strategies = [
    "value_biased",
    "value_weight_ratio",
    "weight_constrained",
    "greedy_randomized",
    "small_large_mix",
    "uniform",
    "normal",
]

alg = GeneticAlgorithm(
    problem,
    evaluator,
    selector,
    crossover,
    mutation_operator,
    population_size=population_size,
    num_generations=num_generations,
)
alg.dev = dev
initialization_effectiveness(alg, strategies)
# -------------------------------------------------------------
selectors = [
    RouletteSelector(evaluator),
    TournamentSelector(evaluator),
    AdaptiveRouletteSelector(evaluator),
    RankingSelector(evaluator),
    VariableTournamentSelector(evaluator),
]
selector_impact_analysis(alg, selectors)
# -------------------------------------------------------------
crossover_operators = [
    BlendCrossover(dev=dev),
    ArithmeticCrossover(dev=dev),
    SimulatedBinaryCrossover(dev=dev),
    MultiPointCrossover(points=[2, 3], dev=dev),
    FixedPointCrossover(fixed_point=3, dev=dev),
]
crossover_efectiveness(
    dataset.chromosome_datasets[:2],
    GeneticAlgorithm,
    evaluator,
    selector,
    crossover_operators,
    mutation_operator,
    population_size,
    "small_large_mix",
)
# -------------------------------------------------------------
crossover_operator = MultiPointCrossover(points=[2, 3], dev=dev)
mutation_operators = [
    BitFlipMutation(mutation_rate),
    DynamicMutation(mutation_rate, num_generations),
    GaussianMutation(mutation_rate),
]
compare_mutation_impact(
    dataset.chromosome_datasets[:2],
    GeneticAlgorithm,
    evaluator,
    selector,
    crossover_operator,
    mutation_operators,
    population_size,
    "small_large_mix",
)
