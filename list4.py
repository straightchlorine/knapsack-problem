#!/usr/bin/env python
from knapsack.analyze.parameters_impact import parameters_impact_analysis
from knapsack.analyze.crossover_operator_impact import crossover_efectiveness
from knapsack.analyze.mutation_impact import mutation_rate_impact
from knapsack.analyze.population_size_impact import population_size_impact
from knapsack.analyze.selector_impact import selector_effectiveness
from knapsack.analyze.tournament_selector_impact import tournament_selector_analysis
from knapsack.analyze.utility import ExperimentConfig
from knapsack.dataset import DataInterface
from knapsack.evaluators.fitness import FitnessEvaluator, ScalingFitnessEvaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.mutations.bitflip_mutation import BitFlipMutation
from knapsack.mutations.dynamic_mutation import DynamicMutation
from knapsack.mutations.gaussian_mutation import GaussianMutation
from knapsack.operators.arithmetic_crossover import ArithmeticCrossover
from knapsack.operators.blend_crossover import BlendCrossover
from knapsack.operators.fixed_point_crossover import FixedPointCrossover
from knapsack.operators.multi_point_crossover import MultiPointCrossover
from knapsack.operators.simulated_binary_crossover import SimulatedBinaryCrossover
from knapsack.operators.uniform_crossover import UniformCrossover
from knapsack.selectors.elitism_selector import ElitismSelector
from knapsack.selectors.random_selector import RandomSelector
from knapsack.selectors.roulette_selector import RouletteSelector
from knapsack.selectors.tournament_selector import TournamentSelector

dev = False

dataset = DataInterface.from_csv("datasets/dataset.csv")
problem = dataset.chromosome_datasets[100]

# ---- basic ----
population_sizes = [5, 6, 10, 50]
generations = [5, 10, 20]
mutation_rates = [0.01, 0.05, 0.1, 0.2]
# ---- basic ----

evaluators = [
    FitnessEvaluator(problem),
    ScalingFitnessEvaluator(problem),
]
eval = evaluators[0]
selectors = [
    RandomSelector(),
    RouletteSelector(eval),
    TournamentSelector(eval),
    ElitismSelector(eval),
]
crossovers = [
    MultiPointCrossover(points=[2, 3], dev=dev),
    FixedPointCrossover(fixed_point=2, dev=dev),
    UniformCrossover(dev=dev),
    ArithmeticCrossover(dev=dev),
    BlendCrossover(dev=dev),
    SimulatedBinaryCrossover(dev=dev),
]
mutations = [
    BitFlipMutation(mutation_rates[0]),
    GaussianMutation(mutation_rates[0]),
    DynamicMutation(mutation_rates[0], 5),
]

config = ExperimentConfig(
    dataset.chromosome_datasets[:10],
    evaluators,
    selectors,
    crossovers,
    mutations,
    population_sizes,
    generations,
    mutation_rates,
    ["normal"],
)

config.generations = [5]
# -------------------------------------------------------------
population_size_impact(GeneticAlgorithm, config)
# -------------------------------------------------------------
selector_effectiveness(GeneticAlgorithm, config)
# -------------------------------------------------------------
crossover_efectiveness(GeneticAlgorithm, config)
# -------------------------------------------------------------
mutation_rate_impact(GeneticAlgorithm, config)
# -------------------------------------------------------------
tournament_selector_analysis(
    GeneticAlgorithm, config, TournamentSelector(eval), [1, 2, 3]
)
# -------------------------------------------------------------
parameters_impact_analysis(GeneticAlgorithm, config)
# -------------------------------------------------------------
