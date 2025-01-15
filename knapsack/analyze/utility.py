from dataclasses import dataclass
from typing import Any, Sequence

from knapsack.dataset import Dataset
from knapsack.evaluators.evaluator import Evaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.mutations.mutation import Mutation
from knapsack.operators.crossover import Crossover
from knapsack.selectors.selector import Selector


@dataclass
class ExperimentConfig:
    """Configuration parameters for genetic algorithm experiments."""

    problems: list[Dataset]
    evaluators: Sequence[Evaluator]
    selectors: Sequence[Selector]
    crossover_operators: Sequence[Crossover]
    mutation_operators: Sequence[Mutation]
    population_sizes: list[int]
    generations: list[int]
    mutation_rates: list[float]
    strategies: list[str]


@dataclass
class ExperimentResults:
    """Results from a single genetic algorithm run."""

    metadata: dict[str, Any]
    execution_time: float
    diversity: list[float]
    best_fitness: list[float]
    average_fitness: list[float]
    worst_fitness: list[float]


def init_alg(alg: type[GeneticAlgorithm], config: ExperimentConfig):
    algorithm = alg(
        config.problems[0],
        config.evaluators[0],
        config.selectors[0],
        config.crossover_operators[0],
        config.mutation_operators[0],
        config.population_sizes[0],
        config.generations[0],
        strategy=config.strategies[0],
    )
    algorithm.dev = False
    return algorithm


def is_class_equal(obj1, obj2):
    """Check if a property is equal between two objects."""
    return obj1.__class__.__name__ == obj2.__class__.__name__
