from dataclasses import dataclass
from statistics import fmean
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
    execution_time: list[float]
    diversity: list[list[float]]
    best_fitness: list[list[float]]
    average_fitness: list[list[float]]
    worst_fitness: list[list[float]]
    optimal_generation: list[int]

    @property
    def mean_execution_time(self) -> float:
        return fmean(self.execution_time) if self.execution_time else 0.0

    @property
    def mean_optimal_generation(self) -> float:
        return fmean(self.optimal_generation) if self.optimal_generation else 0.0

    @property
    def mean_diversity(self) -> list[float]:
        means = [fmean(values) for values in zip(*self.diversity)]
        return means

    @property
    def mean_best_fitness(self) -> list[float]:
        means = [fmean(values) for values in zip(*self.best_fitness)]
        return means

    @property
    def mean_average_fitness(self) -> list[float]:
        means = [fmean(values) for values in zip(*self.average_fitness)]
        return means

    @property
    def mean_worst_fitness(self) -> list[float]:
        means = [fmean(values) for values in zip(*self.worst_fitness)]
        return means


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


def append_experiment_results(results: dict, key, alg: GeneticAlgorithm, execution_time):
    if key not in results:
        results[key] = ExperimentResults(
            metadata={
                "population_size": alg.population_size,
                "mutation_rate": alg.mutation_rate,
                "selector": type(alg.selector).__name__,
                "operator": type(alg.crossover_operator).__name__,
                "evaluator": type(alg.evaluator).__name__,
                "generations": alg.generations,
            },
            execution_time=[execution_time],
            diversity=[],
            best_fitness=[],
            average_fitness=[],
            worst_fitness=[],
            optimal_generation=[],
        )

        results[key].diversity.append(alg.diversity)
        results[key].best_fitness.append(alg.best_fitness)
        results[key].average_fitness.append(alg.average_fitness)
        results[key].worst_fitness.append(alg.worst_fitness)
        results[key].optimal_generation.append(alg.optimal_generation)

    else:
        results[key].execution_time.append(execution_time)
        results[key].diversity.append(alg.diversity)
        results[key].best_fitness.append(alg.best_fitness)
        results[key].average_fitness.append(alg.average_fitness)
        results[key].worst_fitness.append(alg.worst_fitness)
        results[key].optimal_generation.append(alg.optimal_generation)


def print_statistical_summary(results: dict[str, ExperimentResults]):
    """
    Print statistical summary of the results.

    Args:
        results (dict): Dictionary of ExperimentResults objects.
    """
    print("\nStatistical Summary:")
    for operator, data in results.items():
        print(f"\n{operator}:")
        print(f"Execution Time: {data.mean_execution_time:.2f} miliseconds")
        print(f"Best Fitness: {fmean(data.mean_best_fitness):.4f}")
        print(f"Average Fitness: {fmean(data.mean_average_fitness):.4f}")
        print(f"Worst Fitness: {fmean(data.mean_worst_fitness):.4f}")
        print(f"Population Diversity: {fmean(data.mean_diversity):.4f}")
        print(f"Optimal Generation: {data.mean_optimal_generation}")
