import matplotlib.pyplot as plt

from knapsack.analyze.utility import (
    ExperimentConfig,
    append_experiment_results,
    init_alg,
    plot_diversity,
    plot_execution_times,
    plot_optimal_generations,
    plot_performance,
    print_statistical_summary,
)
from knapsack.genetic_algorithm import GeneticAlgorithm


def population_impact_analysis(
    alg: type[GeneticAlgorithm], config: ExperimentConfig, iterations=10
):
    algorithm = init_alg(alg, config)
    results = _measure_metrics(algorithm, config, iterations)

    plot_performance(results)
    plot_diversity(results)
    plot_execution_times(results, "Population Size")
    plot_optimal_generations(results, "Population Size")
    print_statistical_summary(results)

    return results


def _measure_metrics(alg: GeneticAlgorithm, config: ExperimentConfig, iterations):
    results = {}
    for size in config.population_sizes:
        key = f"Population Size: {size}"
        for _ in range(iterations):
            alg.clear_metrics()
            alg.population_size = size
            alg.reinitialize_population()
            execution_time = alg.evolve()
            append_experiment_results(results, key, alg, execution_time)

    return results
