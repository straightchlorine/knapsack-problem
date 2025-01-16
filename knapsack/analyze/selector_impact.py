from knapsack.analyze.utility import (
    ExperimentConfig,
    append_experiment_results,
    init_alg,
    print_statistical_summary,
)
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.visualization.plots import (
    plot_diversity,
    plot_execution_times,
    plot_optimal_generations,
    plot_performance,
)


def selector_effectiveness(
    alg: type[GeneticAlgorithm],
    config: ExperimentConfig,
    iterations=10,
):
    """Test and visualise performance of various selection methods.

    Args:
        alg (type[GeneticAlgorithm]): Genetic algorithm class.
        config (ExperimentConfig): Experiment configuration.
        iterations (int, optional): Number of iterations. Defaults to 10.
    """
    algorithm = init_alg(alg, config)
    results = _measure_metrics(algorithm, config, iterations)

    label = "Selector Operator"
    print_statistical_summary(results)
    plot_performance(results)
    plot_diversity(results)
    plot_execution_times(results, label)
    plot_optimal_generations(results, label)

    return results


def _measure_metrics(alg: GeneticAlgorithm, config: ExperimentConfig, iterations=10):
    results = {}
    for selector in config.selectors:
        alg.selector = selector
        key = type(selector).__name__

        for _ in range(iterations):
            alg.clear_metrics()
            alg.reinitialize_population()
            execution_time = alg.evolve()
            append_experiment_results(results, key, alg, execution_time)

    return results
