from knapsack.analyze.utility import (
    ExperimentConfig,
    append_experiment_results,
    init_alg,
    plot_diversity,
    plot_execution_times,
    plot_optimal_generations,
    plot_performance,
)
from knapsack.genetic_algorithm import GeneticAlgorithm


def crossover_efectiveness(
    alg: type[GeneticAlgorithm],
    config: ExperimentConfig,
    iterations=10,
):
    algorithm = init_alg(alg, config)
    results = _measure_metrics(algorithm, config, iterations)

    plot_performance(results)
    plot_diversity(results)
    plot_execution_times(results, "Crossover Operator")
    plot_optimal_generations(results, "Crossover Operator")


def _measure_metrics(alg: GeneticAlgorithm, config: ExperimentConfig, iterations):
    results = {}
    for operator in config.crossover_operators:
        alg.crossover_operator = operator
        key = operator.__class__.__name__

        for _ in range(iterations):
            for _, problem in enumerate(config.problems):
                alg.clear_metrics()
                alg.dataset = problem
                alg.reinitialize_population()
                execution_time = alg.evolve()
                append_experiment_results(results, key, alg, execution_time)

    return results
