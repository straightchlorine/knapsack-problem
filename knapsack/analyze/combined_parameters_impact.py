from itertools import product
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.api import DataFrame
import seaborn as sns

from knapsack.analyze.utility import ExperimentConfig, ExperimentResults, init_alg
from knapsack.evaluators.evaluator import Evaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.operators.crossover import Crossover
from knapsack.selectors.selector import Selector


def combined_params_impact(
    alg: type[GeneticAlgorithm],
    config: ExperimentConfig,
):
    algorithm = init_alg(alg, config)
    results = _measure_metrics(algorithm, config)
    plot_combined_metric_analysis(results, filter_selector="TournamentSelector")
    return results


def _generate_config_key(
    population_size: int,
    mutation_rate: float,
    generations: int,
    selector: Selector,
    operator: Crossover,
    evaluator: Evaluator,
) -> str:
    """Generate a unique key for a parameter configuration."""
    prefix = f"{population_size}_{mutation_rate}_{generations}-"
    selector_name = type(selector).__name__[:3]
    operator_name = type(operator).__name__[:3]
    evaluator_name = type(evaluator).__name__[:3]
    return f"{prefix}sel={selector_name}_op={operator_name}_eval={evaluator_name}"


def _measure_metrics(
    alg: GeneticAlgorithm,
    config: ExperimentConfig,
):
    results = {}

    # iterate over all combinations of parameters
    for params in product(
        config.evaluators,
        config.selectors,
        config.crossover_operators,
        config.mutation_operators,
        config.mutation_rates,
        config.population_sizes,
        config.generations,
        config.strategies,
    ):
        (
            evaluator,
            selector,
            crossover_operator,
            mutation_operator,
            mutation_rate,
            population_size,
            gens,
            strategy,
        ) = params

        # set all the parameters for the run and reinitialize population
        alg.evaluator = evaluator
        alg.selector = selector
        alg.crossover_operator = crossover_operator
        alg.mutation_operator = mutation_operator
        alg.mutation_rate = mutation_rate
        alg.population_size = population_size
        alg.generations = gens
        alg.strategy = strategy
        alg.reinitialize_population()

        # execute the algorithm
        execution_time = alg.evolve()

        key = _generate_config_key(
            population_size,
            mutation_rate,
            gens,
            selector,
            crossover_operator,
            evaluator,
        )

        results[key] = ExperimentResults(
            metadata={
                "population_size": population_size,
                "mutation_rate": mutation_rate,
                "selector": type(selector).__name__,
                "operator": type(crossover_operator).__name__,
                "evaluator": type(evaluator).__name__,
                "generations": gens,
            },
            execution_time=execution_time,
            diversity=alg.diversity,
            best_fitness=alg.best_fitness,
            average_fitness=alg.average_fitness,
            worst_fitness=alg.worst_fitness,
        )

    return results


def plot_combined_metric_analysis(
    results: dict,
    filter_population_size: Optional[int] = None,
    filter_selector: Optional[str] = None,
    filter_operator: Optional[str] = None,
    filter_evaluator: Optional[str] = None,
):
    data = []
    for config, metrics in results.items():
        metadata = metrics["metadata"]
        for generation, (best, avg, worst, div) in enumerate(
            zip(
                metrics["best_fitness"],
                metrics["average_fitness"],
                metrics["worst_fitness"],
                metrics["diversity"],
            )
        ):
            data.append(
                {
                    "Configuration": config,
                    "Generation": generation,
                    "Best Fitness": best,
                    "Average Fitness": avg,
                    "Worst Fitness": worst,
                    "Diversity": div,
                    "Execution Time": metrics["execution_time"],
                    "Population Size": metadata["population_size"],
                    "Selector": metadata["selector"],
                    "Operator": metadata["operator"],
                    "Evaluator": metadata["evaluator"],
                }
            )

    df = pd.DataFrame(data)

    if filter_population_size is not None:
        df = df[df["Population Size"] == filter_population_size]
    if filter_selector is not None:
        df = df[df["Selector"] == filter_selector]
    if filter_operator is not None:
        df = df[df["Operator"] == filter_operator]
    if filter_evaluator is not None:
        df = df[df["Evaluator"] == filter_evaluator]

    metrics_to_plot = ["Best Fitness", "Average Fitness", "Worst Fitness", "Diversity"]
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 5))

        if type(df) is DataFrame:
            sns.lineplot(
                data=df,
                x="Generation",
                y=metric,
                hue="Configuration",
                marker="o",
            )
        else:
            raise ValueError("Unable to convert to dataframe")

        plt.title(f"{metric} Over Generations")
        plt.grid()
        plt.legend(
            title="Configuration",
            loc="upper left",
            fontsize="small",
            bbox_to_anchor=(1.05, 1),
        )
        plt.tight_layout()
        plt.show()
