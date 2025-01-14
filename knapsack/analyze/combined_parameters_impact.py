from itertools import product
from typing import Optional, Sequence

from pandas.core.api import DataFrame
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from knapsack.dataset import Dataset
from knapsack.evaluators.evaluator import Evaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.mutations.mutation import Mutation
from knapsack.operators.crossover import Crossover
from knapsack.selectors.selector import Selector


def combined_params_impact(
    alg: type[GeneticAlgorithm],
    problem: Dataset,
    population_sizes: list[int],
    mutation_rates: list[float],
    mutation_operators: Sequence[Mutation],
    selectors: Sequence[Selector],
    generations: list[int],
    operators: Sequence[Crossover],
    evaluators: Sequence[Evaluator],
):
    results = _measure_metrics(
        alg,
        problem,
        population_sizes,
        mutation_rates,
        mutation_operators,
        selectors,
        generations,
        operators,
        evaluators,
    )
    plot_combined_metric_analysis(results, filter_selector="TournamentSelector")
    return results


def _measure_metrics(
    alg_class: type[GeneticAlgorithm],
    problem: Dataset,
    population_sizes: list[int],
    mutation_rates: list[float],
    mutation_operators: Sequence[Mutation],
    selectors: Sequence[Selector],
    generations: list[int],
    operators: Sequence[Crossover],
    evaluators: Sequence[Evaluator],
):
    results = {}

    # iterate over all combinations of parameters
    for (
        population_size,
        mutation_rate,
        selector,
        gens,
        crossover_operator,
        evaluator,
        mutation_operator,
    ) in product(
        population_sizes,
        mutation_rates,
        selectors,
        generations,
        operators,
        evaluators,
        mutation_operators,
    ):
        # set appropriate evaluator to the selector
        if hasattr(type(selector), "evaluator"):
            selector.evaluator = evaluator

        # set an appropriate probability to the mutation operator
        mutation_operator.probability = mutation_rate

        alg = alg_class(
            problem,
            evaluator,
            selector,
            crossover_operator,
            mutation_operator,
            population_size=population_size,
            num_generations=gens,
        )

        execution_time = alg.evolve()

        prefix = f"{population_size}_{mutation_rate}_{gens}-"
        selector_name = type(selector).__name__[:3]
        operator_name = type(crossover_operator).__name__[:3]
        evaluator_name = type(evaluator).__name__[:3]
        key = f"{prefix}sel={selector_name}_op={operator_name}_eval={evaluator_name}"

        metadata = {
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "selector": type(selector).__name__,
            "operator": type(crossover_operator).__name__,
            "evaluator": type(evaluator).__name__,
            "generations": gens,
        }

        results[key] = {
            "metadata": metadata,
            "execution_time": execution_time,
            "diversity": alg.diversity,
            "best_fitness": alg.best_fitness,
            "average_fitness": alg.average_fitness,
            "worst_fitness": alg.worst_fitness,
        }

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
