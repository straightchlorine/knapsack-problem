from typing import Sequence
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from knapsack.dataset import Dataset
from knapsack.evaluators.evaluator import Evaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.mutations.mutation import Mutation
from knapsack.operators.crossover import Crossover
from knapsack.selectors.selector import Selector


def compare_mutation_impact(
    problems: list[Dataset],
    algorithm: type[GeneticAlgorithm],
    evaluator: Evaluator,
    selector: Selector,
    crossover_operator: Crossover,
    mutation_operators: Sequence[Mutation],
    population_size: int,
    strategy: str = "value_biased",
):
    result = _measure_comparison_metrics(
        problems,
        algorithm,
        evaluator,
        selector,
        crossover_operator,
        mutation_operators,
        population_size,
        strategy,
    )
    plot_mutation_comparison(result)


def _measure_comparison_metrics(
    problems: list[Dataset],
    algorithm: type[GeneticAlgorithm],
    evaluator: Evaluator,
    selector: Selector,
    crossover_operator: Crossover,
    mutation_operators: Sequence[Mutation],
    population_size: int,
    strategy,
):
    alg = algorithm(
        problems[0],
        evaluator,
        selector,
        crossover_operator,
        mutation_operators[0],
        population_size,
        strategy=strategy,
    )
    alg.dev = False

    columns = [
        "mutation_operator",
        "problem_id",
        "mean_best_fitness",
        "mean_average_fitness",
        "mean_worst_fitness",
        "mean_diversity",
        "execution_time",
        "optimal_generation",
    ]
    df = pd.DataFrame(columns=pd.Index(columns))

    for operator in mutation_operators:
        alg.mutation_operator = operator
        for pid, problem in enumerate(problems):
            alg.set_problem(problem, strategy)
            execution_time = alg.evolve()

            # calculate metrics
            mean_best_fitness = np.mean(alg.best_fitness)
            mean_average_fitness = np.mean(alg.average_fitness)
            mean_worst_fitness = np.mean(alg.worst_fitness)
            mean_diversity = np.mean(alg.diversity)
            optimal_generation = alg.optimal_generation

            new_data = pd.DataFrame(
                [
                    {
                        "mutation_operator": operator.__class__.__name__,
                        "problem_id": pid,
                        "mean_best_fitness": mean_best_fitness,
                        "mean_average_fitness": mean_average_fitness,
                        "mean_worst_fitness": mean_worst_fitness,
                        "mean_diversity": mean_diversity,
                        "execution_time": execution_time,
                        "optimal_generation": optimal_generation,
                    }
                ]
            )
            new_data = new_data.dropna(axis=1, how="all")
            df = pd.concat([df, new_data], ignore_index=True, sort=False)

    return df


def plot_mutation_comparison(results):
    grouped_df = (
        results.groupby("mutation_operator")
        .agg(
            {
                "mean_best_fitness": "mean",
                "mean_average_fitness": "mean",
                "mean_worst_fitness": "mean",
                "mean_diversity": "mean",
                "execution_time": "mean",
                "optimal_generation": "mean",
            }
        )
        .reset_index()
    )

    # -------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.barplot(x="mutation_operator", y="mean_diversity", data=grouped_df)
    plt.title("Mean Diversity by Crossover Operator")
    plt.ylabel("Mean Diversity")
    plt.xlabel("Crossover Operator")
    plt.show()
    # -------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.barplot(x="mutation_operator", y="optimal_generation", data=grouped_df)
    plt.title("Optimal Generation by Crossover Operator")
    plt.ylabel("Optimal Generation")
    plt.xlabel("Crossover Operator")
    plt.show()
    # -------------------------------------------
