import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Sequence
from knapsack.dataset import Dataset
from knapsack.evaluators.evaluator import Evaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.mutations.mutation import Mutation
from knapsack.operators.crossover import Crossover
from knapsack.selectors.selector import Selector


def crossover_efectiveness(
    problems: list[Dataset],
    algorithm: type[GeneticAlgorithm],
    evaluator: Evaluator,
    selector: Selector,
    crossover_operators: Sequence[Crossover],
    mutation_operator: Mutation,
    population_size: int,
    strategy: str = "value_biased",
):
    results = _measure_crossover_effectiveness(
        problems,
        algorithm,
        evaluator,
        selector,
        crossover_operators,
        mutation_operator,
        population_size,
        strategy,
    )
    plot_crossover_effectiveness(results)


def _measure_crossover_effectiveness(
    problems: list[Dataset],
    algorithm: type[GeneticAlgorithm],
    evaluator: Evaluator,
    selector: Selector,
    crossover_operators: Sequence[Crossover],
    mutation_operator: Mutation,
    population_size: int,
    strategy: str = "value_biased",
):
    alg = algorithm(
        problems[0],
        evaluator,
        selector,
        crossover_operators[0],
        mutation_operator,
        population_size,
    )
    alg.dev = False

    columns = [
        "crossover_operator",
        "problem_id",
        "mean_best_fitness",
        "mean_average_fitness",
        "mean_worst_fitness",
        "mean_diversity",
        "execution_time",
        "optimal_generation",
    ]
    df = pd.DataFrame(columns=pd.Index(columns))

    # iterate through operators, set them
    # iterate through problems, set them
    for operator in crossover_operators:
        alg.crossover_operator = operator

        for pid, problem in enumerate(problems):
            alg.set_problem(problem, strategy)
            alg.clear_metrics()
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
                        "crossover_operator": operator.__class__.__name__,
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


def plot_crossover_effectiveness(df: pd.DataFrame):
    grouped_df = (
        df.groupby("crossover_operator")
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
    sns.barplot(x="crossover_operator", y="mean_best_fitness", data=grouped_df)
    plt.title("Mean Best Fitness by Crossover Operator")
    plt.ylabel("Mean Best Fitness")
    plt.xlabel("Crossover Operator")
    plt.show()
    # -------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.barplot(x="crossover_operator", y="mean_average_fitness", data=grouped_df)
    plt.title("Mean Average Fitness by Crossover Operator")
    plt.ylabel("Mean Average Fitness")
    plt.xlabel("Crossover Operator")
    plt.show()
    # -------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.barplot(x="crossover_operator", y="mean_worst_fitness", data=grouped_df)
    plt.title("Mean Worst Fitness by Crossover Operator")
    plt.ylabel("Mean Worst Fitness")
    plt.xlabel("Crossover Operator")
    plt.show()
    # -------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.barplot(x="crossover_operator", y="mean_diversity", data=grouped_df)
    plt.title("Mean Diversity by Crossover Operator")
    plt.ylabel("Mean Diversity")
    plt.xlabel("Crossover Operator")
    plt.show()
    # -------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.barplot(x="crossover_operator", y="execution_time", data=grouped_df)
    plt.title("Execution Time by Crossover Operator")
    plt.ylabel("Execution Time (s)")
    plt.xlabel("Crossover Operator")
    plt.show()
    # -------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.barplot(x="crossover_operator", y="optimal_generation", data=grouped_df)
    plt.title("Optimal Generation by Crossover Operator")
    plt.ylabel("Optimal Generation")
    plt.xlabel("Crossover Operator")
    plt.show()
    # -------------------------------------------
