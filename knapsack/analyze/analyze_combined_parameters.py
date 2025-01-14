from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def combined_analysis(
    alg,
    problem,
    population_sizes,
    mutation_rates,
    selectors,
    generation,
    crossover,
    evaluator,
):
    results = analyze_combined_parameters(
        alg,
        problem,
        population_sizes,
        mutation_rates,
        selectors,
        generation,
        crossover,
        evaluator,
    )
    plot_combined_analysis(results)


def plot_combined_analysis(results):
    data = []
    for config, metrics in results.items():
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
                }
            )

    # Convert data into a Pandas DataFrame
    df = pd.DataFrame(data)

    # Best Fitness Over Generations
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=df,
        x="Generation",
        y="Best Fitness",
        hue="Configuration",
        marker="o",
    )
    plt.title("Best Fitness Over Generations")
    plt.grid()
    plt.legend(
        title="Configuration",
        loc="upper left",
        fontsize="small",
        bbox_to_anchor=(1.05, 1),
    )
    plt.tight_layout()
    plt.show()

    # Average Fitness Over Generations
    plt.figure()
    sns.lineplot(
        data=df,
        x="Generation",
        y="Average Fitness",
        hue="Configuration",
        marker="o",
    )
    plt.title("Average Fitness Over Generations")
    plt.grid()
    plt.legend(
        title="Configuration",
        loc="upper left",
        fontsize="small",
        bbox_to_anchor=(1.05, 1),
    )
    plt.tight_layout()
    plt.show()

    # Worst Fitness Over Generations
    plt.figure()
    sns.lineplot(
        data=df,
        x="Generation",
        y="Worst Fitness",
        hue="Configuration",
        marker="o",
    )
    plt.title("Worst Fitness Over Generations")
    plt.grid()
    plt.legend(
        title="Configuration",
        loc="upper left",
        fontsize="small",
        bbox_to_anchor=(1.05, 1),
    )
    plt.tight_layout()
    plt.show()

    # Diversity Over Generations
    plt.figure()
    sns.lineplot(
        data=df,
        x="Generation",
        y="Diversity",
        hue="Configuration",
        marker="o",
    )
    plt.title("Diversity Over Generations")
    plt.grid()
    plt.legend(
        title="Configuration",
        loc="upper left",
        fontsize="small",
        bbox_to_anchor=(1.05, 1),
    )
    plt.tight_layout()
    plt.show()


def analyze_combined_parameters(
    alg_class,
    problem,
    population_sizes,
    mutation_rates,
    selectors,
    generation,
    crossover,
    evaluator,
):
    results = {}

    for pop_size, mut_rate, selector in product(
        population_sizes,
        mutation_rates,
        selectors,
    ):
        if hasattr(type(selector), "evaluator"):
            selector.evaluator = evaluator

        alg = alg_class(
            problem,
            evaluator,
            selector,
            crossover,
            population_size=pop_size,
            num_generations=generation,
            mutation_rate=mut_rate,
        )

        execution_time = alg.evolve()

        key = f"{pop_size}-{mut_rate}-{type(selector).__name__}"
        results[key] = {
            "execution_time": execution_time,
            "diversity": alg.diversity,
            "best_fitness": alg.best_fitness,
            "average_fitness": alg.average_fitness,
            "worst_fitness": alg.worst_fitness,
        }

    return results
