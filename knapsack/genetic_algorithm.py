import time

import numpy as np

from knapsack.dataset import Dataset
from knapsack.evaluators.evaluator import Evaluator
from knapsack.mutations.mutation import Mutation
from knapsack.operators.crossover import Crossover
from knapsack.population import Population
from knapsack.selectors.selector import Selector


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = (self.end - self.start) * 1000


class GeneticAlgorithm:
    def __init__(
        self,
        dataset,
        evaluator: Evaluator,
        selector: Selector,
        crossover_operator: Crossover,
        mutation_operator: Mutation,
        population_size=20,
        num_generations=10,
        mutation_rate=0.01,
        strategy="value_biased",
    ):
        self.dev = False
        self.dataset = dataset
        self.gene_length = self.dataset.length

        self._population_size = population_size
        self._max_generations = num_generations
        self._mutation_rate = mutation_rate

        self._evaluator = evaluator
        print(evaluator)

        self._mutation_operator = mutation_operator
        if not self._mutation_operator.probability:
            self._mutation_operator.probability = mutation_rate
        print(mutation_operator)

        self._selector = selector
        print(selector)

        self._crossover_operator = crossover_operator
        print(crossover_operator)

        # create and initialize the first population
        self._strategy = strategy
        self.population = Population(
            self.dataset, self.selector, self.population_size, self.gene_length
        )
        self.population.initialize_with_strategy(self.strategy)

    # -------------------------------------------------

    @property
    def generations(self):
        return self._max_generations

    @generations.setter
    def generations(self, generations):
        self._max_generations = generations

    # -------------------------------------------------
    @property
    def evaluator(self):
        return self._evaluator

    @evaluator.setter
    def evaluator(self, evaluator):
        print(evaluator)
        self._evaluator = evaluator
        self.selector.evaluator = evaluator

    # -------------------------------------------------

    @property
    def selector(self):
        return self._selector

    @selector.setter
    def selector(self, selector):
        print(selector)
        self._selector = selector
        self._selector.evaluator = self.evaluator

    # -------------------------------------------------

    @property
    def mutation_operator(self):
        return self._mutation_operator

    @mutation_operator.setter
    def mutation_operator(self, mutation_operator):
        print(mutation_operator)
        self._mutation_operator = mutation_operator

        if not self._mutation_operator.probability:
            self._mutation_operator.probability = self.mutation_rate

    # -------------------------------------------------

    @property
    def mutation_rate(self):
        return self._mutation_rate

    @mutation_rate.setter
    def mutation_rate(self, mutation_rate):
        self._mutation_rate = mutation_rate
        self.mutation_operator.probability = mutation_rate

    # -------------------------------------------------
    @property
    def crossover_operator(self):
        return self._crossover_operator

    @crossover_operator.setter
    def crossover_operator(self, crossover_operator):
        print(crossover_operator)
        self._crossover_operator = crossover_operator

    # -------------------------------------------------

    @property
    def population_size(self):
        return self._population_size

    @population_size.setter
    def population_size(self, population_size):
        self._population_size = population_size

    # -------------------------------------------------

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy

    # -------------------------------------------------

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset
        self.gene_length = self.dataset.length

    # -------------------------------------------------

    def reinitialize_population(self):
        self.population = Population(
            self.dataset, self.selector, self.population_size, self.gene_length
        )
        self.population.initialize_with_strategy(self.strategy)

    def new_generation(self, current_generation):
        # new generation
        new_population = Population(
            self.dataset, self.selector, self.population_size, self.gene_length
        )

        while len(new_population.chromosomes) < self.population_size:
            parents = self.population.select_parents()
            # chose two parents from the selected chromosomes
            parent1, parent2 = parents

            # create a child by crossover and mutation
            children = self.crossover_operator.crossover(parent1, parent2)

            # mutate the children
            if hasattr(self.mutation_operator, "max_generations"):
                children = self.mutation_operator.mutate(children, current_generation)
            else:
                children = self.mutation_operator.mutate(children)

            # add the child to the new population
            new_population.add_chromosome(children)

        return new_population

    def evolve(self):
        """Evolve the population for a set number of generations."""
        self.best_fitness = []
        self.average_fitness = []
        self.worst_fitness = []
        self.diversity = []

        with Timer() as timer:
            for generation in range(self.generations):
                best_solution = self.get_best_solution()
                best_fitness = self.get_solution_fitness(best_solution)
                avg_fitness = np.mean(
                    [
                        self.get_solution_fitness(chrom)
                        for chrom in self.population.chromosomes
                    ]
                )
                worst_fitness = np.min(
                    [
                        self.get_solution_fitness(chrom)
                        for chrom in self.population.chromosomes
                    ]
                )

                diversity = self.population.measure_diversity()

                if not self.best_fitness:
                    self.optimal_generation = generation
                else:
                    if best_fitness > self.best_fitness[-1]:
                        self.optimal_generation = generation

                self.best_fitness.append(best_fitness)
                self.average_fitness.append(avg_fitness)
                self.worst_fitness.append(worst_fitness)
                self.diversity.append(diversity)

                if self.dev:
                    print(
                        f"Generation {generation}: Best Fitness: {best_fitness:.2f}, "
                        f"Average Fitness: {avg_fitness:.2f}, Diversity: {diversity:.2f}%"
                    )

                self.population = self.new_generation(generation)
                self.population.update_selector()

        if self.dev:
            print("=" * 35)
            print(f"Evolution took {timer.interval:.4f} miliseconds.")
        return timer.interval

    def clear_metrics(self):
        self.best_fitness = []
        self.average_fitness = []
        self.worst_fitness = []
        self.diversity = []
        self.optimal_generation = None

    def get_solution_fitness(self, solution):
        """Get the fitness of the solution.
        Args:
            solution (np.ndarray): Solution to evaluate.
        Returns:
            float: Fitness of the solution.
        """
        return self.evaluator.evaluate(solution)

    def get_population_statistics(self):
        """Gather fitness statistics for the current population."""
        fitness_scores = [
            self.evaluator.evaluate(chrom) for chrom in self.population.chromosomes
        ]
        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        worst_fitness = min(fitness_scores)

        return {
            "best_fitness": best_fitness,
            "average_fitness": avg_fitness,
            "worst_fitness": worst_fitness,
        }

    def prompt(self, evaluated):
        """Prompt the evaluated chromosome.

        Args:
            evaluated (np.ndarray): Evaluated chromosome.
        """
        total_weight = np.sum(evaluated * self.dataset.weights)
        total_value = np.sum(evaluated * self.dataset.values)
        evaluation = self.evaluator.evaluate(evaluated)

        capacity = float(self.dataset.capacity)

        total_prompt = "\n".join(
            [
                "=" * 30,
                f"genes={evaluated}",
                "=" * 30,
                f"total weight: {total_weight} | capacity: {capacity}",
                f"total value: {total_value} | evaluation: {evaluation}",
            ]
        )
        print(total_prompt)

    def get_best_solution(self, n=1):
        """Get the best solutions from the population.

        Args:
            n (int): Display top n solutions from the population.

        Returns:
            np.ndarray: Best solution from the population.
        """
        ordered = sorted(
            self.population.chromosomes,
            key=lambda chrom: self.evaluator.evaluate(chrom),
            reverse=True,
        )[:n]

        if self.dev:
            for evaluated in ordered:
                self.prompt(evaluated)

        return ordered[0]
