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
        dataset: Dataset,
        evaluator: Evaluator,
        selector: Selector,
        crossover_operator: Crossover,
        mutation_operator: Mutation,
        population_size=20,
        num_generations=10,
        mutation_rate=0.01,
        strategy="value_biased",
    ):
        self._evaluator = evaluator
        self._selector = selector
        self._selector.evaluator = evaluator

        self.dev = False
        self.dataset = dataset
        self.gene_length = dataset.length

        self._population_size = population_size
        self._max_generations = num_generations
        self._mutation_rate = mutation_rate

        self._mutation_operator = mutation_operator

        if hasattr(self._mutation_operator, "probability"):
            self._mutation_operator._probability = mutation_rate

        self._crossover_operator = crossover_operator
        self._strategy = strategy

        # Initialize the first population
        self.population = Population(
            self.dataset, self.selector, self.population_size, self.gene_length
        )
        self.population.initialize_population(strategy)

    @property
    def selector(self):
        return self._selector

    @selector.setter
    def selector(self, selector: Selector):
        self._selector = selector
        self._selector.evaluator = self.evaluator
        self.population.selector = selector
        self.population.update_selector()

    @property
    def crossover_operator(self):
        return self._crossover_operator

    @crossover_operator.setter
    def crossover_operator(self, operator: Crossover):
        self._crossover_operator = operator

    @property
    def population_size(self):
        return self._population_size

    @population_size.setter
    def population_size(self, size):
        self._population_size = size

    @property
    def evaluator(self):
        return self._evaluator

    @evaluator.setter
    def evaluator(self, evaluator: Evaluator):
        self._evaluator = evaluator
        self._evaluator.dataset = self._dataset

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: Dataset):
        self._dataset = dataset
        self._evaluator.dataset = dataset

    @property
    def generations(self):
        return self._max_generations

    @generations.setter
    def generations(self, value):
        self._max_generations = value

    @property
    def mutation_rate(self):
        return self._mutation_rate

    @mutation_rate.setter
    def mutation_rate(self, value):
        self._mutation_rate = value
        if self._mutation_operator and hasattr(self._mutation_operator, "probability"):
            self._mutation_operator.probability = value

    @property
    def mutation_operator(self):
        return self._mutation_operator

    @mutation_operator.setter
    def mutation_operator(self, operator: Mutation):
        self._mutation_operator = operator
        if not self._mutation_operator.probability:
            self._mutation_operator.probability = self._mutation_rate

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy

    def clear_metrics(self):
        self.best_fitness = []
        self.average_fitness = []
        self.worst_fitness = []
        self.diversity = []
        self.optimal_generation = 0

    def reinitialize_population(self):
        """Reinitialize the population with the current strategy."""
        self.population = Population(
            self.dataset, self._selector, self._population_size, self.gene_length
        )
        self.population.initialize_population(self._strategy)

    def evolve(self):
        """Evolve the population for a set number of generations."""
        self.best_fitness = []
        self.average_fitness = []
        self.worst_fitness = []
        self.diversity = []

        with Timer() as timer:
            for generation in range(self.generations):
                fitness_scores = [
                    self.evaluator.evaluate(chrom)
                    for chrom in self.population.chromosomes
                ]

                best_fitness = max(fitness_scores)
                avg_fitness = np.mean(fitness_scores)
                worst_fitness = min(fitness_scores)
                diversity = self.population.measure_diversity()

                if not self.best_fitness or best_fitness > max(self.best_fitness):
                    self.optimal_generation = generation

                self.best_fitness.append(best_fitness)
                self.average_fitness.append(avg_fitness)
                self.worst_fitness.append(worst_fitness)
                self.diversity.append(diversity)

                if self.dev:
                    print(
                        f"Generation {generation}: "
                        f"Best Fitness: {best_fitness:.2f}, "
                        f"Avg Fitness: {avg_fitness:.2f}, "
                        f"Diversity: {diversity:.2f}%"
                    )

                self.population = self.new_generation(generation)

        if self.dev:
            print(f"Evolution completed in {timer.interval:.2f} ms.")

        return timer.interval

    def new_generation(self, current_generation):
        """Create a new generation of chromosomes."""
        new_population = Population(
            self.dataset, self._selector, self._population_size, self.gene_length
        )

        while len(new_population.chromosomes) < self.population_size:
            parent1, parent2 = self.population.select_parents()
            children = self._crossover_operator.crossover(parent1, parent2)

            if hasattr(self._mutation_operator, "max_generations"):
                children = self._mutation_operator.mutate(children, current_generation)
            else:
                children = self._mutation_operator.mutate(children)

            new_population.add_chromosome(children)

        return new_population

    def get_best_solution(self, n=1):
        """Get the best solutions from the population."""
        return sorted(
            self.population.chromosomes,
            key=lambda chrom: self.evaluator.evaluate(chrom),
            reverse=True,
        )[:n]
