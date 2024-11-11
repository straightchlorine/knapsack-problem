import numpy as np

from knapsack.chromosome import Chromosome
from knapsack.population import Population
from knapsack.evaluators.evaluator import Evaluator
from knapsack.selectors.selector import Selector


class GeneticAlgorithm:
    def __init__(
        self,
        dataset,
        evaluator: Evaluator,
        selector: Selector,
        population_size=100,
        num_generations=500,
        mutation_rate=0.01,
    ):
        self.dataset = dataset
        self.gene_length = self.dataset.length

        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

        self.evaluator = evaluator
        self.selector = selector

        # create and initialize the population
        self.population = Population(
            self.selector, self.population_size, self.gene_length
        )
        self.population.initialize()

    def crossover(self, parent1, parent2):
        # choose random point for crossover
        point = np.random.randint(1, parent1.size - 1)

        # generate genes for two children
        genes = (
            np.concatenate([parent1[:point], parent2[point:]]),
            np.concatenate([parent1[point:], parent2[:point]]),
        )

        # return two chromosomes
        return np.array([genes[0], genes[1]])

    def mutate(self, chromosome):
        # iterate through the genes
        for i in chromosome:
            # if randomly selected probability is less than set, flip the genes
            if np.random.rand() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    def new_generation(self):
        # new generation
        new_population = Population(
            self.selector, self.population_size, self.gene_length
        )

        while len(new_population.chromosomes) < self.population_size:
            parents = self.population.select_parents()
            # chose two parents from the selected chromosomes
            parent1, parent2 = parents

            # create a child by crossover and mutation
            children = self.crossover(parent1, parent2)

            # mutate the children
            for child in children:
                child = self.mutate(child)

            # add the child to the new population
            new_population.add_chromosome(children)

        new_population.update_selector()

        return new_population

    def evolve(self):
        """Evolve the population for a set number of generations."""
        for _ in range(self.num_generations):
            self.population = self.new_generation()

    def prompt(self, evaluated):
        # total_weight = np.sum(evaluated.genes * self.dataset.weights)
        # total_value = np.sum(evaluated.genes * self.dataset.values)
        total_weight = np.sum(evaluated * self.dataset.weights)
        total_value = np.sum(evaluated * self.dataset.values)
        capacity = float(self.dataset.capacity)

        total_prompt = "\n".join(
            [
                "=" * 30,
                str(evaluated),
                "=" * 30,
                f"total weight: {total_weight} | capacity: {capacity}",
                f"total value: {total_value}",
            ]
        )
        print(total_prompt)

    def get_best_solution(self, n=1):
        ordered = sorted(
            self.population.chromosomes,
            key=lambda chrom: self.evaluator.evaluate(chrom),
            reverse=True,
        )[:n]

        print(f"\n{self.dataset}")

        for evaluated in ordered:
            self.prompt(evaluated)

        return ordered[0]
