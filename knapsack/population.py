import numpy as np

from knapsack.dataset import Dataset
from knapsack.selectors.selector import Selector


class Population:
    _dataset: Dataset

    def __init__(
        self, dataset: Dataset, selector: Selector, population_size=100, gene_length=5
    ):
        """Population object, representing a group of chromosomes.

        Args:
            selector (Selector): Selector object, used to select the parents.
            population_size (int): Size of the population.
            gene_length (int): Length of the gene.
        """
        self.population_size = population_size
        self.gene_length = gene_length

        # set the selector
        self.selector = selector

        # empty list of chromosomes
        self.chromosomes: np.ndarray = np.array([])
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    def update_selector(self):
        """Update the selector with a new population of chromosomes."""
        self.selector.population = self.chromosomes

    def __gen_genes_uniform(self):
        """Generate a chromosome with random genes using uniform distribution."""
        return np.random.choice([0, 1], size=self.gene_length)

    def __gen_genes_normal(self):
        """Generate a chromosome using a normal distribution."""
        probabilities = np.random.normal(0.5, 0.15, self.gene_length)
        probabilities = np.clip(probabilities, 0, 1)  # ensure between 1 and 0
        return (probabilities > 0.5).astype(int)

    def __gen_genes_value_biased(self):
        """Generate chromosome biased towards high-value items.

        Value-biased heuristic.

        1. Calculate the probabilities of selecting each item (proportional to value).
        2. Generate random probabilities for each item.
        3. If probability is less than the value probability, return 1, else 0.
        4. Return the chromosome.
        """
        value_probs = np.array(self.dataset.values) / max(self.dataset.values)
        probabilities = np.random.uniform(0, 1, self.gene_length)
        return (probabilities < value_probs).astype(int)

    def __gen_genes_value_weight_ratio(self):
        """Generate chromosome based on value-to-weight ratios.

        Weight to value ration heuristic.

        1. Calculate the value-to-weight ratios for each item.
        2. Normalize the ratios.
        3. Generate random probabilities for each item (uniform distribution).
        4. If probability is less than the normalized ratio, return 1, else 0.
        5. Return the chromosome.
        """
        ratios = np.array(self.dataset.values) / np.array(self.dataset.weights)
        normalized_ratios = ratios / max(ratios)
        probabilities = np.random.uniform(0, 1, self.gene_length)
        return (probabilities < normalized_ratios).astype(int)

    def __gen_genes_weight_constrained(self):
        """Generate chromosome ensuring total weight is close to capacity.

        Weight-constrained heuristic.

        1. Generate a random permutation of the items.
        2. Add items to the chromosome until the total weight is close to the capacity.
        3. Return the chromosome.
        """
        chromosome = np.zeros(self.gene_length)
        current_weight = 0
        for idx in np.random.permutation(self.gene_length):
            if current_weight + self.dataset.weights[idx] <= self.dataset.capacity:
                chromosome[idx] = 1
                current_weight += self.dataset.weights[idx]
        return chromosome.astype(int)

    def __gen_genes_greedy_randomized(self):
        """Generate chromosome using a greedy randomized approach.

        Greedy randomized heuristic.

        1. Calculate the value-to-weight ratios for each item.
        2. Sort the items by the ratio.
        3. Take the top 60% of the items with higher probability.
        4. Generate a random probability for each item.
        5.1 If the item is in the top 60%, use a higher probability (80%)
        5.2 If the item is in the bottom 40%, use a lower probability (20%)
        6. Return the chromosome.
        """
        # calculate value to weight rations and sort by them
        # rations looks like [(idx, ratio), ...]
        ratios = [
            (i, v / w)
            for i, (v, w) in enumerate(
                zip(
                    self.dataset.values,
                    self.dataset.weights,
                )
            )
        ]
        ratios.sort(key=lambda x: x[1], reverse=True)

        # top 60% of items have higher probability 80%
        split_idx = int(self.gene_length * 0.6)
        chromosome = np.zeros(self.gene_length)

        # higher probability of taking the top 60% of items
        # for i = 0 idx could be 3 (if 3 had the best ratio)
        for i, (idx, _) in enumerate(ratios):
            # if item belongs to top 60%, 80% for this item to be in the
            # final chromosome
            if i < split_idx:
                chromosome[idx] = np.random.choice([0, 1], p=[0.2, 0.8])
            else:
                chromosome[idx] = np.random.choice([0, 1], p=[0.8, 0.2])

        return chromosome.astype(int)

    def __gen_genes_small_large_mix(self):
        """Generate chromosome mixing small and large items.

        Heuristic to mix small and large items.

        1. Sort items by weight (ascending).
        2. Find one small item i.e. in the top 40% of weights and choose by random.
        3. Verify how much weight is left.
        4. Include items randomly with lower probability (30%) until the capacity is reached.
        5. Return the chromosome.
        """
        # sort by weight and create empty chromosome
        sorted_indices = np.argsort(self.dataset.weights)
        chromosome = np.zeros(self.gene_length)

        # include one small element (find it among first 40% of weights)
        small_items = sorted_indices[: int(self.gene_length * 0.4)]
        chosen_small = np.random.choice(small_items)
        chromosome[chosen_small] = 1

        # check how much weight is left
        remaining_weight = self.dataset.capacity - self.dataset.weights[chosen_small]
        for idx in sorted_indices:
            if idx != chosen_small and self.dataset.weights[idx] <= remaining_weight:
                if np.random.random() < 0.3:
                    chromosome[idx] = 1
                    remaining_weight -= self.dataset.weights[idx]

        return chromosome.astype(int)

    def initialize_with_strategy(self, strategy="value_biased"):
        """Initialize the population based on a chosen strategy."""
        strategies = {
            "value_biased": self.__gen_genes_value_biased,
            "value_weight_ratio": self.__gen_genes_value_weight_ratio,
            "weight_constrained": self.__gen_genes_weight_constrained,
            "greedy_randomized": self.__gen_genes_greedy_randomized,
            "small_large_mix": self.__gen_genes_small_large_mix,
            "uniform": self.__gen_genes_uniform,
            "normal": self.__gen_genes_normal,
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.chromosomes = np.array(
            [strategies[strategy]() for _ in range(self.population_size)]
        )
        self.update_selector()

    def select_parents(self) -> np.ndarray:
        """Select parents for the next generation.

        Returns:
            np.ndarray: Array of selected parents.
        """
        return self.selector.select()

    def measure_diversity(self):
        """Calculate genetic diversity in the population."""
        unique = {tuple(c) for c in self.chromosomes}
        return len(unique) / len(self.chromosomes) * 100

    def add_chromosome(self, chromosome: np.ndarray):
        """Add a chromosome or a list of them to the population.

        Args:
            chromosome (np.ndarray): Chromosome to add.

        Returns:
            int: Index of the added chromosome.
        """
        # initialize `self.chromosomes` as an empty array if it hasn't been yet
        if not hasattr(self, "chromosomes") or self.chromosomes.size == 0:
            self.chromosomes = np.empty((0, chromosome.shape[1]), dtype=chromosome.dtype)

        self.chromosomes = np.vstack([self.chromosomes, chromosome])
        return len(self.chromosomes) - 1
