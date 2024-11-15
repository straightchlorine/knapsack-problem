import numpy as np

from knapsack.evaluators.evaluator import Evaluator


class FitnessEvaluator(Evaluator):
    def __init__(self, dataset):
        """Basic fitness evaluator.

        Evaluates chromosome to 0, if weight exceeds capacity.
        """
        super().__init__(dataset)

    def evaluate(self, chromosome):
        # total_weight = np.sum(chromosome.genes * self.dataset.weights)
        # total_value = np.sum(chromosome.genes * self.dataset.values)
        total_weight = np.sum(chromosome * self.dataset.weights)
        total_value = np.sum(chromosome * self.dataset.values)

        if total_weight > self.dataset.capacity:
            total_value = 0

        return total_value


class ScalingFitnessEvaluator(Evaluator):
    def __init__(self, dataset):
        """Scaling fitness evaluator.

        Punishment scales with the amount of weight exceeding the capacity.
        """
        super().__init__(dataset)

    def evaluate(self, chromosome):
        total_weight = np.sum(chromosome.genes * self.dataset.weights)
        total_value = np.sum(chromosome.genes * self.dataset.values)

        # punishment scales with the amount of weight exceeding the capacity
        if total_weight > self.dataset.capacity:
            penalty = total_weight - self.dataset.capacity
            return total_value - penalty

        return total_value
