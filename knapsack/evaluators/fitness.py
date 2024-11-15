import numpy as np

from knapsack.evaluators.evaluator import Evaluator


class FitnessEvaluator(Evaluator):
    def __init__(self, dataset):
        """Basic fitness evaluator.

        Evaluates to 0, if weight exceeds capacity.
        """
        super().__init__(dataset)

    def evaluate(self, chromosome):
        total_weight = np.sum(chromosome * self.dataset.weights)
        total_value = np.sum(chromosome * self.dataset.values)

        # value to 0, if weight exceeds capacity
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
        # if weights would be several magnitudes different than the values,
        # it could consider normalization, like:
        # norm_weights = self.dataset.weights / np.sum(self.dataset.weights)
        # norm_values = self.dataset.values / np.sum(self.dataset.values)

        total_weight = np.sum(chromosome * self.dataset.weights)
        total_value = np.sum(chromosome * self.dataset.values)

        # punishment scales with the amount of weight exceeding the capacity
        # and amount of elements within the array
        if total_weight > self.dataset.capacity:
            penalty = (total_weight - self.dataset.capacity) * len(
                self.dataset.weights
            )
            return total_value - penalty

        return total_value
