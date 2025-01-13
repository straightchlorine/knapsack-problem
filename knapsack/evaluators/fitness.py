import numpy as np
import logging
from knapsack.evaluators.evaluator import Evaluator

"""
Both of the evaluators fulfill the requirements.

    * take into account sums of weights and values as well as the capacity
    * both of them penalize chromosomes exceeding the threshold

Note: Test cases will use low population and generation numbers to show some
variety in the results. The goal is to show evaluation rather than correct
results.

Test cases:

    * FitnessEvaluator (5 pop; 5 gen):

        Evaluating chromosomes via FitnessEvaluator
        Dataset(weights=[48. 42. 49. 23. 32.], values=[ 5.  2.  3. 11. 18.], capacity=92)

        Selecting parents parents via ElitismSelector
        ==============================
        genes=[0 0 0 0 1]
        ==============================
        total weight: 32.0 | capacity: 92.0
        total value: 18.0 | evaluation: 18.0
        ==============================
        genes=[0 0 0 0 1]
        ==============================
        total weight: 32.0 | capacity: 92.0
        total value: 18.0 | evaluation: 18.0
        ==============================
        genes=[0 0 0 0 1]
        ==============================
        total weight: 32.0 | capacity: 92.0
        total value: 18.0 | evaluation: 18.0
        ==============================
        genes=[0 0 0 1 0]
        ==============================
        total weight: 23.0 | capacity: 92.0
        total value: 11.0 | evaluation: 11.0
        ==============================
        genes=[0 0 0 1 0]
        ==============================
        total weight: 23.0 | capacity: 92.0
        total value: 11.0 | evaluation: 11.0

        =========== Failed to get the best solution. ===========

        Evaluating chromosomes via FitnessEvaluator
        Dataset(weights=[40. 27. 20.  6. 13.], values=[ 8. 17. 14. 13. 18.], capacity=17)

        Selecting parents parents via ElitismSelector
        ==============================
        genes=[0 0 0 1 0]
        ==============================
        total weight: 6.0 | capacity: 17.0
        total value: 13.0 | evaluation: 13.0
        ==============================
        genes=[0 0 0 1 0]
        ==============================
        total weight: 6.0 | capacity: 17.0
        total value: 13.0 | evaluation: 13.0
        ==============================
        genes=[0 0 0 1 0]
        ==============================
        total weight: 6.0 | capacity: 17.0
        total value: 13.0 | evaluation: 13.0
        ==============================
        genes=[1 0 0 0 0]
        ==============================
        total weight: 40.0 | capacity: 17.0
        total value: 8.0 | evaluation: 0
        ==============================
        genes=[1 0 0 0 0]
        ==============================
        total weight: 40.0 | capacity: 17.0
        total value: 8.0 | evaluation: 0

        =========== Failed to get the best solution. ===========

        Evaluating chromosomes via FitnessEvaluator
        Dataset(weights=[46. 28. 11. 35. 40.], values=[ 2.  1. 18.  8. 11.], capacity=16)

        Selecting parents parents via ElitismSelector
        ==============================
        genes=[0 0 1 0 0]
        ==============================
        total weight: 11.0 | capacity: 16.0
        total value: 18.0 | evaluation: 18.0
        ==============================
        genes=[0 0 1 0 0]
        ==============================
        total weight: 11.0 | capacity: 16.0
        total value: 18.0 | evaluation: 18.0
        ==============================
        genes=[0 0 1 0 0]
        ==============================
        total weight: 11.0 | capacity: 16.0
        total value: 18.0 | evaluation: 18.0
        ==============================
        genes=[1 0 0 0 0]
        ==============================
        total weight: 46.0 | capacity: 16.0
        total value: 2.0 | evaluation: 0
        ==============================
        genes=[1 0 0 0 0]
        ==============================
        total weight: 46.0 | capacity: 16.0
        total value: 2.0 | evaluation: 0

        =========== Managed to get the best solution. ===========

    * ScalingFitnessEvaluator (5 pop; 5 gen):

        Evaluating chromosomes via ScalingFitnessEvaluator
        Dataset(weights=[37. 44. 49. 16. 30.], values=[12. 17.  6.  6.  2.], capacity=16)

        Selecting parents parents via ElitismSelector
        ==============================
        genes=[0 0 0 1 0]
        ==============================
        total weight: 16.0 | capacity: 16.0
        total value: 6.0 | evaluation: 6.0
        ==============================
        genes=[0 0 0 1 0]
        ==============================
        total weight: 16.0 | capacity: 16.0
        total value: 6.0 | evaluation: 6.0
        ==============================
        genes=[0 0 0 1 0]
        ==============================
        total weight: 16.0 | capacity: 16.0
        total value: 6.0 | evaluation: 6.0
        ==============================
        genes=[1 0 0 0 0]
        ==============================
        total weight: 37.0 | capacity: 16.0
        total value: 12.0 | evaluation: -93.0
        ==============================
        genes=[1 0 0 0 0]
        ==============================
        total weight: 37.0 | capacity: 16.0
        total value: 12.0 | evaluation: -93.0

        =========== Managed to get the best solution. ===========

        Evaluating chromosomes via ScalingFitnessEvaluator
        Dataset(weights=[46.  3. 14. 26. 42.], values=[ 3. 16.  3.  1. 10.], capacity=25)

        Selecting parents parents via ElitismSelector
        ==============================
        genes=[0 1 1 1 0]
        ==============================
        total weight: 43.0 | capacity: 25.0
        total value: 20.0 | evaluation: -70.0
        ==============================
        genes=[0 1 1 1 0]
        ==============================
        total weight: 43.0 | capacity: 25.0
        total value: 20.0 | evaluation: -70.0
        ==============================
        genes=[0 1 1 1 0]
        ==============================
        total weight: 43.0 | capacity: 25.0
        total value: 20.0 | evaluation: -70.0
        ==============================
        genes=[1 1 1 0 0]
        ==============================
        total weight: 63.0 | capacity: 25.0
        total value: 22.0 | evaluation: -168.0
        ==============================
        genes=[1 1 1 0 0]
        ==============================
        total weight: 63.0 | capacity: 25.0
        total value: 22.0 | evaluation: -168.0

        =========== Failed to get the best solution. ===========

        Evaluating chromosomes via ScalingFitnessEvaluator
        Dataset(weights=[44. 26. 30. 35. 16.], values=[ 4.  8. 15. 16. 10.], capacity=72)

        Selecting parents parents via ElitismSelector
        ==============================
        genes=[0 0 1 1 0]
        ==============================
        total weight: 65.0 | capacity: 72.0
        total value: 31.0 | evaluation: 31.0
        ==============================
        genes=[0 0 1 1 0]
        ==============================
        total weight: 65.0 | capacity: 72.0
        total value: 31.0 | evaluation: 31.0
        ==============================
        genes=[0 0 1 1 0]
        ==============================
        total weight: 65.0 | capacity: 72.0
        total value: 31.0 | evaluation: 31.0
        ==============================
        genes=[0 1 1 0 0]
        ==============================
        total weight: 56.0 | capacity: 72.0
        total value: 23.0 | evaluation: 23.0
        ==============================
        genes=[0 1 1 0 0]
        ==============================
        total weight: 56.0 | capacity: 72.0
        total value: 23.0 | evaluation: 23.0

        =========== Managed to get the best solution. ===========
"""


class FitnessEvaluator(Evaluator):
    def __init__(self, dataset):
        """Basic fitness evaluator.

        Evaluates to 0, if weight exceeds capacity.

        Args:
            dataset: An object containing weights, values, and capacity.
        """
        super().__init__(dataset)

    def evaluate(self, chromosome):
        """Evaluate the fitness of a chromosome.

        Args:
            chromosome: A binary array representing the chromosome.

        Returns:
            float: The fitness value of the chromosome, or 0 if weight exceeds capacity.
        """
        total_weight = np.sum(chromosome * self.dataset.weights)
        total_value = np.sum(chromosome * self.dataset.values)

        if total_weight > self.dataset.capacity:
            return 0
        return total_value


class ScalingFitnessEvaluator(Evaluator):
    def __init__(
        self, dataset, normalize=False, std_threshold=3, scaling_factor=None
    ):
        """Scaling fitness evaluator.

        Punishment scales with the amount of weight exceeding the capacity.

        Args:
            dataset: An object containing weights, values, and capacity.
            normalize (bool): Whether to normalize weights and values (default: False).
            std_threshold (float): Threshold for standard deviation to mean ratio to apply normalization.
            scaling_factor (float): Scaling factor for penalty calculation. If None, it is computed dynamically.
        """
        super().__init__(dataset)

        # calculate standard deviation ratios for both values and weights
        weight_std_ratio = np.std(self.dataset.weights) / np.mean(
            self.dataset.weights
        )
        value_std_ratio = np.std(self.dataset.values) / np.mean(
            self.dataset.values
        )
        logging.debug("Computed standard deviation ratios.")
        logging.info(f"Weight std ratio: {weight_std_ratio:.4f}")
        logging.info(f"Value std ratio: {value_std_ratio:.4f}")

        # depending on the threshold and the std, decide whether to normalize
        self.normalize = (
            normalize
            or weight_std_ratio > std_threshold
            or value_std_ratio > std_threshold
        )

        if self.normalize:
            self.norm_weights = self.dataset.weights / np.sum(
                self.dataset.weights
            )
            self.norm_values = self.dataset.values / np.sum(
                self.dataset.values
            )
            logging.info("Normalization applied for both values and weights.")
        else:
            self.norm_weights = self.dataset.weights
            self.norm_values = self.dataset.values
            logging.info("Normalization not applied.")

        # calculate scaling factor
        if scaling_factor is None:
            self.scaling_factor = np.mean(self.dataset.values) / np.mean(
                self.dataset.weights
            )
            logging.info(
                f"Calculated scaling factor: {self.scaling_factor:.4f}"
            )
        else:
            self.scaling_factor = scaling_factor
            logging.info(f"Scaling factor used: {self.scaling_factor:.4f}")

    def evaluate(self, chromosome):
        """Evaluate the fitness of a chromosome.

        Args:
            chromosome: A binary array representing the chromosome.

        Returns:
            float: The fitness value of the chromosome.
        """
        total_weight = np.sum(chromosome * self.norm_weights)
        total_value = np.sum(chromosome * self.norm_values)

        if total_weight > self.dataset.capacity:
            excess_weight = total_weight - self.dataset.capacity
            penalty = (
                excess_weight * len(self.dataset.weights) * self.scaling_factor
            )
            return total_value - penalty

        return total_value
