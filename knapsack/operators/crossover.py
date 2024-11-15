class Crossover:
    """Base class for crossover operators."""

    def __init__(self, dev=False):
        self.dev = dev
        print(f"Crossover performed by {self.__class__.__name__}", end=" ")

        if self.dev:
            print("with dev mode enabled\n")
        else:
            print("\n")

    def crossover(self, parent_a, parent_b):
        """Perform crossover in order to generate offspring.

        Args:
            parent_a (np.ndarray): First parent.
            parent_b (np.ndarray): Second parent.

        Returns:
            np.ndarray: Children from the crossover operation.
        """
        raise NotImplementedError(
            "Method 'crossover' must be implemented in a subclass."
        )
