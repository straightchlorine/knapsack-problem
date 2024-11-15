#!/usr/bin/env python

import numpy as np
import pandas as pd


class DataInterface:
    def __init__(self, weights, values, capacities):
        """Main class for the dataset management.

        Fields:
            weights (np.ndarray): Array of weights.
            values (np.ndarray): Array of the values corresponding to each
                weight.
            capacity (int): Total capacity of the knapsack.
        """
        self._weights = np.array(weights)
        self._values = np.array(values)
        self._capacity = np.array(capacities)
        self._gene_length = len(self._weights[0])

        self._chromosome_datasets = []

        for w, v, c in zip(weights, values, capacities):
            self._chromosome_datasets.append(Dataset(w, v, c))

    def random_problem(self):
        """Return random chromosome from the dataset."""
        return np.random.choice(self._chromosome_datasets, size=1)[0]

    @classmethod
    def from_csv(cls, filepath):
        """Load datasets from CSV file and return a DatasetInterface object.

        Note: The CSV file is assumend to have three columns - weight, values
        and capacity, such as:

                 Weights	         Values  	  Capacity
            [46 40 42 38 10]	[12 19 19 15  8]	40
            [11 31  4  6  7]	[ 2  8 18 16  3]	64

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            DatasetInterface: Object storing all the datasets.
        """
        df = pd.read_csv(filepath, delimiter=";")

        if (
            df.columns[0] == "Weight"
            and df.columns[1] == "Value"
            and df.columns[2] == "Capacity"
        ):
            df.rename(
                columns={
                    "Weight": "weights",
                    "Value": "values",
                    "Capacity": "capacities",
                },
                inplace=True,
            )

        # parsing the string representation of the arrays into numpy arrays
        df["weights"] = df["weights"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" ")
        )
        df["values"] = df["values"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" ")
        )

        # loading each parameters into numpy arrays
        weights = df["weights"].values
        values = df["values"].values
        capacities = df["capacities"]

        return cls(weights, values, capacities)

    @property
    def weights(self):
        """The weights property."""
        return self._weights

    @property
    def values(self):
        """The values property."""
        return self._values

    @property
    def capacity(self):
        """The capacity property."""
        return self._capacity

    @property
    def chromosome_datasets(self):
        """The chromosome_datasets property."""
        return self._chromosome_datasets


class Dataset:
    def __init__(self, weights, values, capacity):
        """Class representing the problem dataset.

        Args:
            weights (np.ndarray): weights.
            values (np.ndarray): values corresponding to each weight.
            capacity (int): Total capacity of the knapsack.
        """
        self._weights = weights
        self._values = values
        self._capacity = capacity

    def __repr__(self):
        weights = f"weights={self.weights}"
        values = f"values={self.values}"
        capacity = f"capacity={self.capacity}"
        return f"Dataset({weights}, {values}, {capacity})"

    @property
    def weights(self):
        """The weights property."""
        return self._weights

    @property
    def values(self):
        """The values property."""
        return self._values

    @property
    def capacity(self):
        """The capacity property."""
        return self._capacity

    @property
    def length(self):
        """The length property."""
        return len(self.weights)
