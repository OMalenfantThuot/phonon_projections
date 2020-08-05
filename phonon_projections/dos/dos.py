import numpy as np


class Dos:
    def __init__(self, energies, weights):
        r"""
        """
        self.energies = energies
        self.weights = weights
        assert len(self.energies) == len(
            self.weights
        ), "Energy and weights should be the same size."
        self.emin = self.energies.min()
        self.emax = self.energies.max()

    @property
    def energies(self):
        return self._energies

    @energies.setter
    def energies(self, energies):
        self._energies = np.array(energies)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = np.array(weights)

    @property
    def emin(self):
        return self._emin

    @property
    def emax(self):
        return self._emax

    @emin.setter
    def emin(self, emin):
        self._emin = emin

    @emax.setter
    def emax(self, emax):
        self._emax = emax

    def __getitem__(self, idx):
        return np.array([self.energies[idx], self.weights[idx]])

    def get_weight(self, energy):
        assert (
            self.emin <= energy <= self.emax
        ), "The weight asked for corresponds to an energy out of the dos."
        idx = np.argmin(np.abs(self.energies - energy))

        if self.energies[idx] <= energy:
            idxmin = idx
            idxmax = idx + 1
        else:
            idxmax = idx
            idxmin = idx - 1

        weight = (
            self.weights[idxmin] * (self.energies[idxmax] - energy)
            + self.weights[idxmax] * (energy - self.energies[idxmin])
        ) / (self.energies[idxmax] - self.energies[idxmin])
        return weight
