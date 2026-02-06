import numpy as np
import numpy.typing as npt


def gaussian(
    x: npt.NDArray[np.floating],
    mu: npt.NDArray[np.floating],
    sigma: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return np.exp(-np.power((x - mu), 2) / (2 * np.power(sigma, 2)))


class anfis_function:
    def __init__(
        self,
        mu: npt.NDArray[np.floating],
        sigma: npt.NDArray[np.floating],
        linear_consequent_coefficients: npt.NDArray[np.floating],
    ):
        self.mu = mu
        self.sigma = sigma
        self.num_rules, self.input_space_dim = mu.shape
        self.linear_consequent_coefficients = linear_consequent_coefficients

        assert self.mu.ndim == 2, "mu must be a matrix"
        assert self.sigma.ndim == 2, "sigma must be a matrix"
        assert (
            self.linear_consequent_coefficients.ndim == 2
            and self.linear_consequent_coefficients.shape
            == (self.num_rules, self.input_space_dim + 1)
        ), f"linear_consequent_coefficients must be a (num_rules, input_space_dim + 1) = {(self.num_rules, self.input_space_dim + 1)} matrix"

        assert (
            self.mu.shape == self.sigma.shape
        ), "mu and sigma should have the same shape"

        assert np.all(self.sigma > 0), "sigma must be strictly positive"

    def _get_normalized_firing_strenght(self, x: npt.NDArray[np.floating]):
        membership = gaussian(x, self.mu, self.sigma)
        firing_strenght = np.prod(membership, axis=1)

        return firing_strenght / np.sum(firing_strenght)

    def _get_consequents(self, x: npt.NDArray[np.floating]):
        return np.matmul(
            self.linear_consequent_coefficients,
            np.hstack([x, np.array([1])]),  # Adds 1 at the end of the vector
        )

    def predict(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        x is a matrix of input samples or a single sample
        """
        assert (x.ndim == 1 and x.size == self.input_space_dim) or (
            x.ndim == 2 and x.shape[1] == self.input_space_dim
        ), f"x must be either a vector of the input space, that has dimension {self.input_space_dim}, or a set of samples in matrix form with dimensions (num_samples, input_space_dim)"

        if x.ndim == 1:
            x = x[None, :]

        # TODO: the implementation is inefficient, improve it
        output = []
        for i in range(x.shape[0]):
            normalized_firing_strenght = self._get_normalized_firing_strenght(x[i])
            consequents = self._get_consequents(x[i])
            output.append(np.dot(normalized_firing_strenght, consequents))

        output = np.array(output)

        return output
