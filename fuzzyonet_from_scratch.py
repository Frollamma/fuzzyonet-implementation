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
        assert mu.ndim == 2, "mu must be a matrix"
        assert sigma.ndim == 2, "sigma must be a matrix"
        assert (
            linear_consequent_coefficients.ndim == 1
        ), "linear_consequent_coefficients must be a vector"

        assert mu.shape == sigma.shape, "mu and sigma should have the same shape"

        assert np.any(sigma <= 0), "sigma must be strictly positive"

        self.mu = mu
        self.sigma = sigma
        self.input_space_dim, self.num_rules = mu.shape

        assert (
            linear_consequent_coefficients.size == self.input_space_dim + 1
        ), "linear_consequent_coefficients should be of dimension of the input space + 1"

        self.linear_consequent_coefficients = linear_consequent_coefficients

    def _get_normalized_firing_strenght(self, x: npt.NDArray[np.floating]):
        membership = gaussian(x, self.mu, self.sigma)
        firing_strenght = np.prod(membership, axis=1)

        return firing_strenght / np.sum(firing_strenght)

    def _get_consequents(self, x: npt.NDArray[np.floating]):
        return self.linear_consequent_coefficients * np.concatenate(x, 1)

    def predict(self, x: npt.NDArray[np.floating]):
        """
        x is the input
        """
        assert (
            x.ndim == 1 and x.shape[0] == self.input_space_dim
        ), "x must be a vector of the input space"

        normalized_firing_strenght = self._get_normalized_firing_strenght(x)
        consequents = self._get_consequents(x)

        return normalized_firing_strenght * consequents
