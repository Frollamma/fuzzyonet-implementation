import numpy as np
import numpy.typing as npt
from fuzzyonet.typing import RealValuedFunctionOfSeveralVariables


def gaussian(
    x: npt.NDArray[np.floating],
    mu: npt.NDArray[np.floating],
    sigma: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return np.exp(-np.power((x - mu), 2) / (2 * np.power(sigma, 2)))


class anfis_function:
    """
    An implementation of an ANFIS function as the described in the FuzzyONet paper, i.e. a fuzzy system where fuzzy sets of the input space are of gaussian form, the inference is the product inference, the fuzzy sets of the output space have linear membership function and the defuzzyfication is the point defuzzyfication.
    It supports inference and training.
    """  # TODO: for now it supports just inference

    def __init__(
        self,
        mu: npt.NDArray[np.floating],
        sigma: npt.NDArray[np.floating],
        linear_consequent_coefficient: npt.NDArray[np.floating],
    ):
        """
        mu and sigma are respectively matrices of the means and standard deviations of the gaussian membership functions of the fuzzy sets in the input space; the rows are indexed by rules and the columns are indexed by the input space dimensions.
        linear_consequent_coefficient: matrix of the coefficients used in the linear combination of the output consequents; the rows are indexed by rules and the columns are indexed by the input space dimensions + 1.
        """
        self.mu = mu
        self.sigma = sigma
        self.num_rules, self.input_space_dim = mu.shape
        self.linear_consequent_coefficient = (
            linear_consequent_coefficient  # TODO: maybe find a better name...?
        )

        assert self.mu.ndim == 2, "mu must be a matrix"
        assert self.sigma.ndim == 2, "sigma must be a matrix"
        assert (
            self.linear_consequent_coefficient.ndim == 2
            and self.linear_consequent_coefficient.shape
            == (self.num_rules, self.input_space_dim + 1)
        ), f"linear_consequent_coefficient must be a (num_rules, input_space_dim + 1) = {(self.num_rules, self.input_space_dim + 1)} matrix"

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
            self.linear_consequent_coefficient,
            np.hstack([x, np.array([1])]),  # Adds 1 at the end of the vector
        )

    def predict(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        x: matrix of input samples or a single sample
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


class anfis_functional:
    """
    An implementation of an ANFIS functional as the described in the FuzzyONet paper, i.e. a functional that for each input function calculates a discretization and then an ANFIS function.
    The input function space is V and the output space is the set of real numbers.
    It supports inference and training.
    """  # TODO: for now it supports just inference

    # TODO: in anfis_function implementation I considered that x can be given in multiple samples, here I just considered that it is a single vector. Think about the general case

    def __init__(
        self,
        mu: npt.NDArray[np.floating],
        sigma: npt.NDArray[np.floating],
        linear_consequent_coefficient: npt.NDArray[np.floating],
        discretization: npt.NDArray[np.floating],
    ):
        """
        mu, sigma and linear_consequent_coefficient are the matrices for the ANFIS function as described in anfis_function.
        discretization: matrix with each row corresponding to an argument of a function of the input function space V, the columns correspond to the input dimension
        """
        self.anfis_function = anfis_function(mu, sigma, linear_consequent_coefficient)
        self.discretization = discretization

        assert (
            discretization.ndim == 1 or discretization.ndim == 2
        ), "discretization must be either a vector or a matrix"

        # If discretization is a vector, convert it into a matrix with a single row
        if discretization.ndim == 1:
            discretization = discretization[None, :]

        (
            self.anfis_function_input_space_dim,  # This is also the cardinality of the discretization
            self.input_space_of_the_function_space_dim,
        ) = discretization.shape

        assert (
            self.anfis_function_input_space_dim == self.anfis_function.input_space_dim
        ), "The cardinality of the discretization must be the same of the dimension of the input space of the anfis function"

        self.num_rules = self.anfis_function.num_rules

    def predict(
        self, u: RealValuedFunctionOfSeveralVariables
    ) -> npt.NDArray[np.floating]:
        """
        u: function in the input function space V.

        Returns the output (a real number) of the ANFIS functional calculated in u
        """
        u_discretization = np.apply_along_axis(
            lambda row: float(u(row)),
            1,
            self.discretization,
        )

        # Feed the discretized function values into the underlying ANFIS function
        return self.anfis_function.predict(u_discretization)


class fuzzyonet:
    """
    An implementation of a FuzzyONet (ANFIS operator) as the described in the FuzzyONet paper, i.e. an operator that is the dot product of a vector-valued ANFIS functional (branch net) and a vector-valued ANFIS function (trunk net)
    The input function space is V and there's also an output function space.
    It supports inference and training.
    """  # TODO: for now it supports just inference

    def __init__(
        self,
        mus_branch: npt.NDArray[np.floating],
        sigmas_branch: npt.NDArray[np.floating],
        linear_consequent_coefficients_branch: npt.NDArray[np.floating],
        discretizations_branch: npt.NDArray[np.floating],
        mus_trunk: npt.NDArray[np.floating],
        sigmas_trunk: npt.NDArray[np.floating],
        linear_consequent_coefficients_trunk: npt.NDArray[np.floating],
    ):
        """
        mus_branch, sigmas_branch, linear_consequent_coefficients_branch and discretizations_branch are 3D tensors for the branch net, where if you fix the third dimension, you get the matrices for an ANFIS functional as described in anfis_functional.
        mus_trunk, sigmas_trunk and linear_consequent_coefficients_trunk are 3D tensors for the trunk net, where if you fix the third dimension, you get the matrices for an ANFIS function as described in anfis_function.
        """
        assert mus_branch.ndim == 3, "mus_branch must be a 3D tensor"
        assert sigmas_branch.ndim == 3, "sigmas_branch must be a 3D tensor"
        assert (
            linear_consequent_coefficients_branch.ndim == 3
        ), "linear_consequent_coefficients_branch must be a 3D tensor"
        assert (
            discretizations_branch.ndim == 3
        ), "discretizations_branch must be a 3D tensor"

        assert mus_trunk.ndim == 3, "mus_trunk must be a 3D tensor"
        assert sigmas_trunk.ndim == 3, "sigmas_trunk must be a 3D tensor"
        assert (
            linear_consequent_coefficients_trunk.ndim == 3
        ), "linear_consequent_coefficients_trunk must be a 3D tensor"

        assert (
            mus_branch.shape[2]
            == sigmas_branch.shape[2]
            == linear_consequent_coefficients_branch.shape[2]
            == discretizations_branch[2]
            == mus_trunk.shape[2]
            == sigmas_trunk.shape[2]
            == linear_consequent_coefficients_trunk.shape[2]
        ), "The third dimension of the tensors mus_branch, sigmas_branch, linear_consequent_coefficients_trunk, discretizations_branch, mus_trunk, sigmas_trunk, linear_consequent_coefficients_trunk must be equal"

        self.branch_and_trunk_output_space_dim = mus_branch.shape[2]

        # TODO: Probably this is pretty inefficient
        self.branch = []
        self.trunk = []
        for i in range(self.branch_and_trunk_output_space_dim):
            self.branch.append(
                anfis_functional(
                    mus_branch[:, :, i],
                    sigmas_branch[:, :, i],
                    linear_consequent_coefficients_branch[:, :, i],
                    discretizations_branch[:, :, i],
                )
            )
            self.trunk.append(
                anfis_function(
                    mus_trunk[:, :, i],
                    sigmas_trunk[:, :, i],
                    linear_consequent_coefficients_trunk[:, :, i],
                )
            )

        self.anfis_function_input_space_dim = self.branch[
            0
        ].anfis_function_input_space_dim
        self.input_space_of_the_function_space_dim = self.branch[
            0
        ].input_space_of_the_function_space_dim
        self.num_rules_branch = self.branch[0].num_rules
        self.num_rules_trunk = self.branch[0].num_rules
        self.trunk_input_space_dim = self.trunk[0].input_space_dim

    def predict(
        self, u: RealValuedFunctionOfSeveralVariables
    ) -> RealValuedFunctionOfSeveralVariables:
        """
        u: function in the input function space V.

        Returns the output (a function) of the FuzzyONet calculated in u
        """

        branch_output = []
        for branch_ANFIS_functional in self.branch:
            branch_output.append(branch_ANFIS_functional(u))

        branch_output = np.array(branch_output)

        def output_function(x: npt.NDArray[np.floating]):
            assert (
                x.ndim == 1
            )  # TODO: you should consider the same of multiple samples?
            assert x.shape[0] == self.trunk_input_space_dim

            trunk_output = []
            for trunk_ANFIS_function in self.trunk:
                trunk_output.append(trunk_ANFIS_function(x))

            trunk_output = np.array(trunk_output)

            return np.dot(branch_output, trunk_output)

        return output_function
