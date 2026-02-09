from typing import Callable
import numpy as np
import numpy.typing as npt

# R -> R
RealValuedFunctionOfOneVariable = Callable[[float], float]
# R^n -> R
RealValuedFunctionOfSeveralVariables = Callable[[npt.NDArray[np.floating]], float]
# R -> R^m
VectorValuedFunctionOfOneVariable = Callable[[float], npt.NDArray[np.floating]]
# R^n -> R^m
VectorValuedFunctionOfSeveralVariables = Callable[
    [npt.NDArray[np.floating]], npt.NDArray[np.floating]
]
