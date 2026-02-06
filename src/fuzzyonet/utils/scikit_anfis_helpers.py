from skanfis.fs import FS
import numpy as np
import pandas as pd


def fs_infer_batch(fs: FS, X: pd.DataFrame, verbose=False):
    """
    X: pandas DataFrame with columns matching fs variable names (e.g. x0, x1, banana, etc.)
    Returns: numpy ndarray, shape (n_samples, n_outputs)
    """
    names = X.columns.tolist()
    results = []

    for idx in X.index:
        for name in names:
            fs.set_variable(name, float(X.loc[idx, name]))
        results.append([v for v in fs.inference(verbose=verbose).values()])

    results = np.array(results)
    return results
