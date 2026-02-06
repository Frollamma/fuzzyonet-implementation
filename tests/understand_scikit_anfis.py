from skanfis import scikit_anfis
from skanfis.fs import FS, GaussianFuzzySet, LinguisticVariable
import pandas as pd
from fuzzyonet.utils.scikit_anfis_helpers import fs_infer_batch


SEPARATOR = "\n//////////"

print(f"{SEPARATOR} Initializing scikit_anfis with no params...")
try:
    model_no_params = scikit_anfis()
    print(f"{model_no_params.__dict__ = }")
    print("SUCCESS")
except Exception as e:
    print(f"FAIL: Failed to initialize scikit_anfis with no params. Error: {e}")


print(f"{SEPARATOR} Setting up fuzzy system...")
# Create a TSK fuzzy system object by default
fs = FS()

# Define fuzzy sets and linguistic variables
S_1 = GaussianFuzzySet(mu=2, sigma=0.4270, term="mf0")
S_2 = GaussianFuzzySet(mu=2, sigma=1.3114, term="mf1")
fs.add_linguistic_variable("x0", LinguisticVariable([S_1, S_2]), verbose=True)
S_3 = GaussianFuzzySet(mu=2, sigma=0.4270, term="mf0")
S_4 = GaussianFuzzySet(mu=2, sigma=1.3114, term="mf1")
fs.add_linguistic_variable("x1", LinguisticVariable([S_3, S_4]), verbose=True)
F_1 = GaussianFuzzySet(mu=2, sigma=0.4270, term="mf0")
F_2 = GaussianFuzzySet(mu=2, sigma=1.3114, term="mf1")
fs.add_linguistic_variable("x2", LinguisticVariable([F_1, F_2]), verbose=True)
F_3 = GaussianFuzzySet(mu=2, sigma=0.4270, term="mf0")
F_4 = GaussianFuzzySet(mu=2, sigma=1.3114, term="mf1")
fs.add_linguistic_variable("x3", LinguisticVariable([F_3, F_4]), verbose=True)

# Define fuzzy rules
R1 = "IF (x0 IS mf0) AND (x1 IS mf0) AND (x2 IS mf0) AND (x3 IS mf0) THEN (y0 IS chaotic0)"
R2 = "IF (x0 IS mf0) AND (x1 IS mf0) AND (x2 IS mf0) AND (x3 IS mf1) THEN (y0 IS chaotic1)"
R3 = "IF (x0 IS mf0) AND (x1 IS mf0) AND (x2 IS mf1) AND (x3 IS mf0) THEN (y0 IS chaotic2)"
R4 = "IF (x0 IS mf0) AND (x1 IS mf0) AND (x2 IS mf1) AND (x3 IS mf1) THEN (y0 IS chaotic3)"
R5 = "IF (x0 IS mf0) AND (x1 IS mf1) AND (x2 IS mf0) AND (x3 IS mf0) THEN (y0 IS chaotic4)"
R6 = "IF (x0 IS mf0) AND (x1 IS mf1) AND (x2 IS mf0) AND (x3 IS mf1) THEN (y0 IS chaotic5)"
R7 = "IF (x0 IS mf0) AND (x1 IS mf1) AND (x2 IS mf1) AND (x3 IS mf0) THEN (y0 IS chaotic6)"
R8 = "IF (x0 IS mf0) AND (x1 IS mf1) AND (x2 IS mf1) AND (x3 IS mf1) THEN (y0 IS chaotic7)"
R9 = "IF (x0 IS mf1) AND (x1 IS mf0) AND (x2 IS mf0) AND (x3 IS mf0) THEN (y0 IS chaotic8)"
R10 = "IF (x0 IS mf1) AND (x1 IS mf0) AND (x2 IS mf0) AND (x3 IS mf1) THEN (y0 IS chaotic9)"
R11 = "IF (x0 IS mf1) AND (x1 IS mf0) AND (x2 IS mf1) AND (x3 IS mf0) THEN (y0 IS chaotic10)"
R12 = "IF (x0 IS mf1) AND (x1 IS mf0) AND (x2 IS mf1) AND (x3 IS mf1) THEN (y0 IS chaotic11)"
R13 = "IF (x0 IS mf1) AND (x1 IS mf1) AND (x2 IS mf0) AND (x3 IS mf0) THEN (y0 IS chaotic12)"
R14 = "IF (x0 IS mf1) AND (x1 IS mf1) AND (x2 IS mf0) AND (x3 IS mf1) THEN (y0 IS chaotic13)"
R15 = "IF (x0 IS mf1) AND (x1 IS mf1) AND (x2 IS mf1) AND (x3 IS mf0) THEN (y0 IS chaotic14)"
R16 = "IF (x0 IS mf1) AND (x1 IS mf1) AND (x2 IS mf1) THEN (y0 IS chaotic15)"
fs.add_rules([R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16])

# Define output functions
for i in range(15):
    fs.set_output_function("chaotic" + str(i), "2*x0+2*x1+2*x2+2*x3+1")
fs.set_output_function("chaotic15", "2*x0+2*x1+2*x2+0*x3+1")

print(f"{SEPARATOR} Initializing scikit_anfis from fs...")
try:
    model_from_fs = scikit_anfis(fs=fs)
    print(f"{model_from_fs.__dict__ = }")
    print("SUCCESS")
except Exception as e:
    print(f"FAIL: Failed to initialize scikit_anfis form fs. Error: {e}")

X = pd.DataFrame([[0, 1, 2, 3], [1, 1, 0, 2]], columns=["x0", "x1", "x2", "x3"])

print(f"{SEPARATOR} Evaluating fuzzy system using the predict method...")
try:
    y = fs.predict(X)
    print(f"{y = }")
    print("You will notice that the values are rounded... Why?")
except Exception as e:
    print(
        f"FAIL: Failed to evaluate fuzzy system using the predict method with error {e}"
    )


print(f"{SEPARATOR} Evaluating fuzzy system using the inference method...")
try:
    y = fs_infer_batch(fs, X)
    print(f"{y = }")
    print("You will notice that the values are not rounded!")
except Exception as e:
    print(
        f"FAIL: Failed to evaluate fuzzy system using the inference method with error {e}"
    )

print(f"{SEPARATOR} Evaluating model...")
try:
    y = model_from_fs.predict(X)
    print(f"{y = }")
except Exception as e:
    # It needs to save the model on a file called "tmp.pkl" in order to run
    print(f"FAIL: Failed to evaluate model with error {e}")
