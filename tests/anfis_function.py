import numpy as np
import pandas as pd
from fuzzyonet.fuzzyonet_from_scratch import anfis_function
from skanfis.fs import FS, GaussianFuzzySet, LinguisticVariable
from fuzzyonet.utils.scikit_anfis_helpers import fs_infer_batch


INPUT_SPACE_DIM = 2
NUM_RULES = 9
NUM_OUTPUT_FUZZY_SETS = 4

# Create a TSK fuzzy system
fs = FS()

# Define fuzzy sets and linguistic variables
T_low = GaussianFuzzySet(mu=12, sigma=3, term="low")
T_medium = GaussianFuzzySet(mu=22, sigma=4, term="medium")
T_high = GaussianFuzzySet(mu=30, sigma=3, term="high")
fs.add_linguistic_variable(
    "temperature", LinguisticVariable([T_low, T_medium, T_high]), verbose=True
)
H_low = GaussianFuzzySet(mu=30, sigma=8, term="low")
H_medium = GaussianFuzzySet(mu=55, sigma=10, term="medium")
H_high = GaussianFuzzySet(mu=80, sigma=8, term="high")
fs.add_linguistic_variable(
    "humidity", LinguisticVariable([H_low, H_medium, H_high]), verbose=True
)

# Define fuzzy rules
R1 = "IF (temperature IS low) AND (humidity IS low) THEN (fan_speed IS low)"
R2 = "IF (temperature IS low) AND (humidity IS medium) THEN (fan_speed IS low)"
R3 = "IF (temperature IS low) AND (humidity IS high) THEN (fan_speed IS medium)"

R4 = "IF (temperature IS medium) AND (humidity IS low) THEN (fan_speed IS medium)"
R5 = "IF (temperature IS medium) AND (humidity IS medium) THEN (fan_speed IS medium)"
R6 = "IF (temperature IS medium) AND (humidity IS high) THEN (fan_speed IS high)"

R7 = "IF (temperature IS high) AND (humidity IS low) THEN (fan_speed IS high)"
R8 = "IF (temperature IS high) AND (humidity IS medium) THEN (fan_speed IS very_high)"
R9 = "IF (temperature IS high) AND (humidity IS high) THEN (fan_speed IS very_high)"

fs.add_rules([R1, R2, R3, R4, R5, R6, R7, R8, R9])

# Define output functions
C_low = np.array([0.6, 0.3, 5])
C_medium = np.array([1, 0.6, 10])
C_high = np.array([1.4, 0.9, 15])
C_very_high = np.array([1.8, 1.2, 20])

assert C_low.shape == (INPUT_SPACE_DIM + 1,)
assert C_medium.shape == (INPUT_SPACE_DIM + 1,)
assert C_high.shape == (INPUT_SPACE_DIM + 1,)
assert C_very_high.shape == (INPUT_SPACE_DIM + 1,)

fs.set_output_function(
    "low", f"{C_low[0]}*temperature + {C_low[1]}*humidity + {C_low[2]}"
)
fs.set_output_function(
    "medium", f"{C_medium[0]}*temperature + {C_medium[1]}*humidity + {C_medium[2]}"
)
fs.set_output_function(
    "high", f"{C_high[0]}*temperature + {C_high[1]}*humidity + {C_high[2]}"
)
fs.set_output_function(
    "very_high",
    f"{C_very_high[0]}*temperature + {C_very_high[1]}*humidity + {C_very_high[2]}",
)


# Note: this is a matrix, because each row is a vector
linear_consequent_coefficients = np.array(
    [
        C_low,
        C_low,
        C_medium,
        C_medium,
        C_medium,
        C_high,
        C_high,
        C_very_high,
        C_very_high,
    ],
    dtype=float,
)

assert linear_consequent_coefficients.shape == (
    NUM_RULES,
    INPUT_SPACE_DIM + 1,
)

X = np.array(
    [
        [22, 40],
        [36, 81],
        [10, 68],
        [1, 33],
    ],
    dtype=float,
)

assert X.ndim == 2
assert X.shape[1] == INPUT_SPACE_DIM

X_df = pd.DataFrame(
    X,
    columns=["temperature", "humidity"],
)

y_skanfis = fs_infer_batch(fs, X_df)

# Now we calculate our estimate

# We create two matrices mu and sigma with
# columns corresponding to temperature and humidity
# rows corresponding to the  rules R1, ..., R9
mu = np.array(
    [
        [T_low._funpointer._mu, H_low._funpointer._mu],
        [T_low._funpointer._mu, H_medium._funpointer._mu],
        [T_low._funpointer._mu, H_high._funpointer._mu],
        [T_medium._funpointer._mu, H_low._funpointer._mu],
        [T_medium._funpointer._mu, H_medium._funpointer._mu],
        [T_medium._funpointer._mu, H_high._funpointer._mu],
        [T_high._funpointer._mu, H_low._funpointer._mu],
        [T_high._funpointer._mu, H_medium._funpointer._mu],
        [T_high._funpointer._mu, H_high._funpointer._mu],
    ],
    dtype=float,
)

sigma = np.array(
    [
        [T_low._funpointer._sigma, H_low._funpointer._sigma],
        [T_low._funpointer._sigma, H_medium._funpointer._sigma],
        [T_low._funpointer._sigma, H_high._funpointer._sigma],
        [T_medium._funpointer._sigma, H_low._funpointer._sigma],
        [T_medium._funpointer._sigma, H_medium._funpointer._sigma],
        [T_medium._funpointer._sigma, H_high._funpointer._sigma],
        [T_high._funpointer._sigma, H_low._funpointer._sigma],
        [T_high._funpointer._sigma, H_medium._funpointer._sigma],
        [T_high._funpointer._sigma, H_high._funpointer._sigma],
    ],
    dtype=float,
)

print(f"{sigma = }")

assert mu.shape == (NUM_RULES, INPUT_SPACE_DIM)
assert sigma.shape == (NUM_RULES, INPUT_SPACE_DIM)
assert np.all(sigma > 0)

anfis_model = anfis_function(
    mu=mu, sigma=sigma, linear_consequent_coefficients=linear_consequent_coefficients
)

y_ours = anfis_model.predict(X)[:, None]

print(f"{y_skanfis = }")
print(f"{y_ours = }")

y_mse_err = np.mean(np.power(y_skanfis - y_ours, 2))
y_infty_norm_err = np.max(np.abs(y_skanfis - y_ours))
print(f"{y_mse_err = }")
print(f"{y_infty_norm_err = }")
