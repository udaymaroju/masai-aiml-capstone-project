import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

config_df = pd.read_csv("outputs/part_c_final_config.csv")
config = config_df.iloc[0]

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

# Convert Part C display value "unlimited" back to sklearn's None.
if str(config["max_depth"]) == "unlimited":
    max_depth = None
else:
    max_depth = int(config["max_depth"])

assert selector_k == 7

model_df = pd.read_csv("data/modelling_table.csv")

x = model_df[FEATURES].copy()
y = model_df["is_delayed"].copy()

x_train, x_temp, y_train, y_temp = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

final_pipeline = Pipeline([
    (
        "selector",
        SelectKBest(
            score_func=f_classif,
            k=selector_k
        )
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    )
])

# Fit on Part C training rows only.
final_pipeline.fit(x_train, y_train)

joblib.dump(
    final_pipeline,
    "models/delay_model.pkl"
)

threshold_df = pd.DataFrame({
    "threshold": [threshold]
})

threshold_df.round(4).to_csv(
    "models/delay_threshold.csv",
    index=False
)

feature_order_df = pd.DataFrame({
    "feature": FEATURES
})

feature_order_df.to_csv(
    "models/feature_order.csv",
    index=False
)

raw_history = pd.read_csv("data/raw_shipments.csv")

# Uses the shared Part A cleaning function.
# It returns a cleaned table and the historical median dictionary.
_, medians = clean_shipments(raw_history)

impute_values_df = pd.DataFrame(
    [
        {
            "column": column,
            "median": value
        }
        for column, value in medians.items()
    ]
)

# Required medians: distance, weight, declared value.
assert set(impute_values_df["column"]) == {
    "distance_km",
    "parcel_weight_kg",
    "declared_value_inr",
}

impute_values_df.round(4).to_csv(
    "models/impute_values.csv",
    index=False
)

reloaded_pipeline = joblib.load(
    "models/delay_model.pkl"
)

in_memory_probabilities = final_pipeline.predict_proba(
    x_test
)[:, 1]

reloaded_probabilities = reloaded_pipeline.predict_proba(
    x_test
)[:, 1]

reload_max_abs_probability_diff = float(
    np.max(
        np.abs(
            in_memory_probabilities - reloaded_probabilities
        )
    )
)

reload_check_df = pd.DataFrame([
    {
        "metric": "test_rows_scored",
        "value": len(x_test)
    },
    {
        "metric": "reload_max_abs_probability_diff",
        "value": reload_max_abs_probability_diff
    },
    {
        "metric": "threshold",
        "value": threshold
    }
])

reload_check_df.round(4).to_csv(
    "outputs/part_d_reload_check.csv",
    index=False
)

print("Model package created successfully.")
print("Saved: models/delay_model.pkl")
print("Saved: models/delay_threshold.csv")
print("Saved: models/feature_order.csv")
print("Saved: models/impute_values.csv")
print("\nReload check:")
print(reload_check_df.round(4))