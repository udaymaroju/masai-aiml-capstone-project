import os
import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

os.makedirs("outputs", exist_ok=True)

new_raw = pd.read_csv("data/new_shipments.csv")
hubs_df = pd.read_csv("data/hubs.csv")

delay_model = joblib.load("models/delay_model.pkl")

threshold_df = pd.read_csv("models/delay_threshold.csv")
assert len(threshold_df) == 1
threshold = float(threshold_df.loc[0, "threshold"])

feature_order_df = pd.read_csv("models/feature_order.csv")
feature_order = feature_order_df["feature"].tolist()

impute_values_df = pd.read_csv("models/impute_values.csv")
medians = dict(
    zip(
        impute_values_df["column"],
        impute_values_df["median"]
    )
)

hub_segments = pd.read_csv("outputs/hub_segments.csv")

new_raw["_file_order"] = np.arange(len(new_raw))

clean_new, returned_medians = clean_shipments(
    new_raw,
    medians=medians
)

new_features = build_features(clean_new, hubs_df)

X_new = new_features[feature_order].copy()

# Positive class is is_delayed = 1.
delay_probabilities = delay_model.predict_proba(X_new)[:, 1]

# Compare full-precision probabilities to threshold.
flagged = (
    delay_probabilities >= threshold
).astype(int)

scored = new_features[
    ["shipment_id", "hub_id", "_file_order"]
].copy()

scored["delay_probability"] = delay_probabilities
scored["flagged"] = flagged

scored = scored.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    validate="many_to_one"
)

scored["action"] = np.select(
    [
        scored["flagged"].eq(0),
        scored["segment"].eq("Escalate"),
    ],
    [
        "No action",
        "Escalate to hub manager",
    ],
    default="Proactive customer call"
)

scored = scored.sort_values(
    "_file_order"
).reset_index(drop=True)

actions_df = scored[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action",
    ]
].copy()

actions_df["delay_probability"] = actions_df[
    "delay_probability"
].round(4)

actions_df.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False
)

print("New-shipment action list written to outputs/new_shipment_actions.csv")
print(actions_df)