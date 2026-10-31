import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("outputs", exist_ok=True)
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# raw checks
cleaned_raw = raw.drop_duplicates()
raw_val = pd.to_numeric(
    cleaned_raw["declared_value"].astype("string").str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False),
    errors="coerce",
)
weights_before = cleaned_raw["parcel_weight_kg"]
spellings_before = cleaned_raw["service_type"].nunique()

cleaned, med = clean_shipments(raw)
final_df = build_features(cleaned, hubs)
final_df["is_delayed"] = (final_df["delivery_hours"] > final_df["promised_hours"]).astype(int)
final_df.to_csv("data/modelling_table.csv", index=False)

# IQR checks
hours = final_df["delivery_hours"]
q1 = hours.quantile(0.25)
q3 = hours.quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5 * iqr
high = q3 + 1.5 * iqr

# value checks
val = final_df["declared_value_inr"]
log_val = final_df["log_declared_value"]

rows = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(cleaned_raw)),
    ("rows_after_dedup", len(cleaned_raw)),
    ("missing_distance_km_before", int(cleaned_raw["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(weights_before.isna().sum())),
    ("sentinel_weight_rows", int((weights_before > 100).sum())),
    ("missing_declared_value_before", int(raw_val.isna().sum())),
    ("missing_payment_mode_before", int(cleaned_raw["payment_mode"].isna().sum())),
    ("service_type_spellings_before", int(spellings_before)),
    ("service_type_spellings_after", int(final_df["service_type"].nunique())),
    ("median_distance_km", med["distance_km"]),
    ("median_parcel_weight_kg", med["parcel_weight_kg"]),
    ("median_declared_value_inr", med["declared_value_inr"]),
    ("rows_after_hub_merge", len(final_df)),
    ("nulls_in_features_and_target", int(final_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(final_df["is_delayed"].sum())),
    ("delay_rate", final_df["is_delayed"].mean()),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", low),
    ("hours_upper_bound", high),
    ("hours_outlier_count", int(((hours < low) | (hours > high)).sum())),
    ("declared_value_mean_inr", val.mean()),
    ("declared_value_median_inr", val.median()),
    ("log_declared_value_mean", log_val.mean()),
    ("log_declared_value_median", log_val.median()),
]

checks = pd.DataFrame(rows, columns=["metric", "value"])
checks["value"] = checks["value"].astype(float).round(4)
checks.to_csv("outputs/part_a_checks.csv", index=False)

# extra checks for context
final_df["distance_band"] = pd.cut(
    final_df["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
)
extra_rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    grouped = final_df.groupby(dim, observed=True).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    )
    for name, row in grouped.iterrows():
        extra_rows.append([dim, str(name), int(row["shipments"]), int(row["delayed"]), row["delay_rate"], row["mean_hours"]])

summary = pd.DataFrame(extra_rows, columns=["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"])
summary[["delay_rate", "mean_hours"]] = summary[["delay_rate", "mean_hours"]].round(4)
summary.to_csv("outputs/part_a_delay_by_group.csv", index=False)

corr = final_df.drop(columns=["distance_band"]).select_dtypes(include=[np.number]).corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv")

print("Part A done:", len(final_df), "rows; delayed =", int(final_df["is_delayed"].sum()))