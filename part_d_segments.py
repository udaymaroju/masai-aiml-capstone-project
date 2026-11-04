import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)

model_df = pd.read_csv("data/modelling_table.csv")
hubs_df = pd.read_csv("data/hubs.csv")

# Hub Profile
hub_profile = (
    model_df
    .groupby("hub_id", as_index=False)
    .agg(
        shipments=("hub_id", "size"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
)

hub_profile = hub_profile.sort_values("hub_id", ascending=True).reset_index(drop=True)
hub_profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# K-Means segmentation

profile_columns = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]

X_hubs = hub_profile[profile_columns].copy()
scaler = StandardScaler()
X_hubs_scaled = scaler.fit_transform(X_hubs)
kmeans_3 = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
hub_profile["cluster"] = kmeans_3.fit_predict(X_hubs_scaled)

# Separation Evidence

def across_within_ratio(scaled_matrix, labels):
    """
    Calculate mean distance across clusters divided by mean distance
    within clusters, using all unordered hub pairs.
    """
    within_distances = []
    across_distances = []

    n_hubs = len(scaled_matrix)

    for i in range(n_hubs):
        for j in range(i + 1, n_hubs):
            distance = np.linalg.norm(
                scaled_matrix[i] - scaled_matrix[j]
            )

            if labels[i] == labels[j]:
                within_distances.append(distance)
            else:
                across_distances.append(distance)

    return (
        float(np.mean(across_distances))
        / float(np.mean(within_distances))
    )

k_values = [2, 3, 4, 5]
separation_rows = []

for k in k_values:
    # Same scaled matrix, same hub order, same seed and n_init
    # for every requested K.
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_hubs_scaled)

    # Calculate separation on scaled feature space.
    ratio = across_within_ratio(
        scaled_matrix=X_hubs_scaled,
        labels=labels
    )

    # Count hubs in each cluster and format descending sizes.
    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
        .sort_values(ascending=False)
        .tolist()
    )

    cluster_sizes_text = "-".join(
        str(size)
        for size in cluster_sizes
    )

    separation_rows.append({
        "K": k,
        "across_over_within": ratio,
        "cluster_sizes": cluster_sizes_text
    })


# Required separation-evidence table.
separation_df = pd.DataFrame(
    separation_rows,
    columns=[
        "K",
        "across_over_within",
        "cluster_sizes"
    ]
)

separation_df.round(4).to_csv("outputs/part_d_separation.csv", index=False)

# Segment names and tier check

required_hub_columns = [
    "hub_id",
    "hub_city",
    "hub_tier",
]

missing_hub_columns = [
    column
    for column in required_hub_columns
    if column not in hubs_df.columns
]

if missing_hub_columns:
    raise ValueError(
        f"data/hubs.csv is missing required columns: "
        f"{missing_hub_columns}"
    )

cluster_delay_rates = (
    hub_profile
    .groupby("cluster", as_index=False)["delay_rate"]
    .mean()
    .sort_values("delay_rate", ascending=True)
    .reset_index(drop=True)
)

# Lowest delay rate = Standard, middle = Watch, highest = Escalate.
segment_order = ["Standard", "Watch", "Escalate"]

cluster_to_segment = dict(
    zip(
        cluster_delay_rates["cluster"],
        segment_order
    )
)

hub_profile["segment"] = hub_profile["cluster"].map(
    cluster_to_segment
)

segment_profile = (
    hub_profile
    .groupby("segment", as_index=False)
    .agg(
        hubs=("hub_id", "size"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
)

# Ensure readable business order rather than alphabetical order.
segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories=segment_order,
    ordered=True
)

segment_profile = segment_profile.sort_values(
    "segment"
).reset_index(drop=True)

segment_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

hub_segments = (
    hub_profile[
        ["hub_id", "segment"]
    ]
    .merge(
        hubs_df[
            ["hub_id", "hub_city", "hub_tier"]
        ],
        on="hub_id",
        how="left",
        validate="one_to_one"
    )
)

hub_segments = hub_segments[
    ["hub_id", "segment", "hub_city", "hub_tier"]
].sort_values(
    "hub_id"
).reset_index(drop=True)

hub_segments.to_csv("outputs/hub_segments.csv", index=False)

tier_to_expected_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate",
}

hub_segments["expected_segment"] = hub_segments[
    "hub_tier"
].map(tier_to_expected_segment)

tier_vs_segment_mismatch = hub_segments.loc[
    hub_segments["segment"]
    != hub_segments["expected_segment"],
    ["hub_id", "segment", "hub_city", "hub_tier"]
].copy()

tier_vs_segment_mismatch = tier_vs_segment_mismatch.sort_values(
    "hub_id"
).reset_index(drop=True)

tier_vs_segment_mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)