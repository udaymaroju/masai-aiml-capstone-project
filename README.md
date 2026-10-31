# MASAI AI/ML Capstone Project
# Delhivery Last-Mile Delay Intelligence Pipeline
This repo holds code related to Masai AI/ML Capstone Project

## Data findings
- Service type: Express was delayed most often, with a delay rate of "0.4897" in the output table, compared with Standard at "0.2056" and Economy at "0.0857".
- Hub tier: the delay rate rose from "0.0400" at Tier 1 to "0.4400" at Tier 3, showing a sharp increase in risk as the hub tier worsened.
- Distance band: delay rate increased from "0.0219" in the 0-200 km band to "0.8523" in the 601+ km band, indicating a strong deterioration in on-time performance with longer distances.
- Delivery-hours correlation: the three strongest relationships with `delivery_hours` were `promised_hours` at "+0.8372", `distance_km` at "+0.4013", and `n_stops` at "+0.3247" when ranked by absolute correlation in `part_a_correlations.csv` after excluding `delivery_hours` itself.
- Log transform: `declared_value_inr` was log-transformed because the filled column was heavily right-skewed, with a mean of "1533.0542" and median of "1121.5000" before the transform, while the log version had a mean of "7.0277" and median of "7.0224", which is far more balanced for modeling.
