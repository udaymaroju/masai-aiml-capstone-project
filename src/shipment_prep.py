import pandas as pd
import numpy as np

df = pd.read_csv('data/raw_shipments.csv')

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]


def clean_shipments(raw, medians=None):
    raw = raw.drop_duplicates()
    raw['service_type'] = raw['service_type'].str.strip().str.title()
    raw['payment_mode'] = raw['payment_mode'].fillna('UnKnown')
    raw['declared_value_inr'] = pd.to_numeric(
        raw['declared_value']
        .str.replace('₹', '', regex=False)
        .str.replace('\u20b9', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
        )
    raw = raw.drop(columns=['declared_value'])
    raw['dispatch_time'] = pd.to_datetime(raw['dispatch_time'])

    raw.loc[raw['parcel_weight_kg'] > 100, 'parcel_weight_kg'] = np.nan

    if medians is None:
        medians = {'parcel_weight_kg': float(raw['parcel_weight_kg'].median()),
                   'distance_km': float(raw['distance_km'].median()),
                   'declared_value_inr': float(raw['distance_km'].median())}

    raw['parcel_weight_kg'] = raw['parcel_weight_kg'].fillna(medians.get('parcel_weight_kg'))
    raw['distance_km'] = raw['distance_km'].fillna(medians.get('distance_km'))
    raw['declared_value_inr'] = raw['declared_value_inr'].fillna(medians.get('declared_value_inr'))

    return (raw, medians)

def build_features(clean, hubs):
    new = clean.merge(hubs, on='hub_id', how='left')
    new['promised_hours'] = new['service_type'].map(PROMISED_HOURS).astype(int)
    new['is_cod'] = (new['payment_mode'] == 'COD').astype(int)
    new['is_weekend'] = (new['dispatch_time'].dt.day_of_week >= 5).astype('int')
    new['dispatch_hour'] = new['dispatch_time'].dt.hour.astype(int)
    new['log_declared_value'] = np.log(new['declared_value_inr'])

    return new
    