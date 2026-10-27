import os
import pandas as pd
import numpy as np
from src.shipment_prep import clean_shipments, build_features

os.makedirs("outputs", exist_ok=True)

raw_ship_df = pd.read_csv('data/raw_shipments.csv')
raw_hubs_df = pd.read_csv('data/hubs.csv')

clean, medians = clean_shipments(raw_ship_df)
new_df = build_features(clean, raw_hubs_df)
new_df['is_delayed'] = (new_df['delivery_hours'] > new_df['promised_hours']).astype(int)

new_df.to_csv("data/modelling_table.csv", index=False)