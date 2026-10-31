import pandas as pd
from sklearn.model_selection import train_test_split
from src.shipment_prep import FEATURES
from sklearn.preprocessing import StandardScaler
import numpy as np


data = pd.read_csv("data/modelling_table.csv")
x, y = data[FEATURES], data["delivery_hours"]

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

train_scaled_max_abs_mean = float(np.abs(x_train_scaled).mean(axis = 0)).max()
test_scaled_max_abs_mean = float(np.abs(x_test_scaled).mean(axis = 0)).max()

checks = pd.DataFrame({
    "metric": ['train_rows', 'val_rows', 'test_rows', 'train_scaled_max_abs_mean', 'test_scaled_max_abs_mean'],
    "value": [len(x_train), len(x_val), len(x_test), round(train_scaled_max_abs_mean, 4), round(test_scaled_max_abs_mean, 4)]
})

checks.to_csv("outputs/part_b_checks.csv", index=False)

