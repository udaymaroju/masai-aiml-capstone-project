import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.shipment_prep import FEATURES
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_csv("data/modelling_table.csv")
x, y = data[FEATURES], data["delivery_hours"]
metrics = []

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

train_scaled_max_abs_mean = float(np.abs(x_train_scaled.mean(axis = 0)).max())
test_scaled_max_abs_mean = float(np.abs(x_test_scaled.mean(axis = 0)).max())

checks = pd.DataFrame({
    "metric": ['train_rows', 'val_rows', 'test_rows', 'train_scaled_max_abs_mean', 'test_scaled_max_abs_mean'],
    "value": [len(x_train), len(x_val), len(x_test), round(train_scaled_max_abs_mean, 4), round(test_scaled_max_abs_mean, 4)]
})

checks.to_csv("outputs/part_b_checks.csv", index=False)

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

baseline_pred = np.full(len(y_val), y_train.mean())
baseline_result = regression_metrics(y_val, baseline_pred)

metrics.append({
    "model": "mean_baseline",
    "split": "val",
    **baseline_result
})

# Tree-depth diagnostics

depths = [1, 2, 3, 4, 6, 8, 12, None]
tree_depth_results = []

for d in depths:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42)
    tree.fit(x_train, y_train)
    tree_depth_results.append({"max_depth": "unlimited" if d is None else d, 
                               'train_rmse': np.sqrt(mean_squared_error(y_train, tree.predict(x_train))),
                               'val_rmse': np.sqrt(mean_squared_error(y_val, tree.predict(x_val)))})

tree_depth_results_df = pd.DataFrame(tree_depth_results)
tree_depth_results_df.to_csv("outputs/part_b_tree_depth.csv", index=False)

best_tree_row = tree_depth_results_df.loc[tree_depth_results_df["val_rmse"].idxmin()]
best_tree_depth_label = best_tree_row["max_depth"]

if best_tree_depth_label == "unlimited":
    best_tree_depth = None
else:
    best_tree_depth = int(best_tree_depth_label)

# 4 Candidate Models

models = {'linear_regression': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
          'decision_tree': DecisionTreeRegressor(max_depth=best_tree_depth, random_state=42),
          'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
          'gradient_boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)}


for model_name, model in models.items():
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    result = regression_metrics(y_val, y_val_pred)
    metrics.append({"model": model_name, "split": "val", **result})

metrics_df = pd.DataFrame(metrics, columns=["model", "split", "MAE", "MSE", "RMSE", "R2"])
# metrics_df.round(4).to_csv("outputs/part_b_metrics.csv", index=False)

# Learning Rate experiment for Gradient Boosting

learning_rates = [0.01, 0.1, 1.0]
gb_learning_rate_results = []

for rate in learning_rates:
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=rate, random_state=42)
    gb_model.fit(x_train, y_train)
    y_val_pred = gb_model.predict(x_val)
    result = regression_metrics(y_val, y_val_pred)
    gb_learning_rate_results.append({"learning_rate": rate, 
                                     "val_rmse": result.get("RMSE"), 
                                     "val_r2": result.get("R2")})

gb_learning_rate_results_df = pd.DataFrame(gb_learning_rate_results, columns=["learning_rate", "val_rmse", "val_r2"])
gb_learning_rate_results_df.round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False) 


# Cross-Validation Comparision and final model selection

candidate_metrics = metrics_df[metrics_df['model'] != 'mean_baseline'].copy()
candidate_metrics.sort_values(by='RMSE', ascending=True, inplace=True)

top_two_models = candidate_metrics.head(2)['model'].tolist()
final_model_name = candidate_metrics.iloc[0]['model']

x_train_val = pd.concat([x_train, x_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_metrics = []

for model in top_two_models:
    model_instance = models[model]
    cv_scores = cross_val_score(estimator=model_instance, X=x_train_val, y=y_train_val, cv=cv)
    cv_metrics.append({"model": model, "mean_r2": float(np.mean(cv_scores)), "std_r2": float(np.std(cv_scores))})

cv_metrics_df = pd.DataFrame(cv_metrics, columns=["model", "mean_r2", "std_r2"])
cv_metrics_df.round(4).to_csv("outputs/part_b_cv.csv", index=False)

final_model = models[final_model_name]

final_model.fit(x_train_val, y_train_val)
pred = final_model.predict(x_test)

final_result = regression_metrics(y_test, pred)

final_row = {"model": final_model_name, "split": "test", **final_result}

final_metrics_df = pd.concat([metrics_df, pd.DataFrame([final_row])], ignore_index=True)
final_metrics_df.round(4).to_csv("outputs/part_b_metrics.csv", index=False)

