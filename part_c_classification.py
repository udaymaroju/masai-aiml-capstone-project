import numpy as np
import pandas as pd
import ast
import os
from src.shipment_prep import FEATURES, NOISE_COLUMNS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

os.makedirs("outputs", exist_ok=True)

data = pd.read_csv("data/modelling_table.csv")

x = data[FEATURES].copy()
y = data["is_delayed"].copy()

model_rows = []

# 1 Stratified Split

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 2 Blind baseline and 3 classifiers

def classification_metrics(y_true, y_pred):
    # Assumes 0 = on time and 1 = delayed.

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }

# 2.1 Blind Prediction

blind_predictions = np.zeros(len(y_val), dtype=int)

blind_result = classification_metrics(
    y_true=y_val,
    y_pred=blind_predictions
)

model_rows.append({"model": "blind_all_on_time", **blind_result})

# 2.2 Logistic Regression

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

logistic_model.fit(x_train, y_train)
logistic_pred = logistic_model.predict(x_val)
logistic_result = classification_metrics(
    y_true=y_val,
    y_pred=logistic_pred
)

model_rows.append({"model": "logistic_regression", **logistic_result})

# 2.3 Decision Tree Classifier

tree_depths = [2, 3, 4, 5, 6, 8, 10]

best_tree_model = None
best_tree_depth = None
best_tree_f1 = -1.0

for depth in tree_depths:
    tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree_model.fit(x_train, y_train)

    tree_predictions = tree_model.predict(x_val)

    current_f1 = float(
        f1_score(y_val, tree_predictions, zero_division=0)
    )

    # The list is ordered from smaller to larger depth.
    # Using > rather than >= preserves the smaller depth on an F1 tie.
    if current_f1 > best_tree_f1:
        best_tree_f1 = current_f1
        best_tree_depth = depth
        best_tree_model = tree_model

best_tree_predictions = best_tree_model.predict(x_val)

best_tree_result = classification_metrics(
    y_true=y_val,
    y_pred=best_tree_predictions
)

model_rows.append({
    "model": f"decision_tree_depth_{best_tree_depth}",
    **best_tree_result
})

# 2.4 Random Forest Classifier

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(x_train, y_train)

random_forest_predictions = random_forest_model.predict(x_val)

random_forest_result = classification_metrics(
    y_true=y_val,
    y_pred=random_forest_predictions
)

model_rows.append({
    "model": "random_forest_default",
    **random_forest_result
})

models_df = pd.DataFrame(
    model_rows,
    columns=["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]
    )

models_df.round(4).to_csv("outputs/part_c_models.csv",index=False)

# 3 Columns carry signal

rf_all_features = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_all_features.fit(x_train, y_train)

rf_importances = rf_all_features.feature_importances_

selector_all = SelectKBest(
    score_func=f_classif,
    k="all"
)

selector_all.fit(x_train, y_train)
f_scores = selector_all.scores_

feature_scores_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": [float(score) for score in f_scores],
    "rf_importance": [float(importance) for importance in rf_importances]
})

feature_scores_df["f_rank"] = (
    feature_scores_df["f_score"]
    .rank(ascending=False, method="first")
    .astype(int)
)

feature_scores_df["is_noise"] = (
    feature_scores_df["feature"].isin(NOISE_COLUMNS)
)

feature_scores_df = feature_scores_df[
    [
        "feature",
        "f_score",
        "f_rank",
        "rf_importance",
        "is_noise"
    ]
]

feature_scores_df.round(4).to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False
)  

k_values = [5, 7, 10, 12, 17]

selectk_rows = []

for k in k_values:
    # A pipeline ensures the selector is fitted only on training rows.
    selectk_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    # Fit selector and forest only on training data.
    selectk_pipeline.fit(x_train, y_train)

    # Generate default-threshold (0.5) predictions on validation data.
    val_predictions = selectk_pipeline.predict(x_val)

    # Record validation F1 score.
    val_f1 = float(
        f1_score(y_val, val_predictions, zero_division=0)
    )

    selectk_rows.append({
        "k": k,
        "val_f1": val_f1
    })

selectk_df = pd.DataFrame(
    selectk_rows,
    columns=["k", "val_f1"]
)

selectk_df.round(4).to_csv(
    "outputs/part_c_selectk.csv",
    index=False
)

# 4 Tuning

k = 7

selector = SelectKBest(score_func=f_classif, k=k)
X_train_7 = selector.fit_transform(x_train, y_train)
X_val_7 = selector.transform(x_val)

selected_features = np.array(x_train.columns)[selector.get_support()]

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

rf_base = RandomForestClassifier(
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5
)

grid_search.fit(X_train_7, y_train)
grid_val_predictions = grid_search.best_estimator_.predict(X_val_7)

grid_val_f1 = f1_score(
    y_val,
    grid_val_predictions,
    zero_division=0
)

randomized_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)
randomized_search.fit(X_train_7, y_train)
randomized_val_predictions = randomized_search.best_estimator_.predict(
    X_val_7
)

randomized_val_f1 = f1_score(
    y_val,
    randomized_val_predictions,
    zero_division=0
)

default_rf_7 = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_rf_7.fit(X_train_7, y_train)

default_val_predictions = default_rf_7.predict(X_val_7)

default_val_f1 = f1_score(
    y_val,
    default_val_predictions,
    zero_division=0
)

tuning_rows = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": float(grid_search.best_score_),
        "val_f1": grid_val_f1
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(randomized_search.best_params_),
        "best_cv_accuracy": float(randomized_search.best_score_),
        "val_f1": randomized_val_f1
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": (
            "{'n_estimators': 100, 'max_depth': None, "
            "'min_samples_leaf': 1}"
        ),
        "best_cv_accuracy": np.nan,
        "val_f1": default_val_f1
    }
]

tuning_df = pd.DataFrame(
    tuning_rows,
    columns=[
        "search",
        "candidates",
        "cv_folds",
        "total_fits",
        "best_params",
        "best_cv_accuracy",
        "val_f1"
    ]
)

tuning_df.round(4).to_csv("outputs/part_c_tuning.csv", index=False)

tie_priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2
}

tuning_for_selection = tuning_df.copy()

tuning_for_selection["tie_priority"] = (
    tuning_for_selection["search"].map(tie_priority)
)

tuning_for_selection = tuning_for_selection.sort_values(
    by=["val_f1", "tie_priority"],
    ascending=[False, True]
).reset_index(drop=True)

winning_row = tuning_for_selection.iloc[0]
winning_val_f1 = float(winning_row["val_f1"])

chosen_from = winning_row["search"]
chosen_params = ast.literal_eval(winning_row["best_params"])

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42,
                                     **chosen_params))])

final_model.fit(x_train, y_train)

val_probabilities = final_model.predict_proba(x_val)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

threshold_rows = []

for threshold in thresholds:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    threshold_metrics = classification_metrics(
        y_true=y_val,
        y_pred=val_predictions
    )

    threshold_rows.append({
        "threshold": threshold,
        **threshold_metrics
    })

thresholds_df = pd.DataFrame(
    threshold_rows,
    columns=[
        "threshold",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]
)

thresholds_df.round(4).to_csv(
    "outputs/part_c_thresholds.csv",
    index=False
)

eligible_thresholds = thresholds_df.loc[
    thresholds_df["recall"] >= 0.80
]

if len(eligible_thresholds) > 0:
    chosen_threshold = float(
        eligible_thresholds["threshold"].max()
    )
else:
    chosen_threshold = 0.1


chosen_threshold_row = thresholds_df.loc[
    thresholds_df["threshold"] == chosen_threshold
].iloc[0]


test_probabilities = final_model.predict_proba(x_test)[:, 1]
test_predictions = (test_probabilities >= chosen_threshold).astype(int)

test_metrics = classification_metrics(
    y_true=y_test,
    y_pred=test_predictions
)

final_test_df = pd.DataFrame(
    [{
        "threshold": float(chosen_threshold),
        **test_metrics
    }],
    columns=[
        "threshold",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]
)

final_test_df.round(4).to_csv(
    "outputs/part_c_final_test.csv",
    index=False
)

max_depth_for_csv = (
    "unlimited"
    if chosen_params["max_depth"] is None
    else chosen_params["max_depth"]
)

final_config_df = pd.DataFrame(
    [{
        "selector_k": 7,
        "n_estimators": int(chosen_params["n_estimators"]),
        "max_depth": max_depth_for_csv,
        "min_samples_leaf": int(chosen_params["min_samples_leaf"]),
        "threshold": float(chosen_threshold),
        "chosen_from": chosen_from
    }],
    columns=[
        "selector_k",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "threshold",
        "chosen_from"
    ]
)

final_config_df.round(4).to_csv(
    "outputs/part_c_final_config.csv",
    index=False
)