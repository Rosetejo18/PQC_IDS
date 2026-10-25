
# ============================================================
# CIC-PQC_OAV v1
# DECISION TREE CLASSIFIER
#
# Uses the PREPROCESSED datasets generated previously:
#
# X_train_preprocessed.csv
# X_val_preprocessed.csv
# X_test_preprocessed.csv
# y_train.csv
# y_val.csv
# y_test.csv
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score
)

import matplotlib.pyplot as plt


# ============================================================
# 2. LOAD PREPROCESSED DATA
# ============================================================

X_train = pd.read_csv("X_train_preprocessed.csv")
X_val   = pd.read_csv("X_val_preprocessed.csv")
X_test  = pd.read_csv("X_test_preprocessed.csv")

y_train = pd.read_csv("y_train.csv").squeeze("columns")
y_val   = pd.read_csv("y_val.csv").squeeze("columns")
y_test  = pd.read_csv("y_test.csv").squeeze("columns")


print("=" * 70)
print("DATA LOADED")
print("=" * 70)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# 3. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print("\nTraining:")
print(y_train.value_counts().sort_index())

print("\nValidation:")
print(y_val.value_counts().sort_index())

print("\nTest:")
print(y_test.value_counts().sort_index())


# ============================================================
# 4. DEFINE METRIC FUNCTION
# ============================================================

def evaluate_model(model, X, y, dataset_name):
    """
    Evaluate a trained binary classification model.
    """

    predictions = model.predict(X)

    # Probability for positive class
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = None

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    if probabilities is not None:
        roc_auc = roc_auc_score(
            y,
            probabilities
        )

        pr_auc = average_precision_score(
            y,
            probabilities
        )
    else:
        roc_auc = np.nan
        pr_auc = np.nan


    print("\n" + "=" * 70)
    print(dataset_name.upper())
    print("=" * 70)

    print(f"Accuracy  : {accuracy:.6f}")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1-score  : {f1:.6f}")
    print(f"ROC-AUC   : {roc_auc:.6f}")
    print(f"PR-AUC    : {pr_auc:.6f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["Normal", "Anomaly"],
            digits=4,
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


# ============================================================
# 5. BASELINE DECISION TREE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE DECISION TREE")
print("=" * 70)

start_time = time.time()

dt_baseline = DecisionTreeClassifier(
    random_state=42
)

dt_baseline.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print(f"Training time: {training_time:.4f} seconds")

# Evaluate
baseline_train_metrics = evaluate_model(
    dt_baseline,
    X_train,
    y_train,
    "Training Set"
)

baseline_val_metrics = evaluate_model(
    dt_baseline,
    X_val,
    y_val,
    "Validation Set"
)


# ============================================================
# 6. DECISION TREE HYPERPARAMETER EXPERIMENT
# ============================================================
#
# The paper performs comparative ML experiments.
#
# We evaluate several Decision Tree configurations using the
# VALIDATION SET.
#
# The TEST SET is deliberately NOT used here.
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE HYPERPARAMETER EXPERIMENT")
print("=" * 70)


parameter_grid = [

    # Default-like tree
    {
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    # Shallow trees
    {
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    # Larger minimum split
    {
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 1
    },

    {
        "max_depth": None,
        "min_samples_split": 10,
        "min_samples_leaf": 1
    },

    # Minimum leaf size
    {
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 5
    },

    {
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 10
    },

    # More regularized
    {
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 5
    },

    {
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 5
    },

    {
        "max_depth": 20,
        "min_samples_split": 10,
        "min_samples_leaf": 5
    }
]


results = []


for i, params in enumerate(parameter_grid, start=1):

    print(
        f"\nExperiment {i}/{len(parameter_grid)}"
    )

    print(params)

    start_time = time.time()

    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=42,
        **params
    )

    model.fit(
        X_train,
        y_train
    )

    train_time = time.time() - start_time

    # Validation predictions
    val_predictions = model.predict(X_val)

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        val_probabilities
    )

    pr_auc = average_precision_score(
        y_val,
        val_probabilities
    )

    results.append({
        "experiment": i,
        "max_depth": params["max_depth"],
        "min_samples_split": params["min_samples_split"],
        "min_samples_leaf": params["min_samples_leaf"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "training_time_sec": train_time
    })


# ============================================================
# 7. DISPLAY ALL EXPERIMENTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("ALL DECISION TREE EXPERIMENTS")
print("=" * 70)

display(
    results_df.sort_values(
        "f1",
        ascending=False
    ).reset_index(drop=True)
)


# ============================================================
# 8. SELECT BEST CONFIGURATION
# ============================================================

best_index = results_df["f1"].idxmax()

best_max_depth = results_df.loc[best_index, "max_depth"]

# Convert numpy/pandas float such as np.float64(10.0)
# back to a Python int, while preserving None
if pd.isna(best_max_depth):
    best_max_depth = None
else:
    best_max_depth = int(best_max_depth)

best_params = {
    "max_depth": best_max_depth,
    "min_samples_split": int(
        results_df.loc[best_index, "min_samples_split"]
    ),
    "min_samples_leaf": int(
        results_df.loc[best_index, "min_samples_leaf"]
    )
}

print("\n" + "=" * 70)
print("BEST DECISION TREE CONFIGURATION")
print("=" * 70)

print(results_df.loc[best_index])

print("\nSelected parameters:")
print(best_params)

print("\nParameter types:")
for key, value in best_params.items():
    print(
        f"{key}: {value!r} "
        f"({type(value).__name__})"
    )


# ============================================================
# 9. RETRAIN BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING BEST DECISION TREE")
print("=" * 70)

start_time = time.time()

best_dt = DecisionTreeClassifier(
    criterion="gini",
    random_state=42,
    **best_params
)

best_dt.fit(
    X_train,
    y_train
)

best_training_time = time.time() - start_time

print(
    f"Training time: "
    f"{best_training_time:.4f} seconds"
)


# ============================================================
# 10. EVALUATE BEST MODEL ON TRAINING SET
# ============================================================

best_train_metrics = evaluate_model(
    best_dt,
    X_train,
    y_train,
    "Best Model - Training Set"
)


# ============================================================
# 11. EVALUATE BEST MODEL ON VALIDATION SET
# ============================================================

best_val_metrics = evaluate_model(
    best_dt,
    X_val,
    y_val,
    "Best Model - Validation Set"
)


# ============================================================
# 12. FINAL TEST EVALUATION
# ============================================================
#
# IMPORTANT:
# The test set is used ONLY after selecting the model.
# ============================================================

best_test_metrics = evaluate_model(
    best_dt,
    X_test,
    y_test,
    "Best Model - TEST SET"
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("TEST SET CONFUSION MATRIX")
print("=" * 70)

y_test_pred = best_dt.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print(cm)

print("\nInterpretation:")
print("TN =", cm[0, 0])
print("FP =", cm[0, 1])
print("FN =", cm[1, 0])
print("TP =", cm[1, 1])


# Plot
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"]
)

disp.plot()

plt.title(
    "Decision Tree - CIC-PQC_OAV v1 Test Set"
)

plt.tight_layout()
plt.show()


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": best_dt.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

display(feature_importance)


# ============================================================
# 15. PLOT TOP 15 FEATURES
# ============================================================

top_n = 15

top_features = feature_importance.head(top_n)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.title(
    "Top 15 Decision Tree Feature Importances"
)

plt.tight_layout()
plt.show()


# ============================================================
# 16. FINAL RESULTS TABLE
# ============================================================

final_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "PR-AUC",
        "Training Time (sec)"
    ],

    "Training": [
        best_train_metrics["accuracy"],
        best_train_metrics["precision"],
        best_train_metrics["recall"],
        best_train_metrics["f1"],
        best_train_metrics["roc_auc"],
        best_train_metrics["pr_auc"],
        best_training_time
    ],

    "Validation": [
        best_val_metrics["accuracy"],
        best_val_metrics["precision"],
        best_val_metrics["recall"],
        best_val_metrics["f1"],
        best_val_metrics["roc_auc"],
        best_val_metrics["pr_auc"],
        np.nan
    ],

    "Test": [
        best_test_metrics["accuracy"],
        best_test_metrics["precision"],
        best_test_metrics["recall"],
        best_test_metrics["f1"],
        best_test_metrics["roc_auc"],
        best_test_metrics["pr_auc"],
        np.nan
    ]
})


print("\n" + "=" * 70)
print("FINAL DECISION TREE RESULTS")
print("=" * 70)

display(
    final_results.round(6)
)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "decision_tree_hyperparameter_results.csv",
    index=False
)

final_results.to_csv(
    "decision_tree_final_results.csv",
    index=False
)

feature_importance.to_csv(
    "decision_tree_feature_importance.csv",
    index=False
)


print("\nResults saved:")
print("1. decision_tree_hyperparameter_results.csv")
print("2. decision_tree_final_results.csv")
print("3. decision_tree_feature_importance.csv")


# ============================================================
# 18. FINAL MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL SUMMARY")
print("=" * 70)

print("Algorithm       : Decision Tree")
print("Criterion       : Gini")
print("Random state    : 42")

print(
    f"Max depth       : "
    f"{best_params['max_depth']}"
)

print(
    f"Min samples split: "
    f"{best_params['min_samples_split']}"
)

print(
    f"Min samples leaf : "
    f"{best_params['min_samples_leaf']}"
)

print(
    f"\nTest Accuracy   : "
    f"{best_test_metrics['accuracy']:.6f}"
)

print(
    f"Test Precision  : "
    f"{best_test_metrics['precision']:.6f}"
)

print(
    f"Test Recall     : "
    f"{best_test_metrics['recall']:.6f}"
)

print(
    f"Test F1-score   : "
    f"{best_test_metrics['f1']:.6f}"
)

print(
    f"Test ROC-AUC    : "
    f"{best_test_metrics['roc_auc']:.6f}"
)

print(
    f"Test PR-AUC     : "
    f"{best_test_metrics['pr_auc']:.6f}"
)
