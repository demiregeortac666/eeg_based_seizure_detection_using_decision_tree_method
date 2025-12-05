#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cross_validation.py

This script:
 1. Reads the feature matrix
 2. Trains models with different cross-validation strategies
 3. Evaluates model performance
 4. Saves the best model
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump

def plot_roc_curve(y_true, y_score, output_path="output/roc_curve.png"):
    """Plots the ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(output_path, dpi=150)
    print(f"ROC curve saved: {output_path}")

def plot_confusion_matrix(y_true, y_pred, output_path="output/confusion_matrix.png"):
    """Visualizes the confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['NonSeizure', 'Seizure']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Display matrix values
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path, dpi=150)
    print(f"Confusion matrix saved: {output_path}")

def plot_precision_recall(y_true, y_score, output_path="output/precision_recall.png"):
    """Plots the Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    plt.savefig(output_path, dpi=150)
    print(f"Precision-Recall curve saved: {output_path}")

def create_classifier(model_type, params=None):
    """Creates a classifier of the specified type"""
    if params is None:
        params = {}

    if model_type == 'rf':
        defaults = {'n_estimators': 100, 'random_state': 42}
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    elif model_type == 'svm':
        defaults = {'kernel': 'rbf', 'probability': True, 'random_state': 42}
        defaults.update(params)
        return SVC(**defaults)
    elif model_type == 'knn':
        defaults = {'n_neighbors': 5}
        defaults.update(params)
        return KNeighborsClassifier(**defaults)
    elif model_type == 'dt':
        defaults = {'random_state': 42}
        defaults.update(params)
        return DecisionTreeClassifier(**defaults)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def prepare_data(df, target_col='label', test_size=0.2, normalize=True, handle_nan='fill_zero'):
    """Splits data into training and test sets"""
    # Identify categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Target variable must be numeric
    if target_col in cat_cols:
        # If categorical, convert to numeric (NonSeizure=0, Seizure=1)
        label_map = {'NonSeizure': 0, 'Seizure': 1} if df[target_col].isin(['NonSeizure', 'Seizure']).all() else None
        if label_map:
            df[target_col] = df[target_col].map(label_map)
            print(f"Column '{target_col}' converted from categorical to numeric.")
        else:
            raise ValueError(f"Column '{target_col}' could not be converted to numeric.")

    # Create feature and target matrices
    exclude_cols = [col for col in cat_cols if col != target_col]
    feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]

    if not feature_cols:
        raise ValueError("No numeric features found!")

    # Handle NaN values
    if handle_nan == 'fill_zero':
        print(f"Filling NaN values with 0. Number of NaN values: {df[feature_cols].isna().sum().sum()}")
        df[feature_cols] = df[feature_cols].fillna(0)
    elif handle_nan == 'drop_rows':
        nan_count_before = df.shape[0]
        df = df.dropna(subset=feature_cols)
        print(f"Rows containing NaN values removed. Number of removed rows: {nan_count_before - df.shape[0]}")
    elif handle_nan == 'drop_cols':
        nan_cols = [col for col in feature_cols if df[col].isna().any()]
        if nan_cols:
            print(f"{len(nan_cols)} columns containing NaN values removed.")
            feature_cols = [col for col in feature_cols if col not in nan_cols]
            if not feature_cols:
                raise ValueError("All features contain NaN values! Cannot proceed.")

    X = df[feature_cols].values
    y = df[target_col].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)

    # Standardization
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save the scaler
        with open("output/cv_scaler.pkl", 'wb') as f:
            pickle.dump((scaler, feature_cols), f)
        print("Scaler saved: output/cv_scaler.pkl")

    return X_train, X_test, y_train, y_test, feature_cols

def k_fold_cv(X, y, model_type, k=5, params=None):
    """Model evaluation with K-fold cross validation"""
    clf = create_classifier(model_type, params)

    # Use StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Cross validation scores
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring='f1')

    print(f"\n{k}-fold Cross Validation Results ({model_type}):")
    print(f"  F1 Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return cv_scores.mean()

def grid_search_cv(X, y, model_type, param_grid, k=5):
    """Hyperparameter optimization with Grid Search"""
    base_clf = create_classifier(model_type)

    # Use StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Grid Search
    grid_search = GridSearchCV(
        base_clf, param_grid, cv=skf,
        scoring='f1', n_jobs=-1, verbose=1
    )

    grid_search.fit(X, y)

    print(f"\nGrid Search Results ({model_type}):")
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best F1 score: {grid_search.best_score_:.3f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def evaluate_model(clf, X_test, y_test, model_name, output_dir="output"):
    """Evaluates the model on test data"""
    # Predictions
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)

    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nModel Evaluation ({model_name}):")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['NonSeizure', 'Seizure']))

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred,
                          output_path=f"{output_dir}/confusion_matrix_{model_name}.png")

    # ROC curve
    plot_roc_curve(y_test, y_scores,
                  output_path=f"{output_dir}/roc_curve_{model_name}.png")

    # Precision-Recall curve
    plot_precision_recall(y_test, y_scores,
                         output_path=f"{output_dir}/precision_recall_{model_name}.png")

    # Store results in a dictionary
    results = {
        'accuracy': accuracy,
        'f1': f1,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred,
                                                     target_names=['NonSeizure', 'Seizure'],
                                                     output_dict=True)
    }

    return results

def save_model(clf, feature_cols, model_name, output_dir="output"):
    """Saves the model to disk"""
    model_path = f"{output_dir}/model_{model_name}.joblib"
    dump(clf, model_path)

    # Save model feature names and metadata
    meta_path = f"{output_dir}/model_{model_name}_meta.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump({
            'feature_cols': feature_cols,
            'model_type': model_name,
            'creation_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f)

    print(f"Model saved: {model_path}")
    print(f"Model metadata saved: {meta_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluates EEG classification models with Cross-Validation")
    parser.add_argument("--input", type=str,
                        default="output/normalized_features.csv",
                        help="Input CSV file (default: output/normalized_features.csv)")
    parser.add_argument("--output-dir", type=str,
                        default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--model", type=str,
                        choices=['rf', 'svm', 'knn', 'dt'],
                        default='rf',
                        help="Model type (default: rf)")
    parser.add_argument("--k-fold", type=int, default=5,
                        help="K value for K-fold CV (default: 5)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set ratio (default: 0.2)")
    parser.add_argument("--grid-search", action="store_true",
                        help="Perform hyperparameter optimization with Grid Search")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize",
                        help="Do not normalize features")
    parser.add_argument("--nan-handling", type=str,
                        choices=['fill_zero', 'drop_rows', 'drop_cols'],
                        default='fill_zero',
                        help="Method for handling NaN values (default: fill_zero)")

    args = parser.parse_args()

    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1

    # Check output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input)

    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    try:
        # Prepare data
        X_train, X_test, y_train, y_test, feature_cols = prepare_data(
            df, test_size=args.test_size, normalize=args.normalize, handle_nan=args.nan_handling)

        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

        # Model parameters
        param_grids = {
            'rf': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            },
            'dt': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }

        # Cross-validation
        mean_cv_score = k_fold_cv(X_train, y_train, args.model, k=args.k_fold)

        if args.grid_search:
            # Parameter optimization with Grid Search
            best_clf, best_params, best_score = grid_search_cv(
                X_train, y_train, args.model, param_grids[args.model], k=args.k_fold)

            # Evaluate with the best model
            print("\nTesting model with best parameters...")
            clf = best_clf
        else:
            # Normal model training
            print("\nTraining model with default parameters...")
            clf = create_classifier(args.model)
            clf.fit(X_train, y_train)

        # Evaluate on test set
        results = evaluate_model(clf, X_test, y_test, args.model, output_dir=args.output_dir)

        # Save the model
        save_model(clf, feature_cols, args.model, output_dir=args.output_dir)

        # Save performance results
        results_path = f"{args.output_dir}/model_{args.model}_results.json"
        pd.DataFrame([results]).to_json(results_path, orient='records')
        print(f"Evaluation results saved: {results_path}")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
