from __future__ import annotations

from itertools import combinations
import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .features import compute_mechanism_summary_features

ELASTIC_NET_C_VALUES: tuple[float, ...] = (0.3, 1.0, 3.0)
ELASTIC_NET_L1_RATIOS: tuple[float, ...] = (0.1, 0.5, 0.9)
SVM_C_VALUES: tuple[float, ...] = (0.3, 1.0, 3.0)
XGBOOST_LEARNING_RATES: tuple[float, ...] = (0.05, 0.1)
XGBOOST_MAX_DEPTHS: tuple[int, ...] = (2, 4)
XGBOOST_N_ESTIMATORS: tuple[int, ...] = (100, 200)
MLP_ALPHA_VALUES: tuple[float, ...] = (0.001,)
MLP_HIDDEN_LAYER_SIZES: tuple[tuple[int, ...], ...] = ((32,),)
SEQUENCE_STEP_CANDIDATES: tuple[int, ...] = (8, 16)
SEQUENCE_CLASSIFIER_C_VALUES: tuple[float, ...] = (1.0,)
CLASSIFIER_FEATURE_K_CANDIDATES: tuple[int, ...] = (20, 40, 80)
CLASSIFIER_OUTER_FOLDS = 5
CLASSIFIER_INNER_FOLDS = 3
CLASSIFIER_NAMES: tuple[str, ...] = ("elastic_net", "svm", "xgboost", "mlp", "lstm", "transformer")
SIGNATURE_SOURCE_CLASSIFIER_NAMES: tuple[str, ...] = ("elastic_net", "svm", "xgboost")


class TabularSequenceEncoder(BaseEstimator, TransformerMixin):
    """Deterministic sequence-style featurizer for tabular omics classifiers.

    The STAD classifier input is a tabular feature matrix, not a real temporal sequence.
    This encoder reshapes selected features into a reproducible pseudo-sequence so that
    LSTM/Transformer-style baselines can be compared without adding a heavy deep-learning
    framework dependency to the course project.
    """

    def __init__(
        self,
        mode: str = "lstm",
        n_steps: int = 8,
        attention_dim: int = 8,
        random_state: int = 0,
    ) -> None:
        self.mode = mode
        self.n_steps = n_steps
        self.attention_dim = attention_dim
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "TabularSequenceEncoder":
        if self.mode not in {"lstm", "transformer"}:
            raise ValueError("mode must be either 'lstm' or 'transformer'")
        self.n_features_in_ = int(np.asarray(x).shape[1])
        rng = np.random.default_rng(self.random_state + self.n_steps + self.attention_dim)
        self._wq = rng.normal(scale=0.4, size=(4, self.attention_dim))
        self._wk = rng.normal(scale=0.4, size=(4, self.attention_dim))
        self._wv = rng.normal(scale=0.4, size=(4, self.attention_dim))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        sequence = self._reshape(values)
        if self.mode == "lstm":
            return self._lstm_summary(sequence)
        return self._transformer_summary(sequence)

    def _reshape(self, values: np.ndarray) -> np.ndarray:
        n_samples, n_features = values.shape
        n_steps = max(2, min(int(self.n_steps), max(2, n_features)))
        step_width = int(np.ceil(n_features / n_steps))
        padded_width = n_steps * step_width
        if padded_width > n_features:
            values = np.pad(values, ((0, 0), (0, padded_width - n_features)), mode="constant")
        return values.reshape(n_samples, n_steps, step_width)

    def _step_statistics(self, sequence: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                sequence.mean(axis=2),
                sequence.std(axis=2),
                sequence.max(axis=2),
                sequence.min(axis=2),
            ],
            axis=2,
        )

    def _lstm_summary(self, sequence: np.ndarray) -> np.ndarray:
        step_mean = sequence.mean(axis=2)
        step_std = sequence.std(axis=2)
        hidden = np.zeros(step_mean.shape[0], dtype=np.float32)
        hidden_history: list[np.ndarray] = []
        for t in range(step_mean.shape[1]):
            gate = 1.0 / (1.0 + np.exp(-(step_mean[:, t] + 0.5 * step_std[:, t])))
            candidate = np.tanh(step_mean[:, t])
            hidden = gate * hidden + (1.0 - gate) * candidate
            hidden_history.append(hidden.copy())
        history = np.vstack(hidden_history).T
        return np.vstack(
            [
                history[:, -1],
                history.mean(axis=1),
                history.max(axis=1),
                history.min(axis=1),
                step_std.mean(axis=1),
            ]
        ).T.astype(np.float32)

    def _transformer_summary(self, sequence: np.ndarray) -> np.ndarray:
        stats = self._step_statistics(sequence)
        positions = np.linspace(-1.0, 1.0, stats.shape[1], dtype=np.float32)
        stats = stats.copy()
        stats[:, :, 0] = stats[:, :, 0] + positions
        q = stats @ self._wq
        k = stats @ self._wk
        v = stats @ self._wv
        scores = np.matmul(q, np.swapaxes(k, 1, 2)) / np.sqrt(float(self.attention_dim))
        scores = scores - scores.max(axis=2, keepdims=True)
        attention = np.exp(scores)
        attention = attention / attention.sum(axis=2, keepdims=True).clip(min=1e-8)
        context = np.matmul(attention, v)
        pooled = np.concatenate(
            [
                context.mean(axis=1),
                context.max(axis=1),
                stats.mean(axis=1),
            ],
            axis=1,
        )
        return pooled.astype(np.float32)


def build_classifier_feature_matrix(
    omics: dict[str, pd.DataFrame],
    embeddings: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    mechanism_features = compute_mechanism_summary_features(omics).add_prefix("summary__")
    embedding_features = [
        embedding.add_prefix(f"latent__{modality_name}__")
        for modality_name, embedding in embeddings.items()
    ]
    merged = pd.concat([mechanism_features] + embedding_features, axis=1).sort_index()
    merged = merged.fillna(0.0)
    return merged.astype(np.float32)


def _feature_k_grid(n_features: int) -> list[int | str]:
    numeric = sorted({max(1, min(candidate, n_features)) for candidate in CLASSIFIER_FEATURE_K_CANDIDATES})
    if n_features <= max(CLASSIFIER_FEATURE_K_CANDIDATES):
        numeric.append("all")
    return numeric


def _cv_splits_for_labels(labels: pd.Series, target_splits: int) -> int:
    min_class_size = int(labels.value_counts().min())
    return max(2, min(target_splits, min_class_size))


def _build_elastic_net_grid(feature_frame: pd.DataFrame, labels: pd.Series, random_state: int) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k="all")),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    param_grid = {
        "select__k": _feature_k_grid(feature_frame.shape[1]),
        "clf__C": list(ELASTIC_NET_C_VALUES),
        "clf__l1_ratio": list(ELASTIC_NET_L1_RATIOS),
    }
    inner_splits = _cv_splits_for_labels(labels, CLASSIFIER_INNER_FOLDS)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
    )


def _build_svm_grid(feature_frame: pd.DataFrame, labels: pd.Series, random_state: int) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k="all")),
            (
                "clf",
                SVC(
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    param_grid = {
        "select__k": _feature_k_grid(feature_frame.shape[1]),
        "clf__C": list(SVM_C_VALUES),
    }
    inner_splits = _cv_splits_for_labels(labels, CLASSIFIER_INNER_FOLDS)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
    )


def _build_xgboost_grid(feature_frame: pd.DataFrame, labels: pd.Series, random_state: int) -> GridSearchCV:
    n_classes = int(labels.nunique())
    xgb_kwargs = {
        "objective": "binary:logistic" if n_classes == 2 else "multi:softprob",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "verbosity": 0,
    }
    if n_classes > 2:
        xgb_kwargs["num_class"] = n_classes
    pipeline = Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif, k="all")),
            (
                "clf",
                XGBClassifier(**xgb_kwargs),
            ),
        ]
    )
    param_grid = {
        "select__k": _feature_k_grid(feature_frame.shape[1]),
        "clf__learning_rate": list(XGBOOST_LEARNING_RATES),
        "clf__max_depth": list(XGBOOST_MAX_DEPTHS),
        "clf__n_estimators": list(XGBOOST_N_ESTIMATORS),
    }
    inner_splits = _cv_splits_for_labels(labels, CLASSIFIER_INNER_FOLDS)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
    )


def _build_mlp_grid(feature_frame: pd.DataFrame, labels: pd.Series, random_state: int) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k="all")),
            (
                "clf",
                MLPClassifier(
                    activation="relu",
                    solver="adam",
                    early_stopping=False,
                    max_iter=400,
                    random_state=random_state,
                ),
            ),
        ]
    )
    param_grid = {
        "select__k": _feature_k_grid(feature_frame.shape[1]),
        "clf__hidden_layer_sizes": list(MLP_HIDDEN_LAYER_SIZES),
        "clf__alpha": list(MLP_ALPHA_VALUES),
    }
    inner_splits = _cv_splits_for_labels(labels, CLASSIFIER_INNER_FOLDS)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
    )


def _build_sequence_grid(
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    random_state: int,
    mode: str,
) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k="all")),
            ("sequence", TabularSequenceEncoder(mode=mode, random_state=random_state)),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    param_grid = {
        "select__k": _feature_k_grid(feature_frame.shape[1]),
        "sequence__n_steps": list(SEQUENCE_STEP_CANDIDATES),
        "clf__C": list(SEQUENCE_CLASSIFIER_C_VALUES),
    }
    inner_splits = _cv_splits_for_labels(labels, CLASSIFIER_INNER_FOLDS)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
    )


def _build_lstm_grid(feature_frame: pd.DataFrame, labels: pd.Series, random_state: int) -> GridSearchCV:
    return _build_sequence_grid(feature_frame, labels, random_state, mode="lstm")


def _build_transformer_grid(feature_frame: pd.DataFrame, labels: pd.Series, random_state: int) -> GridSearchCV:
    return _build_sequence_grid(feature_frame, labels, random_state, mode="transformer")


def _selected_feature_names(best_estimator: Pipeline, feature_names: pd.Index) -> list[str]:
    variance_mask = best_estimator.named_steps["variance"].get_support()
    post_variance_names = feature_names[variance_mask]
    select_mask = best_estimator.named_steps["select"].get_support()
    return post_variance_names[select_mask].tolist()


def _multiclass_auc(y_true_encoded: np.ndarray, probabilities: np.ndarray, class_labels: np.ndarray) -> float:
    if len(class_labels) <= 2:
        positive_class = class_labels[1] if len(class_labels) == 2 else class_labels[0]
        y_binary = (y_true_encoded == positive_class).astype(int)
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return float(roc_auc_score(y_binary, probabilities[:, 1]))
        return float("nan")
    y_true_bin = label_binarize(y_true_encoded, classes=class_labels)
    return float(roc_auc_score(y_true_bin, probabilities, multi_class="ovr", average="macro"))


def _coefficient_row_labels(class_names: np.ndarray, n_rows: int) -> list[str]:
    if n_rows == len(class_names):
        return [str(class_name) for class_name in class_names]
    pair_labels = [f"{left}__vs__{right}" for left, right in combinations(class_names, 2)]
    if n_rows == len(pair_labels):
        return pair_labels
    return [f"component_{idx + 1}" for idx in range(n_rows)]


def _coefficient_table(best_model: Pipeline, selected_features: list[str]) -> pd.DataFrame:
    classifier = best_model.named_steps["clf"]
    if hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_, dtype=float)
        row_labels = _coefficient_row_labels(np.asarray(getattr(classifier, "classes_", []), dtype=object), coefficients.shape[0])
        rows: list[dict[str, object]] = []
        for row_label, coef_row in zip(row_labels, coefficients, strict=False):
            for feature_name, coefficient in zip(selected_features, coef_row, strict=False):
                rows.append(
                    {
                        "class_label": str(row_label),
                        "feature_name": feature_name,
                        "coefficient": float(coefficient),
                        "abs_coefficient": float(abs(coefficient)),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["class_label", "feature_name", "coefficient", "abs_coefficient"])
        return pd.DataFrame(rows).sort_values(["class_label", "abs_coefficient"], ascending=[True, False])
    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_, dtype=float)
        rows = [
            {
                "class_label": "global_importance",
                "feature_name": feature_name,
                "coefficient": float(importance),
                "abs_coefficient": float(abs(importance)),
            }
            for feature_name, importance in zip(selected_features, importances, strict=False)
        ]
        return pd.DataFrame(rows).sort_values(["abs_coefficient", "feature_name"], ascending=[False, True]).reset_index(drop=True)
    if hasattr(classifier, "coefs_") and classifier.coefs_:
        first_layer = np.asarray(classifier.coefs_[0], dtype=float)
        if first_layer.shape[0] == len(selected_features):
            importances = np.abs(first_layer).sum(axis=1)
            rows = [
                {
                    "class_label": "first_layer_importance",
                    "feature_name": feature_name,
                    "coefficient": float(importance),
                    "abs_coefficient": float(abs(importance)),
                }
                for feature_name, importance in zip(selected_features, importances, strict=False)
            ]
            return pd.DataFrame(rows).sort_values(["abs_coefficient", "feature_name"], ascending=[False, True]).reset_index(drop=True)
    return pd.DataFrame(columns=["class_label", "feature_name", "coefficient", "abs_coefficient"])


def _run_classifier_cv(
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    random_state: int,
    classifier_name: str,
    search_builder: Callable[[pd.DataFrame, pd.Series, int], GridSearchCV],
    encode_training_labels: bool = False,
) -> dict[str, object]:
    x = feature_frame.sort_index()
    y = labels.sort_index().loc[x.index]
    if y.nunique() < 2:
        raise ValueError("Need at least two classes to train classifier")

    outer_splits = _cv_splits_for_labels(y, CLASSIFIER_OUTER_FOLDS)
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_labels = np.unique(y_encoded)

    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    selection_counts = pd.Series(0, index=x.columns, dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        fit_y_train = (
            pd.Series(encoder.transform(y_train), index=y_train.index, name=y_train.name)
            if encode_training_labels
            else y_train
        )
        search = search_builder(x_train, fit_y_train, random_state + fold_idx)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            search.fit(x_train, fit_y_train)
        best_model = search.best_estimator_
        selected_features = _selected_feature_names(best_model, x_train.columns)
        selection_counts.loc[selected_features] += 1

        y_pred = best_model.predict(x_test)
        if encode_training_labels:
            y_pred = encoder.inverse_transform(np.asarray(y_pred, dtype=int))
        probabilities = best_model.predict_proba(x_test)
        y_test_encoded = encoder.transform(y_test)
        auc = _multiclass_auc(y_test_encoded, probabilities, class_labels)
        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "macro_ovr_auroc": auc,
                "best_params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
                "selected_feature_count": int(len(selected_features)),
            }
        )
        class_names = best_model.named_steps["clf"].classes_
        if encode_training_labels:
            class_names = encoder.inverse_transform(np.asarray(class_names, dtype=int))
        for row_idx, patient_id in enumerate(x_test.index):
            row = {
                "patient_id": patient_id,
                "fold": fold_idx,
                "true_label": str(y_test.loc[patient_id]),
                "predicted_label": str(y_pred[row_idx]),
            }
            for class_idx, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = float(probabilities[row_idx, class_idx])
            prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows).sort_values("patient_id").reset_index(drop=True)
    fold_metrics = pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True)
    confusion = pd.DataFrame(
        confusion_matrix(predictions["true_label"], predictions["predicted_label"], labels=sorted(y.unique())),
        index=sorted(y.unique()),
        columns=sorted(y.unique()),
    )
    overall_auc = _multiclass_auc(
        encoder.transform(predictions["true_label"]),
        predictions[[f"prob_{class_name}" for class_name in encoder.classes_]].to_numpy(dtype=float),
        np.arange(len(encoder.classes_)),
    )
    summary = pd.DataFrame(
        [
            {
                "classifier_name": classifier_name,
                "n_samples": int(len(y)),
                "n_classes": int(y.nunique()),
                "n_features": int(x.shape[1]),
                "outer_folds": int(outer_splits),
                "macro_f1": float(f1_score(predictions["true_label"], predictions["predicted_label"], average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(predictions["true_label"], predictions["predicted_label"])),
                "accuracy": float(accuracy_score(predictions["true_label"], predictions["predicted_label"])),
                "macro_ovr_auroc": overall_auc,
            }
        ]
    )

    fit_y = pd.Series(encoder.transform(y), index=y.index, name=y.name) if encode_training_labels else y
    final_search = search_builder(x, fit_y, random_state + 999)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        final_search.fit(x, fit_y)
    final_model = final_search.best_estimator_
    final_selected_features = _selected_feature_names(final_model, x.columns)
    coefficient_table = _coefficient_table(final_model, final_selected_features)
    selected_feature_table = (
        pd.DataFrame(
            {
                "feature_name": selection_counts.index,
                "selected_fold_count": selection_counts.to_numpy(),
                "selected_fold_fraction": selection_counts.to_numpy() / outer_splits,
                "selected_in_final_model": selection_counts.index.isin(final_selected_features),
            }
        )
        .sort_values(["selected_in_final_model", "selected_fold_count", "feature_name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    final_params = pd.DataFrame(
        [
            {
                "classifier_name": classifier_name,
                "best_params": json.dumps(final_search.best_params_, ensure_ascii=False, sort_keys=True),
                "selected_feature_count": int(len(final_selected_features)),
            }
        ]
    )
    return {
        "feature_matrix": x,
        "predictions": predictions,
        "fold_metrics": fold_metrics,
        "summary": summary,
        "confusion": confusion,
        "selected_features": selected_feature_table,
        "coefficients": coefficient_table,
        "final_params": final_params,
    }


def run_elastic_net_cv(
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    random_state: int,
) -> dict[str, object]:
    return _run_classifier_cv(
        feature_frame=feature_frame,
        labels=labels,
        random_state=random_state,
        classifier_name="elastic_net",
        search_builder=_build_elastic_net_grid,
    )


def run_svm_cv(
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    random_state: int,
) -> dict[str, object]:
    return _run_classifier_cv(
        feature_frame=feature_frame,
        labels=labels,
        random_state=random_state,
        classifier_name="svm",
        search_builder=_build_svm_grid,
    )


def run_xgboost_cv(
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    random_state: int,
) -> dict[str, object]:
    return _run_classifier_cv(
        feature_frame=feature_frame,
        labels=labels,
        random_state=random_state,
        classifier_name="xgboost",
        search_builder=_build_xgboost_grid,
        encode_training_labels=True,
    )


def run_mlp_cv(
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    random_state: int,
) -> dict[str, object]:
    return _run_classifier_cv(
        feature_frame=feature_frame,
        labels=labels,
        random_state=random_state,
        classifier_name="mlp",
        search_builder=_build_mlp_grid,
    )


def run_lstm_cv(
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    random_state: int,
) -> dict[str, object]:
    return _run_classifier_cv(
        feature_frame=feature_frame,
        labels=labels,
        random_state=random_state,
        classifier_name="lstm",
        search_builder=_build_lstm_grid,
    )


def run_transformer_cv(
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    random_state: int,
) -> dict[str, object]:
    return _run_classifier_cv(
        feature_frame=feature_frame,
        labels=labels,
        random_state=random_state,
        classifier_name="transformer",
        search_builder=_build_transformer_grid,
    )


def export_classifier_results(
    out_dir: Path,
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    cv_result: dict[str, object],
    queue_key: str,
    analysis_name: str,
    scheme_name: str,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels.rename("subtype").to_csv(out_dir / "labels.tsv", sep="\t", index=True, index_label="patient_id")
    feature_frame.to_csv(out_dir / "feature_matrix.tsv", sep="\t", index=True)
    cv_result["predictions"].to_csv(out_dir / "cv_predictions.tsv", sep="\t", index=False)
    cv_result["fold_metrics"].to_csv(out_dir / "cv_fold_metrics.tsv", sep="\t", index=False)
    cv_result["summary"].to_csv(out_dir / "cv_summary.tsv", sep="\t", index=False)
    cv_result["confusion"].to_csv(out_dir / "confusion_matrix.tsv", sep="\t", index=True)
    cv_result["selected_features"].to_csv(out_dir / "selected_features.tsv", sep="\t", index=False)
    cv_result["coefficients"].to_csv(out_dir / "coefficients.tsv", sep="\t", index=False)
    cv_result["final_params"].to_csv(out_dir / "final_model_params.tsv", sep="\t", index=False)

    summary = cv_result["summary"].copy()
    summary.insert(0, "queue_key", queue_key)
    summary.insert(1, "analysis_name", analysis_name)
    summary.insert(2, "label_scheme_name", scheme_name)
    summary.insert(3, "output_dir", str(out_dir))
    summary["n_selected_features_final"] = int(cv_result["final_params"]["selected_feature_count"].iloc[0])
    summary["best_params"] = str(cv_result["final_params"]["best_params"].iloc[0])
    return summary


def export_elastic_net_results(
    out_dir: Path,
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    cv_result: dict[str, object],
    queue_key: str,
    analysis_name: str,
    scheme_name: str,
) -> pd.DataFrame:
    return export_classifier_results(out_dir, feature_frame, labels, cv_result, queue_key, analysis_name, scheme_name)


def export_svm_results(
    out_dir: Path,
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    cv_result: dict[str, object],
    queue_key: str,
    analysis_name: str,
    scheme_name: str,
) -> pd.DataFrame:
    return export_classifier_results(out_dir, feature_frame, labels, cv_result, queue_key, analysis_name, scheme_name)


def export_xgboost_results(
    out_dir: Path,
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    cv_result: dict[str, object],
    queue_key: str,
    analysis_name: str,
    scheme_name: str,
) -> pd.DataFrame:
    return export_classifier_results(out_dir, feature_frame, labels, cv_result, queue_key, analysis_name, scheme_name)


def export_mlp_results(
    out_dir: Path,
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    cv_result: dict[str, object],
    queue_key: str,
    analysis_name: str,
    scheme_name: str,
) -> pd.DataFrame:
    return export_classifier_results(out_dir, feature_frame, labels, cv_result, queue_key, analysis_name, scheme_name)


def export_lstm_results(
    out_dir: Path,
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    cv_result: dict[str, object],
    queue_key: str,
    analysis_name: str,
    scheme_name: str,
) -> pd.DataFrame:
    return export_classifier_results(out_dir, feature_frame, labels, cv_result, queue_key, analysis_name, scheme_name)


def export_transformer_results(
    out_dir: Path,
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    cv_result: dict[str, object],
    queue_key: str,
    analysis_name: str,
    scheme_name: str,
) -> pd.DataFrame:
    return export_classifier_results(out_dir, feature_frame, labels, cv_result, queue_key, analysis_name, scheme_name)


CLASSIFIER_RUNNERS: dict[str, Callable[[pd.DataFrame, pd.Series, int], dict[str, object]]] = {
    "elastic_net": run_elastic_net_cv,
    "svm": run_svm_cv,
    "xgboost": run_xgboost_cv,
    "mlp": run_mlp_cv,
    "lstm": run_lstm_cv,
    "transformer": run_transformer_cv,
}


CLASSIFIER_EXPORTERS: dict[str, Callable[[Path, pd.DataFrame, pd.Series, dict[str, object], str, str, str], pd.DataFrame]] = {
    "elastic_net": export_elastic_net_results,
    "svm": export_svm_results,
    "xgboost": export_xgboost_results,
    "mlp": export_mlp_results,
    "lstm": export_lstm_results,
    "transformer": export_transformer_results,
}
