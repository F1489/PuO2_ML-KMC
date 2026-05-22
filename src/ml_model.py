"""Model training and inference helpers for event Delta E prediction."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

try:  # Optional strong tabular models. The project still runs without them.
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None
    CatBoostRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBRegressor = None


TARGET_COLUMNS = {
    "delta_E",
    "delta_fluorite_order",
    "delta_bulk_order",
    "delta_coordination_error",
    "is_energy_lowering",
    "is_crystallizing_event",
    "atom_type",
    "event_kind",
    "uncertainty",
    "good_event_probability",
    "step",
    "source_state",
    "order_biased_source",
}


class RegressionEnsemble:
    """Small averaging ensemble that exposes uncertainty through member spread."""

    def __init__(self, members: list[object], name: str):
        self.members = members
        self.name = name

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionEnsemble":
        for member in self.members:
            member.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.vstack([member.predict(X) for member in self.members])
        return np.mean(predictions, axis=0)

    @property
    def feature_importances_(self) -> np.ndarray:
        importances = [
            np.asarray(member.feature_importances_, dtype=float)
            for member in self.members
            if hasattr(member, "feature_importances_")
        ]
        if not importances:
            raise AttributeError("No ensemble member exposes feature_importances_")
        return np.mean(np.vstack(importances), axis=0)


def _make_external_regressors(seed: int, ensemble_size: int) -> dict[str, object]:
    candidates: dict[str, object] = {}
    if CatBoostRegressor is not None:
        members = [
            CatBoostRegressor(
                iterations=450,
                depth=6,
                learning_rate=0.045,
                loss_function="RMSE",
                random_seed=seed + i,
                verbose=False,
                allow_writing_files=False,
            )
            for i in range(max(1, ensemble_size))
        ]
        candidates["CatBoostRegressorEnsemble"] = RegressionEnsemble(members, "CatBoostRegressorEnsemble")
    if XGBRegressor is not None:
        members = [
            XGBRegressor(
                n_estimators=420,
                max_depth=5,
                learning_rate=0.045,
                subsample=0.88,
                colsample_bytree=0.88,
                objective="reg:squarederror",
                random_state=seed + i,
                n_jobs=-1,
            )
            for i in range(max(1, ensemble_size))
        ]
        candidates["XGBRegressorEnsemble"] = RegressionEnsemble(members, "XGBRegressorEnsemble")
    if LGBMRegressor is not None:
        members = [
            LGBMRegressor(
                n_estimators=500,
                max_depth=-1,
                num_leaves=31,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed + i,
                n_jobs=-1,
                verbose=-1,
            )
            for i in range(max(1, ensemble_size))
        ]
        candidates["LGBMRegressorEnsemble"] = RegressionEnsemble(members, "LGBMRegressorEnsemble")
    return candidates


def _make_external_classifiers(seed: int) -> dict[str, object]:
    candidates: dict[str, object] = {}
    if CatBoostClassifier is not None:
        candidates["CatBoostClassifier"] = CatBoostClassifier(
            iterations=350,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
    if XGBClassifier is not None:
        candidates["XGBClassifier"] = XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.88,
            colsample_bytree=0.88,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    if LGBMClassifier is not None:
        candidates["LGBMClassifier"] = LGBMClassifier(
            n_estimators=400,
            num_leaves=31,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    return candidates


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2D333B",
            "axes.labelcolor": "#1F2328",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.color": "#1F2328",
            "ytick.color": "#1F2328",
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "legend.facecolor": "white",
            "legend.edgecolor": "#C9CED6",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in TARGET_COLUMNS]


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _kmc_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_lowering = y_true < 0.0
    y_pred_lowering = y_pred < 0.0
    top_fraction = min(0.2, max(1.0 / max(len(y_true), 1), 0.05))
    top_n = max(1, int(np.ceil(len(y_true) * top_fraction)))
    top_indices = np.argsort(y_pred)[:top_n]
    sign_accuracy = float(np.mean(y_true_lowering == y_pred_lowering))
    true_positive = float(np.sum(y_true_lowering & y_pred_lowering))
    recall = true_positive / max(float(np.sum(y_true_lowering)), 1.0)
    precision = true_positive / max(float(np.sum(y_pred_lowering)), 1.0)
    top_precision = float(np.mean(y_true_lowering[top_indices])) if len(top_indices) else 0.0
    top_mean_delta_e = float(np.mean(y_true[top_indices])) if len(top_indices) else 0.0
    metrics = _regression_metrics(y_true, y_pred)
    metrics.update(
        {
            "sign_accuracy_from_regression": sign_accuracy,
            "precision_energy_lowering_from_regression": precision,
            "recall_energy_lowering_from_regression": recall,
            "top_event_fraction": float(top_fraction),
            "top_event_precision_energy_lowering": top_precision,
            "top_event_mean_exact_delta_E": top_mean_delta_e,
            "kmc_model_score": float(
                0.35 * sign_accuracy
                + 0.25 * recall
                + 0.25 * top_precision
                + 0.15 * max(0.0, metrics["r2"])
            ),
        }
    )
    return metrics


def _classifier_probabilities(model, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if not hasattr(model, "predict_proba"):
        return predict_good_event(model, X).astype(float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(X.shape[0], dtype=float)
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def _threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probabilities >= threshold).astype(int)
    accuracy = float(accuracy_score(y_true, pred))
    precision = float(precision_score(y_true, pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, pred, pos_label=1, zero_division=0))
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "kmc_threshold_score": float(0.30 * accuracy + 0.25 * precision + 0.45 * recall),
    }


def tune_classifier_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    """Choose a probability threshold with a recall-heavy kMC objective."""
    thresholds = np.round(np.arange(0.20, 0.801, 0.025), 3)
    rows = [_threshold_metrics(y_true, probabilities, float(threshold)) for threshold in thresholds]
    best = max(rows, key=lambda row: row["kmc_threshold_score"])
    return {"selected": best, "candidates": rows}


def _plot_predicted_vs_exact(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """Save a Russian predicted-vs-exact Delta E plot."""
    _apply_plot_style()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.scatter(y_true, y_pred, s=28, alpha=0.78, color="#2F6B9A", edgecolor="white", linewidth=0.4)
    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([low, high], [low, high], color="#A33D3D", linewidth=2, label="идеальное совпадение")
    ax.set_title("Проверка ML-прогноза энергии", fontweight="bold")
    ax.set_xlabel("Точное Delta E, эВ")
    ax.set_ylabel("Предсказанное Delta E, эВ")
    ax.grid(True, color="#D7DCE2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB7C2")
    ax.spines["bottom"].set_color("#AEB7C2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_feature_importance(model, feature_columns: list[str], model_dir: Path) -> None:
    """Save feature importance table and plot when the selected model exposes it."""
    if not hasattr(model, "feature_importances_"):
        return
    importance = np.asarray(model.feature_importances_, dtype=float)
    df = pd.DataFrame({"feature": feature_columns, "importance": importance})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(model_dir / "feature_importance.csv", index=False)

    _apply_plot_style()
    top = df.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.barh(top["feature"], top["importance"], color="#245C4F")
    ax.set_title("Важность признаков для прогноза Delta E", fontweight="bold")
    ax.set_xlabel("Важность")
    ax.grid(True, axis="x", color="#D7DCE2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB7C2")
    ax.spines["bottom"].set_color("#AEB7C2")
    fig.tight_layout()
    fig.savefig(model_dir / "feature_importance.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def train_models(
    dataset_csv: str | Path,
    model_dir: str | Path,
    seed: int = 42,
    classifier_target: str = "is_energy_lowering",
    use_external_models: bool = True,
    ensemble_size: int = 3,
) -> dict[str, object]:
    """Train and compare Delta E regressors plus an event-quality classifier."""
    df = pd.read_csv(dataset_csv)
    if classifier_target not in df.columns:
        classifier_target = "is_energy_lowering"
    feature_columns = _feature_columns(df)
    X = df[feature_columns].to_numpy(dtype=float)
    y_reg = df["delta_E"].to_numpy(dtype=float)
    y_cls = df[classifier_target].to_numpy(dtype=int)

    class_counts = np.bincount(y_cls, minlength=2)
    stratify = y_cls if len(np.unique(y_cls)) > 1 and np.min(class_counts[class_counts > 0]) >= 2 else None
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=seed, stratify=stratify
    )
    candidate_regressors = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=350,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=seed),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=seed),
    }
    if use_external_models:
        candidate_regressors.update(_make_external_regressors(seed, ensemble_size))
    regression_report: dict[str, dict[str, float]] = {}
    fitted_regressors = {}
    for name, candidate in candidate_regressors.items():
        candidate.fit(X_train, yr_train)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            pred = candidate.predict(X_test)
        regression_report[name] = _kmc_regression_metrics(yr_test, pred)
        fitted_regressors[name] = candidate
    best_regressor_name = max(regression_report, key=lambda name: regression_report[name]["kmc_model_score"])
    regressor = fitted_regressors[best_regressor_name]

    candidate_classifiers = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=350,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
    }
    if use_external_models:
        candidate_classifiers.update(_make_external_classifiers(seed))
    classifier_report: dict[str, dict[str, float]] = {}
    fitted_classifiers = {}
    for name, candidate in candidate_classifiers.items():
        candidate.fit(X_train, yc_train)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            pred = candidate.predict(X_test)
        accuracy = float(accuracy_score(yc_test, pred))
        precision = float(precision_score(yc_test, pred, pos_label=1, zero_division=0))
        recall = float(recall_score(yc_test, pred, pos_label=1, zero_division=0))
        classifier_report[name] = {
            "accuracy": accuracy,
            f"precision_{classifier_target}": precision,
            f"recall_{classifier_target}": recall,
            "kmc_classifier_score": float(0.35 * accuracy + 0.30 * precision + 0.35 * recall),
        }
        fitted_classifiers[name] = candidate
    best_classifier_name = max(classifier_report, key=lambda name: classifier_report[name]["kmc_classifier_score"])
    classifier = fitted_classifiers[best_classifier_name]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        pred_reg = regressor.predict(X_test)
        pred_cls = classifier.predict(X_test)
    classifier_probabilities = _classifier_probabilities(classifier, X_test)
    threshold_report = tune_classifier_threshold(yc_test, classifier_probabilities)
    selected_threshold = float(threshold_report["selected"]["threshold"])
    pred_cls_tuned = (classifier_probabilities >= selected_threshold).astype(int)
    metrics = {
        "mae": float(mean_absolute_error(yr_test, pred_reg)),
        "rmse": float(np.sqrt(mean_squared_error(yr_test, pred_reg))),
        "r2": float(r2_score(yr_test, pred_reg)),
        "accuracy": float(accuracy_score(yc_test, pred_cls)),
        f"precision_{classifier_target}": float(precision_score(yc_test, pred_cls, pos_label=1, zero_division=0)),
        f"recall_{classifier_target}": float(recall_score(yc_test, pred_cls, pos_label=1, zero_division=0)),
        "tuned_threshold": selected_threshold,
        "tuned_accuracy": float(accuracy_score(yc_test, pred_cls_tuned)),
        f"tuned_precision_{classifier_target}": float(precision_score(yc_test, pred_cls_tuned, pos_label=1, zero_division=0)),
        f"tuned_recall_{classifier_target}": float(recall_score(yc_test, pred_cls_tuned, pos_label=1, zero_division=0)),
        "best_regressor": best_regressor_name,
        "best_classifier": best_classifier_name,
        "classifier_target": classifier_target,
        "model_selection_objective": "kMC-oriented ranking/sign score",
        "n_training_rows": int(len(df)),
        "n_features": int(len(feature_columns)),
        "use_external_models": bool(use_external_models),
        "ensemble_size": int(max(1, ensemble_size)),
    }

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, model_dir / "regressor.joblib")
    joblib.dump(classifier, model_dir / "classifier.joblib")
    joblib.dump(feature_columns, model_dir / "feature_columns.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (model_dir / "regressor_comparison.json").write_text(json.dumps(regression_report, indent=2), encoding="utf-8")
    (model_dir / "classifier_comparison.json").write_text(json.dumps(classifier_report, indent=2), encoding="utf-8")
    (model_dir / "classifier_threshold.json").write_text(
        json.dumps(threshold_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    model_selection_report = {
        "seed": seed,
        "test_size": 0.2,
        "classifier_target": classifier_target,
        "feature_columns": feature_columns,
        "selected": {
            "regressor": best_regressor_name,
            "classifier": best_classifier_name,
            "classifier_threshold": selected_threshold,
            "objective": "kMC-oriented ranking/sign score",
        },
        "regressors": regression_report,
        "classifiers": classifier_report,
    }
    (model_dir / "model_selection_report.json").write_text(
        json.dumps(model_selection_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _save_feature_importance(regressor, feature_columns, model_dir)
    _plot_predicted_vs_exact(yr_test, pred_reg, model_dir / "predicted_vs_exact_delta_E.png")
    return metrics


def load_models(model_dir: str | Path):
    """Load regressor, classifier, and feature column metadata."""
    model_dir = Path(model_dir)
    return (
        joblib.load(model_dir / "regressor.joblib"),
        joblib.load(model_dir / "classifier.joblib"),
        joblib.load(model_dir / "feature_columns.joblib"),
    )


def load_classifier_threshold(model_dir: str | Path, default: float = 0.5) -> float:
    """Load the tuned classifier probability threshold when available."""
    path = Path(model_dir) / "classifier_threshold.json"
    if not path.exists():
        return float(default)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return float(report.get("selected", {}).get("threshold", default))
    except (OSError, ValueError, TypeError):
        return float(default)


def predict_delta_E(model, X) -> np.ndarray:
    """Predict Delta E in eV for one or more feature rows."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        return np.asarray(model.predict(np.asarray(X, dtype=float)))


def predict_good_event(classifier, X) -> np.ndarray:
    """Predict whether events lower energy."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        return np.asarray(classifier.predict(np.asarray(X, dtype=float)))


def predict_good_event_probability(classifier, X) -> np.ndarray:
    """Return P(Delta E < 0) for one or more feature rows."""
    X = np.asarray(X, dtype=float)
    return _classifier_probabilities(classifier, X)


def estimate_uncertainty_rf(model, X) -> np.ndarray:
    """Estimate model uncertainty from ensemble spread when available."""
    X = np.asarray(X, dtype=float)
    if isinstance(model, RegressionEnsemble):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            member_predictions = np.vstack([member.predict(X) for member in model.members])
        return np.std(member_predictions, axis=0)
    if isinstance(model, RandomForestRegressor | ExtraTreesRegressor):
        tree_predictions = np.vstack([tree.predict(X) for tree in model.estimators_])
        return np.std(tree_predictions, axis=0)
    if isinstance(model, GradientBoostingRegressor):
        tree_predictions = np.vstack([tree[0].predict(X) for tree in model.estimators_])
        return np.std(tree_predictions, axis=0)
    return np.zeros(X.shape[0], dtype=float)
