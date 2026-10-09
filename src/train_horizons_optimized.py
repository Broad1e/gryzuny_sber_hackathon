import os
import yaml

try:
    with open("config/lightgbm_params.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        DATA_DIRECTORY = config.get("paths", {}).get("data_dir", "data/raw")
        PROCESSED_DIRECTORY = config.get("paths", {}).get("processed_dir", "data/processed")
        
        # Загрузка гиперпараметров и признаков
        yaml_params = config.get("hyperparameters", {})
        yaml_features = config.get("features", {})
except FileNotFoundError:
    DATA_DIRECTORY = "data/raw"
    PROCESSED_DIRECTORY = "data/processed"
    yaml_params = {}
    yaml_features = {}
import warnings
import lightgbm as lgb
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_multi_horizon_forecasting(n_trials=500):
    file_path = os.path.join("train_features_with_macro.parquet")
    if not os.path.exists(file_path):
        print(f"ОШИБКА: Файл {file_path} не найден.")
        return

    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["territory_id", "category", "date"]).reset_index(drop=True)

    horizons = [1, 3, 6, 12]
    grouped = df.groupby(["territory_id", "category"])

    for h in horizons:
        df[f"target_{h}"] = grouped["value"].shift(-h)

    cat_features = ["territory_id", "category"]
    for c in cat_features:
        df[c] = df[c].astype("category")

    features = [
        "territory_id", "category", "month", "year",
        "lag_1", "lag_2", "lag_3", "rolling_mean_3",
        "spatial_lag_1", "market_access",
        "macro_news_sentiment", "cbr_key_rate"
    ]

    def objective(trial):
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "boosting_type": "gbdt",
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 500, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 100.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 2.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 100.0),
            "max_bin": trial.suggest_int("max_bin", 31, 511),
            "min_data_in_bin": trial.suggest_int("min_data_in_bin", 1, 100),
            "cat_smooth": trial.suggest_float("cat_smooth", 1e-3, 100.0, log=True),
            "cat_l2": trial.suggest_float("cat_l2", 1e-3, 100.0, log=True),
            "max_cat_threshold": trial.suggest_int("max_cat_threshold", 4, 64),
            "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 4, 32),
            "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }

        horizon_maes = []

        for h in horizons:
            target_col = f"target_{h}"
            df_h = df.dropna(subset=[target_col])
            dates = sorted(df_h["date"].unique())
            val_dates = dates[-3:]
            maes = []

            for val_date in val_dates:
                train_df = df_h[df_h["date"] < val_date]
                val_df = df_h[df_h["date"] == val_date]

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    train_df[features],
                    train_df[target_col],
                    eval_set=[(val_df[features], val_df[target_col])],
                    eval_metric="mae",
                    categorical_feature=cat_features,
                    callbacks=[lgb.early_stopping(100, verbose=False)]
                )

                pred = model.predict(
                    val_df[features],
                    num_iteration=model.best_iteration_
                )

                maes.append(
                    mean_absolute_error(
                        val_df[target_col],
                        pred
                    )
                )

            horizon_maes.append(np.mean(maes))

        return np.mean(horizon_maes)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("\nЛучший средний MAE:", round(study.best_value, 2))
    print("\nЛучшие общие параметры:")
    for param, value in study.best_params.items():
        print(f"{param}: {value}")

    print("\nMAE по каждому горизонту с лучшими параметрами:")

    best_params = {
        **study.best_params,
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 3000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    for h in horizons:
        target_col = f"target_{h}"
        df_h = df.dropna(subset=[target_col])
        dates = sorted(df_h["date"].unique())
        val_dates = dates[-3:]
        maes = []

        for val_date in val_dates:
            train_df = df_h[df_h["date"] < val_date]
            val_df = df_h[df_h["date"] == val_date]

            model = lgb.LGBMRegressor(**best_params)
            model.fit(
                train_df[features],
                train_df[target_col],
                eval_set=[(val_df[features], val_df[target_col])],
                eval_metric="mae",
                categorical_feature=cat_features,
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )

            pred = model.predict(
                val_df[features],
                num_iteration=model.best_iteration_
            )

            maes.append(mean_absolute_error(val_df[target_col], pred))

        print(f"{h:2d} мес.: MAE = {np.mean(maes):,.2f}")

run_multi_horizon_forecasting(
    n_trials=500
)