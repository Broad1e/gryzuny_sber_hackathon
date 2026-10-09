import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

FILE_PATH = "train_features_with_macro.parquet"

HORIZONS = [1, 3, 6, 12]

CAT_FEATURES = [
    "territory_id",
    "category"
]

FEATURES = [
    "territory_id",
    "category",
    "month",
    "year",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "spatial_lag_1",
    "market_access",
    "macro_news_sentiment",
    "cbr_key_rate"
]

CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 8,
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "random_seed": 42,
    "l2_leaf_reg": 5,
    "random_strength": 1,
    "bagging_temperature": 1,
    "task_type": "CPU",
    "thread_count": -1,
    "verbose": 100
}

def run_catboost_forecasting():

    if not os.path.exists(FILE_PATH):
        print(f"Файл {FILE_PATH} не найден")
        return

    df = pd.read_parquet(FILE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["territory_id", "category", "date"]
    ).reset_index(drop=True)

    print(f"Строк: {len(df):,}")

    grouped = df.groupby(
        ["territory_id", "category"],
        observed=True
    )

    for h in HORIZONS:
        df[f"target_{h}"] = grouped["value"].shift(-h)
        df[f"target_date_{h}"] = grouped["date"].shift(-h)

    results = {}

    for h in HORIZONS:

        print(f"\nГОРИЗОНТ {h} МЕСЯЦЕВ")

        target_col = f"target_{h}"
        target_date_col = f"target_date_{h}"

        df_h = df.dropna(
            subset=[
                target_col,
                target_date_col
            ]
        ).copy()

        dates = sorted(df_h["date"].unique())
        val_dates = dates[-3:]

        maes = []
        r2s = []

        for fold, val_date in enumerate(val_dates, start=1):

            val_date = pd.Timestamp(val_date)

            val_df = df_h[
                df_h["date"] == val_date
            ]

            train_df = df_h[
                df_h[target_date_col] < val_date
            ]

            if len(train_df) == 0:
                train_df = df_h[
                    df_h["date"] < val_date
                ]

            print(
                f"Fold {fold} | "
                f"Дата: {val_date.date()} | "
                f"Train: {len(train_df):,} | "
                f"Val: {len(val_df):,}"
            )

            X_train = train_df[FEATURES]
            y_train = train_df[target_col]

            X_val = val_df[FEATURES]
            y_val = val_df[target_col]

            model = CatBoostRegressor(
                **CATBOOST_PARAMS
            )

            model.fit(
                X_train,
                y_train,
                cat_features=CAT_FEATURES,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100
            )

            pred = model.predict(X_val)

            mae = mean_absolute_error(
                y_val,
                pred
            )

            r2 = r2_score(
                y_val,
                pred
            )

            maes.append(mae)
            r2s.append(r2)

            print(
                f"MAE: {mae:,.2f} | "
                f"R²: {r2:.4f} | "
                f"Trees: {model.best_iteration_}"
            )

        results[h] = {
            "mae": np.mean(maes),
            "r2": np.mean(r2s)
        }

        print(
            f"ИТОГ {h} мес. | "
            f"MAE: {results[h]['mae']:,.2f} | "
            f"R²: {results[h]['r2']:.4f}"
        )

    print("\n" + "=" * 50)
    print("CATBOOST RESULTS")
    print("=" * 50)

    for h in HORIZONS:
        print(
            f"{h} мес. | "
            f"MAE: {results[h]['mae']:,.2f} | "
            f"R²: {results[h]['r2']:.4f}"
        )

    best_h = min(
        results,
        key=lambda x: results[x]["mae"]
    )

    print(
        f"\nЛучший горизонт: {best_h} мес."
    )

    print(
        f"MAE: {results[best_h]['mae']:,.2f}"
    )


if __name__ == "__main__":
    run_catboost_forecasting()