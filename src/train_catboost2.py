from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

FILE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_features_with_macro2.parquet"
HORIZONS = [1, 3, 6, 12]

CAT_FEATURES = ["territory_id", "category"]

FEATURES = [
    "territory_id", "category", "month", "year",
    "lag_1", "lag_2", "lag_3", "rolling_mean_3",
    "spatial_lag_1", "market_access",
    "cbr_impact", "sentiment_impact",
    "sentiment_elasticity", "news_regional_impact"
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


def run_multi_horizon_forecasting():
    print("=" * 70)
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("=" * 70)
    if not FILE_PATH.is_file():
        print(f"ОШИБКА: файл {FILE_PATH} не найден.")
        return
    df = pd.read_parquet(FILE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(
        ["territory_id", "category", "date"]
    ).reset_index(drop=True)
    print("Генерация макро-признаков (дельты и impact)...")
    grouped_macro = df.groupby(
        ["territory_id", "category"],
        observed=True
    )
    df["cbr_rate_diff"] = (
        grouped_macro["cbr_key_rate"]
        .diff()
        .fillna(0)
    )
    df["sentiment_diff"] = (
        grouped_macro["macro_news_sentiment"]
        .diff()
        .fillna(0)
    )
    df["cbr_impact"] = (
        df["cbr_rate_diff"] * df["rolling_mean_3"]
    )
    df["sentiment_impact"] = (
        df["sentiment_diff"] * df["rolling_mean_3"]
    )
    df["recent_volatility"] = (
        df[["lag_1", "lag_2", "lag_3"]]
        .std(axis=1)
        .fillna(0)
        / (df["rolling_mean_3"] + 1)
    )
    df["sentiment_elasticity"] = (
        df["sentiment_diff"] * df["recent_volatility"]
    )
    df["news_regional_impact"] = (
        df["sentiment_diff"] * df["market_access"]
    )
    for c in CAT_FEATURES:
        df[c] = (
            df[c]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )
    numeric_features = [
        col for col in FEATURES
        if col not in CAT_FEATURES
    ]
    df[numeric_features] = (
        df[numeric_features]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    print(f"Количество строк: {len(df):,}")
    print(
        f"Период: "
        f"{df['date'].min().date()} — "
        f"{df['date'].max().date()}"
    )
    print("\n" + "=" * 70)
    print("2. СОЗДАНИЕ TARGET ДЛЯ 1, 3, 6 И 12 МЕСЯЦЕВ")
    print("=" * 70)
    grouped = df.groupby(
        ["territory_id", "category"],
        observed=True
    )
    for h in HORIZONS:
        df[f"target_{h}"] = (
            grouped["value"].shift(-h)
        )
        df[f"target_date_{h}"] = (
            grouped["date"].shift(-h)
        )
        print(f"Горизонт {h:2d} мес. создан.")
    print("\n" + "=" * 70)
    print("3. ОБУЧЕНИЕ CATBOOST")
    print("=" * 70)
    results = {}
    for h in HORIZONS:
        print("\n" + "#" * 70)
        print(f"ГОРИЗОНТ {h} МЕС.")
        print("#" * 70)
        target_col = f"target_{h}"
        target_date_col = f"target_date_{h}"
        df_h = df.dropna(
            subset=[target_col, target_date_col]
        ).copy()
        print(
            f"Доступно строк с известным target: "
            f"{len(df_h):,}"
        )
        dates = sorted(df_h["date"].unique())
        val_dates = dates[-3:]
        fold_maes = []
        fold_r2 = []
        fold_iterations = []
        for fold_number, val_date in enumerate(
            val_dates, start=1
        ):
            val_date = pd.Timestamp(val_date)
            val_df = df_h[
                df_h["date"] == val_date
            ].copy()
            strict_train = df_h[
                df_h[target_date_col] < val_date
            ].copy()
            if len(strict_train) > 0:
                train_df = strict_train
                validation_mode = "STRICT"
            else:
                train_df = df_h[
                    df_h["date"] < val_date
                ].copy()
                validation_mode = "FALLBACK"
            if len(train_df) == 0:
                train_df = df_h[
                    df_h["date"] != val_date
                ].copy()
                validation_mode = "FALLBACK-ALL"
            if len(val_df) == 0:
                continue
            if len(train_df) == 0:
                print(
                    f"\nFold {fold_number} пропущен: "
                    "нет обучающих данных."
                )
                continue
            print(f"\nFold {fold_number}")
            print(f"Validation date: {val_date.date()}")
            print(f"Режим: {validation_mode}")
            print(f"Train: {len(train_df):,}")
            print(f"Validation: {len(val_df):,}")
            X_train = train_df[FEATURES]
            y_train = train_df[target_col]
            X_val = val_df[FEATURES]
            y_val = val_df[target_col]
            train_pool = Pool(
                data=X_train,
                label=y_train,
                cat_features=CAT_FEATURES
            )
            val_pool = Pool(
                data=X_val,
                label=y_val,
                cat_features=CAT_FEATURES
            )
            model = CatBoostRegressor(
                **CATBOOST_PARAMS
            )
            model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=30,
                use_best_model=True
            )
            y_pred = model.predict(val_pool)
            mae = mean_absolute_error(
                y_val, y_pred
            )
            r2 = r2_score(
                y_val, y_pred
            )
            best_iteration = model.get_best_iteration()
            fold_maes.append(mae)
            fold_r2.append(r2)
            fold_iterations.append(best_iteration)
            print(f"\nMAE: {mae:,.2f} руб.")
            print(f"R²: {r2:.4f}")
            print(f"Best iteration: {best_iteration}")
        if len(fold_maes) == 0:
            print(
                f"\nДля горизонта {h} мес. "
                "нет допустимых фолдов."
            )
            continue
        mean_mae = np.mean(fold_maes)
        mean_r2 = np.mean(fold_r2)
        mean_iteration = np.mean(fold_iterations)
        results[h] = {
            "mae": mean_mae,
            "r2": mean_r2,
            "best_iteration": mean_iteration
        }
        print("\n" + "-" * 60)
        print(f"ИТОГ ДЛЯ {h} МЕС.")
        print(f"Средний MAE: {mean_mae:,.2f} руб.")
        print(f"Средний R²: {mean_r2:.4f}")
        print(
            f"Средняя best iteration: "
            f"{mean_iteration:.0f}"
        )
        print("-" * 60)
    print("\n" + "=" * 85)
    print("🏆 ИТОГОВЫЕ МЕТРИКИ CATBOOST")
    print("=" * 85)
    print(
        f"{'Горизонт':<15}"
        f"{'MAE, руб.':<25}"
        f"{'R²':<15}"
        f"{'Best iteration':<20}"
    )
    print("-" * 85)
    for h in HORIZONS:
        if h not in results:
            continue
        print(
            f"{str(h) + ' мес.':<15}"
            f"{results[h]['mae']:<25,.2f}"
            f"{results[h]['r2']:<15.4f}"
            f"{results[h]['best_iteration']:<20.0f}"
        )
    print("=" * 85)
    if not results:
        print("Нет результатов для сравнения.")
        return
    best_horizon = min(
        results,
        key=lambda h: results[h]["mae"]
    )
    print("\nЛучший результат по MAE:")
    print(f"Горизонт: {best_horizon} мес.")
    print(
        f"MAE: "
        f"{results[best_horizon]['mae']:,.2f} руб."
    )


if __name__ == "__main__":
    run_multi_horizon_forecasting()
