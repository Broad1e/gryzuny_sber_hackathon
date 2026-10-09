
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
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

FILE_PATH = os.path.join(PROCESSED_DIRECTORY, "train_features_with_macro.parquet")
HORIZONS = [1, 3, 6, 12]

CAT_FEATURES = ["territory_id", "category"]
if yaml_features:
    # Динамическая сборка признаков из конфига
    FEATURES = ["territory_id", "category", "month", "year"]
    if yaml_features.get("time_lags"):
        FEATURES += [f"lag_{l}" for l in yaml_features["time_lags"]]
    if yaml_features.get("rolling_windows"):
        FEATURES += [f"rolling_mean_{w}" for w in yaml_features["rolling_windows"]]
    FEATURES += yaml_features.get("spatial", [])
    FEATURES += yaml_features.get("macro", [])
    if "cbr_impact" not in FEATURES: FEATURES.extend(["cbr_impact", "sentiment_impact"]) # Локализованные фичи


FEATURES = [
    "territory_id", "category", "month", "year",
    "lag_1", "lag_2", "lag_3", "rolling_mean_3",
    "spatial_lag_1", "market_access",

    # Тональность новостей RuBERT
    "macro_news_sentiment",
    "nlp_negative_asof",
    "nlp_neutral_asof",
    "nlp_positive_asof",

    # Динамика новостного фона
    "sentiment_diff",
    "sentiment_rolling_3",

    # Макроэкономические показатели
    "cbr_key_rate",
    "cbr_rate_diff",

    # Экономические темы новостей
    "news_topic_inflation",
    "news_topic_credit",
    "news_topic_demand",
    "news_topic_supply",

    # Индикаторы отсутствующих данных
    "news_missing",
    "rate_missing"
]
if yaml_features:
    # Динамическая сборка признаков из конфига
    FEATURES = ["territory_id", "category", "month", "year"]
    if yaml_features.get("time_lags"):
        FEATURES += [f"lag_{l}" for l in yaml_features["time_lags"]]
    if yaml_features.get("rolling_windows"):
        FEATURES += [f"rolling_mean_{w}" for w in yaml_features["rolling_windows"]]
    FEATURES += yaml_features.get("spatial", [])
    FEATURES += yaml_features.get("macro", [])
    if "cbr_impact" not in FEATURES: FEATURES.extend(["cbr_impact", "sentiment_impact"]) # Локализованные фичи


BEST_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.0220176595001618,
    "num_leaves": 183,
    "max_depth": 15,
    "min_child_samples": 19,
    "min_child_weight": 0.007151638195234174,
    "min_split_gain": 1.3599999878907225,
    "subsample": 0.5407062072232978,
    "subsample_freq": 4,
    "colsample_bytree": 0.5368114047175374,
    "reg_alpha": 4.796707195721752e-05,
    "reg_lambda": 0.00014036887761526744,
    "path_smooth": 3.4236657953556566,
    "max_bin": 331,
    "min_data_in_bin": 6,
    "cat_smooth": 0.017087825167306073,
    "cat_l2": 0.0024036342826980704,
    "max_cat_threshold": 58,
    "max_cat_to_onehot": 18,
    "extra_trees": False,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}
if yaml_params:
    BEST_PARAMS.update(yaml_params)



def run_multi_horizon_forecasting():
    
    print("1. ЗАГРУЗКА ДАННЫХ")
    

    if not os.path.exists(FILE_PATH):
        print(f"ОШИБКА: файл {FILE_PATH} не найден.")
        return

    df = pd.read_parquet(FILE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(
        ["territory_id", "category", "date"]
    ).reset_index(drop=True)

    print(f"Количество строк: {len(df):,}")
    print(
        f"Период: {df['date'].min().date()} — "
        f"{df['date'].max().date()}"
    )

    
    print("2. СОЗДАНИЕ TARGET ДЛЯ 1, 3, 6 И 12 МЕСЯЦЕВ")
    

    grouped = df.groupby(
        ["territory_id", "category"],
        observed=True
    )

    for h in HORIZONS:
        df[f"target_{h}"] = grouped["value"].shift(-h)
        df[f"target_date_{h}"] = grouped["date"].shift(-h)
        print(f"Горизонт {h:2d} мес. создан.")

    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")

    
    print("3. ОБУЧЕНИЕ МОДЕЛЕЙ")
    

    results = {}

    for h in HORIZONS:
        
        print(f"ГОРИЗОНТ {h} МЕС.")
        

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

            print(f"\nFold {fold_number}")
            print(f"Validation date: {val_date.date()}")
            print(f"Режим: {validation_mode}")
            print(f"Train: {len(train_df):,}")
            print(f"Validation: {len(val_df):,}")

            X_train = train_df[FEATURES]
            y_train = train_df[target_col]

            X_val = val_df[FEATURES]
            y_val = val_df[target_col]

            model = lgb.LGBMRegressor(**BEST_PARAMS)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                categorical_feature=CAT_FEATURES,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=30,
                        verbose=False
                    )
                ]
            )

            y_pred = model.predict(
                X_val,
                num_iteration=model.best_iteration_
            )

            mae = mean_absolute_error(
                y_val, y_pred
            )

            r2 = r2_score(
                y_val, y_pred
            )

            fold_maes.append(mae)
            fold_r2.append(r2)
            fold_iterations.append(
                model.best_iteration_
            )

            print(f"MAE: {mae:,.2f} руб.")
            print(f"R²: {r2:.4f}")
            print(
                f"Best iteration: "
                f"{model.best_iteration_}"
            )

        mean_mae = np.mean(fold_maes)
        mean_r2 = np.mean(fold_r2)
        mean_iteration = np.mean(fold_iterations)

        results[h] = {
            "mae": mean_mae,
            "r2": mean_r2,
            "best_iteration": mean_iteration
        }

        
        print(f"ИТОГ ДЛЯ {h} МЕС.")
        print(f"Средний MAE: {mean_mae:,.2f} руб.")
        print(f"Средний R²: {mean_r2:.4f}")
        

    
    print("Итоговые метрики LIGHTGBM")
    

    print(
        f"{'Горизонт':<15}"
        f"{'MAE, руб.':<25}"
        f"{'R²':<15}"
    )

    

    for h in HORIZONS:
        print(
            f"{str(h) + ' мес.':<15}"
            f"{results[h]['mae']:<25,.2f}"
            f"{results[h]['r2']:<15.4f}"
        )

    

    best_horizon = min(
        results,
        key=lambda h: results[h]["mae"]
    )

    print("\nЛучший результат по MAE:")
    print(f"Горизонт: {best_horizon} мес.")
    print(
        f"MAE: {results[best_horizon]['mae']:,.2f} руб."
    )

def plot_presentation_forecasts():
    print("\n" + "=" * 75)
    print("ГЕНЕРАЦИЯ ГРАФИКОВ ДЛЯ ПРЕЗЕНТАЦИИ (Без новостей)")
    print("=" * 75)
    
    df = pd.read_parquet(FILE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["territory_id", "category", "date"]).reset_index(drop=True)
    
    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")

    # 2 муниципалитета
    top_territories = df.groupby('territory_id')['value'].sum().nlargest(2).index.tolist()
    top_cat = df['category'].value_counts().index[0]
    
    dates = sorted(df["date"].unique())
    split_date = dates[-6] 
    
    train_df = df[df['date'] < split_date].dropna(subset=['value'] + FEATURES)
    model = lgb.LGBMRegressor(**BEST_PARAMS)
    model.fit(train_df[FEATURES], train_df['value'], categorical_feature=CAT_FEATURES)

    for terr_id in top_territories:
        df_plot = df[(df['territory_id'] == terr_id) & (df['category'] == top_cat)].dropna(subset=FEATURES).copy()
        
        train_local = df_plot[df_plot['date'] < split_date]
        test_local = df_plot[df_plot['date'] >= split_date]
        
        if len(test_local) == 0: continue
            
        preds = model.predict(test_local[FEATURES])
        
        plt.figure(figsize=(12, 6))
        plt.plot(train_local['date'], train_local['value'], color='#2c3e50', linewidth=2.5, marker='o', label='Обучение (Факт)')
        plt.plot(test_local['date'], test_local['value'], color='#27ae60', linewidth=2.5, marker='o', label='Тест (Факт)')
        plt.plot(test_local['date'], preds, color='#e67e22', linewidth=2.5, linestyle='--', marker='X', markersize=8, label='Прогноз (Базовая модель)')
        
        plt.axvline(x=train_local['date'].iloc[-1], color='gray', linestyle=':')
        plt.title(f'Качество базового прогноза (МО {terr_id})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        out_name = f'baseline_forecast_top_{terr_id}.png'
        plt.savefig(out_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" График базового прогноза сохранен: {out_name}")

if __name__ == "__main__":
    run_multi_horizon_forecasting()
    plot_presentation_forecasts()
