from pathlib import Path
import random
import warnings
import logging

import numpy as np
import pandas as pd
import torch

from chronos import ChronosBoltPipeline
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)

FILE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_features_with_macro2.parquet"

HORIZONS = [1, 3, 6, 12]

GROUP_COLS = ["territory_id", "category"]

SAMPLE_SIZE = 1000
RANDOM_STATE = 42

MIN_TRAIN_SIZE = 6
N_VALIDATION_FOLDS = 3

BATCH_SIZE = 32

CHRONOS_PARAMS = {
    "model_name": "amazon/chronos-bolt-small",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "torch_dtype": torch.float32
}


def prepare_chronos_context(
    train_group,
    target_date
):
    history = train_group[
        ["date", "value"]
    ].copy()
    history["date"] = pd.to_datetime(
        history["date"]
    )
    history["value"] = pd.to_numeric(
        history["value"],
        errors="coerce"
    )
    history = (
        history
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["date", "value"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )
    if history.empty:
        return None
    history["date"] = (
        history["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    history = (
        history
        .drop_duplicates("date", keep="last")
        .set_index("date")
        .sort_index()
    )
    target_date = pd.Timestamp(
        target_date
    ).to_period("M").to_timestamp()
    history = history[
        history.index < target_date
    ]
    if history.empty:
        return None
    monthly_index = pd.date_range(
        start=history.index.min(),
        end=history.index.max(),
        freq="MS"
    )
    series = (
        history["value"]
        .reindex(monthly_index)
        .astype(float)
    )
    if series.notna().sum() < MIN_TRAIN_SIZE:
        return None
    last_date = series.index.max()
    prediction_length = (
        (target_date.year - last_date.year) * 12
        + (target_date.month - last_date.month)
    )
    if prediction_length < 1:
        return None
    context = torch.tensor(
        series.values,
        dtype=torch.float32
    )
    return {
        "context": context,
        "prediction_length": prediction_length,
        "last_date": last_date
    }


def predict_chronos_batch(
    pipeline,
    batch_items
):
    if not batch_items:
        return []
    contexts = [
        item["context"]
        for item in batch_items
    ]
    max_prediction_length = max(
        item["prediction_length"]
        for item in batch_items
    )
    with torch.no_grad():
        predictions = pipeline.predict(
            contexts,
            prediction_length=max_prediction_length
        )
    median_predictions = (
        predictions[:, 4, :]
        .cpu()
        .numpy()
    )
    results = []
    for i, item in enumerate(batch_items):
        step = item["prediction_length"] - 1
        y_pred = float(
            median_predictions[i, step]
        )
        results.append(y_pred)
    return results


def run_multi_horizon_chronos():
    print("=" * 70)
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("=" * 70)
    if not FILE_PATH.is_file():
        print(
            f"ОШИБКА: файл {FILE_PATH} не найден."
        )
        return
    df = pd.read_parquet(FILE_PATH)
    df["date"] = pd.to_datetime(
        df["date"]
    )
    if "consumption" in df.columns and "value" not in df.columns:
        df = df.rename(
            columns={"consumption": "value"}
        )
    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )
    df = (
        df.sort_values(
            GROUP_COLS + ["date"]
        )
        .reset_index(drop=True)
    )
    print(f"Количество строк: {len(df):,}")
    print(
        f"Период: "
        f"{df['date'].min().date()} — "
        f"{df['date'].max().date()}"
    )
    print("\n" + "=" * 70)
    print("2. ПОДГОТОВКА ВРЕМЕННЫХ РЯДОВ")
    print("=" * 70)
    all_groups = list(
        df.groupby(
            GROUP_COLS,
            observed=True
        ).groups.keys()
    )
    random.seed(RANDOM_STATE)
    if SAMPLE_SIZE is None:
        selected_groups = all_groups
    else:
        selected_groups = random.sample(
            all_groups,
            min(SAMPLE_SIZE, len(all_groups))
        )
    selected_index = pd.MultiIndex.from_tuples(
        selected_groups,
        names=GROUP_COLS
    )
    df_index = pd.MultiIndex.from_frame(
        df[GROUP_COLS]
    )
    df = df[
        df_index.isin(selected_index)
    ].copy()
    print(
        f"Выбрано временных рядов: "
        f"{len(selected_groups):,}"
    )
    print(
        f"Количество строк после выборки: "
        f"{len(df):,}"
    )
    print("\n" + "=" * 70)
    print("3. СОЗДАНИЕ TARGET ДЛЯ 1, 3, 6 И 12 МЕСЯЦЕВ")
    print("=" * 70)
    grouped = df.groupby(
        GROUP_COLS,
        observed=True
    )
    for h in HORIZONS:
        df[f"target_{h}"] = (
            grouped["value"].shift(-h)
        )
        df[f"target_date_{h}"] = (
            grouped["date"].shift(-h)
        )
        print(
            f"Горизонт {h:2d} мес. создан."
        )
    print("\n" + "=" * 70)
    print("4. ЗАГРУЗКА CHRONOS")
    print("=" * 70)
    print(
        f"Модель: {CHRONOS_PARAMS['model_name']}"
    )
    print(
        f"Устройство: {CHRONOS_PARAMS['device']}"
    )
    print("Загрузка предобученной модели...")
    pipeline = ChronosBoltPipeline.from_pretrained(
        CHRONOS_PARAMS["model_name"],
        device_map=CHRONOS_PARAMS["device"],
        torch_dtype=CHRONOS_PARAMS["torch_dtype"]
    )
    print("Chronos успешно загружен.")
    print("\n" + "=" * 70)
    print("5. ПРОГНОЗИРОВАНИЕ CHRONOS ZERO-SHOT")
    print("=" * 70)
    results = {}
    all_predictions = []
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
        dates = sorted(
            df_h["date"].unique()
        )
        val_dates = dates[-N_VALIDATION_FOLDS:]
        fold_maes = []
        fold_r2 = []
        for fold_number, val_date in enumerate(
            val_dates,
            start=1
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
                    "нет данных."
                )
                continue
            print("\n" + "-" * 60)
            print(f"Fold {fold_number}")
            print(f"Validation date: {val_date.date()}")
            print(f"Режим: {validation_mode}")
            print(f"Train: {len(train_df):,}")
            print(f"Validation: {len(val_df):,}")
            train_groups = {
                key: group
                for key, group in train_df.groupby(
                    GROUP_COLS,
                    observed=True
                )
            }
            val_groups = list(
                val_df.groupby(
                    GROUP_COLS,
                    observed=True
                )
            )
            prepared_items = []
            for key, val_group in val_groups:
                territory_id, category = key
                train_group = train_groups.get(key)
                if train_group is None:
                    continue
                for _, val_row in val_group.iterrows():
                    target_date = val_row[target_date_col]
                    y_true = val_row[target_col]
                    if pd.isna(target_date) or pd.isna(y_true):
                        continue
                    prepared = prepare_chronos_context(
                        train_group,
                        target_date
                    )
                    if prepared is None:
                        continue
                    prepared_items.append({
                        "territory_id": territory_id,
                        "category": category,
                        "target_date": target_date,
                        "y_true": float(y_true),
                        "context": prepared["context"],
                        "prediction_length": prepared[
                            "prediction_length"
                        ]
                    })
            print(
                f"Подготовлено рядов: "
                f"{len(prepared_items):,}"
            )
            if not prepared_items:
                print(
                    "Нет подходящих рядов для прогноза."
                )
                continue
            fold_predictions = []
            for batch_start in tqdm(
                range(0, len(prepared_items), BATCH_SIZE),
                desc=f"Chronos | {h} мес. | Fold {fold_number}"
            ):
                batch_items = prepared_items[
                    batch_start:batch_start + BATCH_SIZE
                ]
                try:
                    batch_preds = predict_chronos_batch(
                        pipeline,
                        batch_items
                    )
                except Exception as e:
                    print(
                        f"\nОшибка batch: {e}"
                    )
                    continue
                for item, y_pred in zip(
                    batch_items,
                    batch_preds
                ):
                    fold_predictions.append({
                        "territory_id": item["territory_id"],
                        "category": item["category"],
                        "ds": item["target_date"],
                        "y_true": item["y_true"],
                        "y_pred": y_pred,
                        "horizon": h,
                        "fold": fold_number,
                        "validation_mode": validation_mode,
                        "validation_date": val_date
                    })
            if not fold_predictions:
                print(
                    "Не удалось получить прогнозы."
                )
                continue
            fold_result = pd.DataFrame(
                fold_predictions
            )
            fold_result = (
                fold_result
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["y_true", "y_pred"])
            )
            if fold_result.empty:
                continue
            all_predictions.append(fold_result)
            y_true = fold_result["y_true"]
            y_pred = fold_result["y_pred"]
            mae = mean_absolute_error(
                y_true,
                y_pred
            )
            if len(y_true) >= 2:
                r2 = r2_score(
                    y_true,
                    y_pred
                )
            else:
                r2 = np.nan
            fold_maes.append(mae)
            fold_r2.append(r2)
            print(
                f"\nУспешно спрогнозировано рядов: "
                f"{len(fold_result):,}"
            )
            print(f"MAE: {mae:,.2f} руб.")
            print(f"R²: {r2:.4f}")
        if not fold_maes:
            print(
                f"\nДля горизонта {h} мес. "
                "нет допустимых фолдов."
            )
            continue
        mean_mae = np.mean(fold_maes)
        mean_r2 = (
            np.nanmean(fold_r2)
            if np.isfinite(fold_r2).any()
            else np.nan
        )
        results[h] = {
            "mae": mean_mae,
            "r2": mean_r2,
            "folds": len(fold_maes)
        }
        print("\n" + "-" * 60)
        print(f"ИТОГ ДЛЯ {h} МЕС.")
        print(f"Средний MAE: {mean_mae:,.2f} руб.")
        print(f"Средний R²: {mean_r2:.4f}")
        print(f"Количество фолдов: {len(fold_maes)}")
        print("-" * 60)
    print("\n" + "=" * 85)
    print("🏆 ИТОГОВЫЕ МЕТРИКИ CHRONOS ZERO-SHOT")
    print("=" * 85)
    print(
        f"{'Горизонт':<15}"
        f"{'MAE, руб.':<25}"
        f"{'R²':<15}"
        f"{'Фолдов':<15}"
    )
    print("-" * 85)
    for h in HORIZONS:
        if h not in results:
            continue
        print(
            f"{str(h) + ' мес.':<15}"
            f"{results[h]['mae']:<25,.2f}"
            f"{results[h]['r2']:<15.4f}"
            f"{results[h]['folds']:<15}"
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
    if all_predictions:
        final_predictions = pd.concat(
            all_predictions,
            ignore_index=True
        )
        output_file = Path(__file__).resolve().parents[1] / "data" / "processed" / "chronos_multi_horizon_predictions.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_predictions.to_csv(
            output_file,
            index=False
        )
        print(
            f"\nПрогнозы сохранены: {output_file}"
        )
        print(
            f"Всего прогнозов: "
            f"{len(final_predictions):,}"
        )
    return results


if __name__ == "__main__":
    run_multi_horizon_chronos()
