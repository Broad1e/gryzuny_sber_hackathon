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
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# Требуется: pip install chronos-forecasting torch transformers
import torch
from chronos import ChronosPipeline
from transformers import TimesFm2_5ModelForPrediction

  # ВНИМАНИЕ: укажите здесь свой путь к файлу consumption.parquet
HORIZONS = [1, 3, 6, 12]
SAMPLE_SIZE = 500  # Выборка для оценки, иначе расчет на CPU займет слишком много времени
CHRONOS_MODEL_NAME = "amazon/chronos-t5-small"
TIMESFM_MODEL_NAME = "google/timesfm-2.5-200m-transformers"


def run_evaluation():
    
    print("Оценка Foundation Models (Chronos и TimesFM в режиме Zero-Shot)")
    

    # Загружаем сырые данные (модели не нужны дополнительные признаки)
    consumption_path = os.path.join(DATA_DIRECTORY, "consumption.parquet")
    if not os.path.exists(consumption_path):
        consumption_path = os.path.join(DATA_DIRECTORY, "consumption.csv")

    print(f"\n1. Загрузка данных из {os.path.basename(consumption_path)}")
    if consumption_path.endswith(".parquet"):
        df = pd.read_parquet(consumption_path)
    else:
        df = pd.read_csv(consumption_path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["territory_id", "category", "date"]).reset_index(drop=True)

    value_col = "value" if "value" in df.columns else "consumption"

    print(f"   Всего строк: {len(df):,}")
    print(f"   Период: {df['date'].min().date()} — {df['date'].max().date()}")

    # Получаем уникальные ряды (муниципалитет + категория)
    keys = df.groupby(["territory_id", "category"]).size().reset_index()[["territory_id", "category"]]

    if SAMPLE_SIZE and SAMPLE_SIZE < len(keys):
        keys = keys.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"\n2. Случайная выборка: {SAMPLE_SIZE} рядов")
    else:
        print(f"\n2. Оценка по всем {len(keys)} рядам")

    print(f"\n3. Загрузка модели {CHRONOS_MODEL_NAME}")
    chronos_pipeline = ChronosPipeline.from_pretrained(
        CHRONOS_MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    
    print(f"4. Загрузка модели {TIMESFM_MODEL_NAME}")
    try:
        timesfm_model = TimesFm2_5ModelForPrediction.from_pretrained(
            TIMESFM_MODEL_NAME,
            device_map="cpu",
        )
    except Exception as e:
        print(f"   Ошибка загрузки TimesFM: {e}")
        timesfm_model = None

    chronos_results = {}
    timesfm_results = {}

    for h in HORIZONS:
        print(f"\n{'#' * 70}")
        print(f"Горизонт {h} мес.")
        print(f"{'#' * 70}")

        chr_true, chr_pred = [], []
        tfm_true, tfm_pred = [], []
        skipped = 0

        for idx, row in keys.iterrows():
            tid = row["territory_id"]
            cat = row["category"]

            ts = df[(df["territory_id"] == tid) & (df["category"] == cat)].sort_values("date")
            values = ts[value_col].values

            # Пропускаем ряды, в которых недостаточно данных для контекста
            if len(values) < h + 3:
                skipped += 1
                continue

            # Отделяем контекст от фактических значений для проверки
            context = values[:-h]
            actual = values[-h:] 

            # -- Chronos --
            context_tensor_chr = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            try:
                forecast_chr = chronos_pipeline.predict(
                    context_tensor_chr,
                    prediction_length=h,
                    num_samples=20,
                )
                median_pred_chr = forecast_chr.median(dim=1).values.squeeze().numpy()
                if h == 1:
                    median_pred_chr = np.atleast_1d(median_pred_chr)
                chr_true.append(actual[-1])
                chr_pred.append(median_pred_chr[-1])
            except Exception:
                pass

            # -- TimesFM --
            if timesfm_model is not None:
                context_tensor_tfm = torch.tensor(context, dtype=torch.float32, device=timesfm_model.device).unsqueeze(0)
                try:
                    with torch.no_grad():
                        outputs = timesfm_model(past_values=context_tensor_tfm, return_dict=True)
                    # Point forecasts (mean)
                    tfm_forecast = outputs.mean_predictions.cpu().squeeze().numpy()
                    if h == 1:
                        tfm_forecast = np.atleast_1d(tfm_forecast)
                    # TimesFM defaults to a horizon of 128 (or similar). We take the h-th element (index h-1)
                    tfm_true.append(actual[-1])
                    tfm_pred.append(tfm_forecast[h-1])
                except Exception:
                    pass

            if (idx + 1) % 100 == 0:
                print(f"   Обработано {idx + 1}/{len(keys)}")

        if len(chr_true) > 0:
            chr_true = np.array(chr_true)
            chr_pred = np.array(chr_pred)
            chronos_results[h] = {
                "mae": mean_absolute_error(chr_true, chr_pred),
                "r2": r2_score(chr_true, chr_pred)
            }
        
        if len(tfm_true) > 0:
            tfm_true = np.array(tfm_true)
            tfm_pred = np.array(tfm_pred)
            timesfm_results[h] = {
                "mae": mean_absolute_error(tfm_true, tfm_pred),
                "r2": r2_score(tfm_true, tfm_pred)
            }

        print(f"\n   Оценено рядов: {len(chr_true)}, пропущено: {skipped}")
        if h in chronos_results:
            print(f"   Chronos MAE: {chronos_results[h]['mae']:,.2f} руб.")
        if h in timesfm_results:
            print(f"   TimesFM MAE: {timesfm_results[h]['mae']:,.2f} руб.")

    # Сравнение с бейзлайном и основным решением
    
    print("Сравнение моделей (MAE, руб.)")
    
    print(f"{'Модель':<25}{'1 мес.':<15}{'3 мес.':<15}{'6 мес.':<15}{'12 мес.':<15}")
    
    print(f"{'Prophet (baseline)':<25}{'~900':<15}{'—':<15}{'—':<15}{'—':<15}")
    print(f"{'LightGBM (наш)':<25}{'399.52':<15}{'540.67':<15}{'540.27':<15}{'803.58':<15}")

    if chronos_results:
        chr_row = f"{'Chronos (Foundation)':<25}"
        for h in HORIZONS:
            if h in chronos_results:
                chr_row += f"{chronos_results[h]['mae']:<15,.2f}"
            else:
                chr_row += f"{'—':<15}"
        print(chr_row)

    if timesfm_results:
        tfm_row = f"{'TimesFM (Foundation)':<25}"
        for h in HORIZONS:
            if h in timesfm_results:
                tfm_row += f"{timesfm_results[h]['mae']:<15,.2f}"
            else:
                tfm_row += f"{'—':<15}"
        print(tfm_row)

    


if __name__ == "__main__":
    run_evaluation()
