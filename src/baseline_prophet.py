import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score
import os
import warnings
from tqdm import tqdm
import logging

warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

def train_predict_prophet_horizons(group_data):
    territory_id, category, df_group = group_data
    
    df_prophet = df_group[['date', 'value']].rename(columns={'date': 'ds', 'value': 'y'})
    
    # Жесткий сплит: обучаемся на 2023 годе, тестируем на 2024 (чтобы иметь горизонт 12 мес)
    cutoff_date = '2023-12-01'
    train_df = df_prophet[df_prophet['ds'] <= cutoff_date]
    test_df = df_prophet[df_prophet['ds'] > cutoff_date]
    
    # Нам нужно, чтобы в тесте было достаточно месяцев для горизонта 12
    if len(train_df) < 6 or len(test_df) < 12:
        return None
        
    m = Prophet(
        yearly_seasonality=False, 
        weekly_seasonality=False, 
        daily_seasonality=False,
    )
    
    try:
        m.fit(train_df)
        
        # Строим прогноз на даты из тестовой выборки (весь 2024 год)
        future = test_df[['ds']]
        forecast = m.predict(future)
        
        result = pd.DataFrame({
            'territory_id': territory_id,
            'category': category,
            'ds': test_df['ds'].values,
            'y_true': test_df['y'].values,
            'y_pred': forecast['yhat'].values
        })
        return result
    except Exception:
        return None

def run_prophet_horizons(sample_size=1000):
    print("1. Загрузка исходных данных...")
    path_consumption = os.path.join('consumption.parquet')
    df = pd.read_parquet(path_consumption)
    df['date'] = pd.to_datetime(df['date'])
    
    if 'consumption' in df.columns and 'value' not in df.columns:
        df = df.rename(columns={'consumption': 'value'})
        
    print("2. Подготовка независимых временных рядов...")
    groups = []
    for (terr, cat), group in df.groupby(['territory_id', 'category']):
        groups.append((terr, cat, group))
        
    import random
    random.seed(42)
    groups = random.sample(groups, min(sample_size, len(groups)))
    print(f"ℹ️ Выбрано {len(groups)} случайных рядов для оценки (Baseline).")

    print(f"3. Обучение моделей Prophet (без мультипроцессинга)...")
    results = []
    
    # Обучаем последовательно
    for g in tqdm(groups, desc="Обучение Prophet", total=len(groups)):
        res = train_predict_prophet_horizons(g)
        if res is not None:
            results.append(res)
                
    print("\n4. Оценка качества на горизонтах 1, 3, 6, 12 месяцев...")
    if not results:
        print("Не удалось получить прогнозы.")
        return
        
    final_results = pd.concat(results, ignore_index=True)
    final_results = final_results.dropna()
    
    # Так как мы учились до Декабря 2023, горизонты четко ложатся на 2024 год:
    horizon_mapping = {
        1: '2024-01-01',
        3: '2024-03-01',
        6: '2024-06-01',
        12: '2024-12-01'
    }
    
    print("\n" + "="*50)
    print("🏆 ИТОГОВЫЕ МЕТРИКИ PROPHET ПО ГОРИЗОНТАМ")
    print("="*50)
    
    for h, date_str in horizon_mapping.items():
        # Берем только предсказания для конкретного горизонта (конкретного месяца)
        h_df = final_results[final_results['ds'] == date_str]
        
        if not h_df.empty:
            mae = mean_absolute_error(h_df['y_true'], h_df['y_pred'])
            r2 = r2_score(h_df['y_true'], h_df['y_pred'])
            print(f"Горизонт {h:2d} мес. (Дата: {date_str[:7]}) | MAE: {mae:,.2f} руб. | R^2: {r2:.4f}")
        else:
            print(f"Горизонт {h:2d} мес. | Нет данных для оценки")
            
    print("="*50 + "\n")

if __name__ == "__main__":
    run_prophet_horizons()
