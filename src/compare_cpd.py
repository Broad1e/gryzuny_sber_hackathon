import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import ruptures as rpt
import lightgbm as lgb
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
import logging

warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

# =====================================================================
# НАСТРОЙКИ ЧУВСТВИТЕЛЬНОСТИ АЛГОРИТМОВ (Настройка гиперпараметров)
# Снижение порога увеличивает чувствительность алгоритма найдет соответствующий алгоритм.
# =====================================================================

# 1. ML-Residuals: Во сколько раз ошибка должна превышать стандартное отклонение.
# Рекомендуемое значение: 1.5. Меньшие значения приведут к гиперчувствительности.
ML_STD_MULTIPLIER = 0.4

# 2. Prophet: Доля от максимального излома тренда, которую мы считаем значимой.
# Рекомендуемое значение: 0.2.
PROPHET_SENSITIVITY = 0.2 

# 3. Ruptures (PELT): Множитель штрафа за добавление новой точки.
# Рекомендуемое значение: 1.5.
RUPTURES_PENALTY_MULT = 0.05

# =====================================================================

def compare_cpd_methods(data_dir):
    print("1. Загрузка данных...")
    file_path = os.path.join(data_dir, 'train_features_with_macro2.parquet')
    if not os.path.exists(file_path):
        print("Файл не найден. Сначала сгенерируй признаки.")
        return
        
    df = pd.read_parquet(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    for c in ['territory_id', 'category']:
        df[c] = df[c].astype('category')
    
    top_terr = df.groupby('territory_id')['value'].sum().idxmax()
    cat = 'Все категории' if 'Все категории' in df['category'].values else df['category'].unique()[0]
        
    print(f"Выбран тестовый ряд -> Муниципалитет: {top_terr}, Категория: {cat}")
    
    df_ts = df[(df['territory_id'] == top_terr) & (df['category'] == cat)].sort_values('date').reset_index(drop=True)
    
    if len(df_ts) < 10:
        print("Слишком мало данных для анализа этого ряда.")
        return

    # === МЕТОД 1: Prophet Changepoints ===
    print("\n2. Запуск Prophet Changepoints...")
    df_prophet = df_ts[['date', 'value']].rename(columns={'date': 'ds', 'value': 'y'})
    
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, changepoint_range=1.0)
    m.fit(df_prophet)
    
    deltas = np.abs(m.params['delta'][0])
    max_delta = np.max(deltas) if len(deltas) > 0 else 0
    threshold_prophet = max_delta * PROPHET_SENSITIVITY
    cp_prophet_idx = np.where(deltas > threshold_prophet)[0]
    prophet_dates = m.changepoints.iloc[cp_prophet_idx].tolist()
    
    # === МЕТОД 2: Ruptures (алгоритм PELT) ===
    print("3. Запуск Ruptures (PELT)...")
    y_array = df_ts['value'].values
    
    algo = rpt.Pelt(model="l1").fit(y_array)
    dynamic_penalty = np.std(y_array) * RUPTURES_PENALTY_MULT
    result_rpt = algo.predict(pen=dynamic_penalty)
    
    ruptures_dates = []
    for idx in result_rpt[:-1]: 
        if idx < len(df_ts):
            ruptures_dates.append(df_ts['date'].iloc[idx])
            
    # === МЕТОД 3: ML-Residuals ===
    print("4. Запуск ML-Residuals...")
    cat_features = ['territory_id', 'category']
    df_ml = df.copy()
        
    features = ['territory_id', 'category', 'month', 'year', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'market_access',
                'spatial_lag_1', 'cbr_key_rate', 'macro_news_sentiment', 'cbr_rate_diff', 'sentiment_diff']
    
    df_ml_clean = df_ml.dropna(subset=['lag_1', 'value'])
    X = df_ml_clean[features]
    y = df_ml_clean['value']
    
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X, y, categorical_feature=cat_features)
    
    df_ts_clean = df_ts.dropna(subset=['lag_1', 'value']).copy()
    df_ts_clean['pred'] = model.predict(df_ts_clean[features])
    df_ts_clean['residual'] = np.abs(df_ts_clean['value'] - df_ts_clean['pred'])
    
    valid_ts = df_ts_clean[df_ts_clean['date'] > '2023-03-01']
    
    mean_res = valid_ts['residual'].mean()
    std_res = valid_ts['residual'].std()
    threshold_ml = mean_res + ML_STD_MULTIPLIER * std_res
    
    lgbm_dates = valid_ts[valid_ts['residual'] > threshold_ml]['date'].tolist()

    # === ИТОГИ И ВИЗУАЛИЗАЦИЯ ===
    print("\n" + "="*40)
    print("ИТОГИ ПОИСКА ШОКОВ (Точек разладки)")
    print("="*40)
    print(f"Prophet нашел: {len(prophet_dates)} шт. -> {[d.strftime('%Y-%m') for d in prophet_dates]}")
    print(f"Ruptures нашел: {len(ruptures_dates)} шт. -> {[d.strftime('%Y-%m') for d in ruptures_dates]}")
    print(f"ML-Residuals нашел: {len(lgbm_dates)} шт. -> {[d.strftime('%Y-%m') for d in lgbm_dates]}")
    print("="*40)
    
    plt.figure(figsize=(16, 8))
    plt.plot(df_ts['date'], df_ts['value'], label='Фактические траты', color='black', linewidth=2.5)
    
    for i, d in enumerate(prophet_dates):
        plt.axvline(x=d, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Prophet' if i == 0 else "")
        
    for i, d in enumerate(ruptures_dates):
        plt.axvline(x=d, color='blue', linestyle='-.', alpha=0.7, linewidth=2, label='Ruptures (PELT)' if i == 0 else "")
        
    for i, d in enumerate(lgbm_dates):
        plt.axvline(x=d, color='green', linestyle=':', alpha=0.9, linewidth=3, label='ML-Residuals (Наш метод)' if i == 0 else "")
        
    plt.title(f"Сравнение алгоритмов поиска шоков (Муниципалитет: {top_terr}, {cat})", fontsize=16, fontweight='bold')
    plt.xlabel("Дата", fontsize=14)
    plt.ylabel("Объем потребления", fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.xticks(df_ts['date'], [d.strftime('%Y-%m') for d in df_ts['date']], rotation=45)
    plt.tight_layout()
    
    plot_path = os.path.join(data_dir, 'cpd_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nГрафик сохранен в: {plot_path}")

if __name__ == "__main__":
    
    compare_cpd_methods(DATA_DIRECTORY)
