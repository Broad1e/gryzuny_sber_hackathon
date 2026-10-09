import pandas as pd
import numpy as np
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
warnings.filterwarnings('ignore')

def prepare_features(data_dir):
    print("1. Загрузка исходных данных...")
    path_consumption = os.path.join(data_dir, 'consumption.parquet')
    path_market = os.path.join(data_dir, 'market_access.parquet')
    path_connections = os.path.join(data_dir, 'connection.parquet')
    
    # Проверка наличия файлов
    for p in [path_consumption, path_market, path_connections]:
        if not os.path.exists(p):
            print(f"ОШИБКА: Файл {p} не найден. Убедитесь, что данные лежат в этой папке.")
            return

    consumption = pd.read_parquet(path_consumption)
    market = pd.read_parquet(path_market)
    connections = pd.read_parquet(path_connections)

    print("2. Обработка времени и сортировка...")
    # Преобразуем date в формат datetime для корректной сортировки
    # Предполагаем формат дат наподобие '2023-01' или '2023-01-01'
    consumption['date'] = pd.to_datetime(consumption['date'])
    consumption = consumption.sort_values(by=['territory_id', 'category', 'date']).reset_index(drop=True)

    # Базовые временные признаки (сезонность)
    consumption['month'] = consumption['date'].dt.month
    consumption['year'] = consumption['date'].dt.year

    print("3. Генерация временных признаков (Лаги и скользящие средние)...")
    grouped = consumption.groupby(['territory_id', 'category'])
    
    # Формирование временных лагов для оценки локального тренда
    consumption['lag_1'] = grouped['value'].shift(1)
    consumption['lag_2'] = grouped['value'].shift(2)
    consumption['lag_3'] = grouped['value'].shift(3)

    # Скользящее среднее для сглаживания дисперсии
    # Используем lag_1, чтобы не использовать данные из будущего (data leakage)
    consumption['rolling_mean_3'] = grouped['lag_1'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    print("4. Добавление индекса доступности рынков (market_access)...")
    consumption = consumption.merge(market, on='territory_id', how='left')

    print("5. Генерация пространственных признаков (Влияние соседей)...")
    # Ограничение графа: выбор 5 ближайших соседей для каждого МО
    # Группируем связи (берем минимальную дистанцию между X и Y, неважно ж/д или авто)
    conn_simple = connections.groupby(['territory_id_x', 'territory_id_y'])['distance'].min().reset_index()
    conn_simple['rank'] = conn_simple.groupby('territory_id_x')['distance'].rank(method="dense", ascending=True)
    top_neighbors = conn_simple[conn_simple['rank'] <= 5]

    # Берем данные о потреблении соседей за прошлый месяц (lag_1)
    neighbor_data = consumption[['territory_id', 'date', 'category', 'lag_1']].dropna()
    neighbor_data = neighbor_data.rename(columns={'territory_id': 'territory_id_y', 'lag_1': 'neighbor_lag_1'})
    
    # Объединяем связи с данными соседей
    merged_neighbors = top_neighbors.merge(neighbor_data, on='territory_id_y')
    
    # Расчет взвешенной суммы с учетом расстояния
    merged_neighbors['weight'] = 1 / (merged_neighbors['distance'] + 1) # Сглаживание знаменателя для предотвращения деления на ноль
    merged_neighbors['weighted_val'] = merged_neighbors['neighbor_lag_1'] * merged_neighbors['weight']
    
    # Агрегируем результаты для каждого города X
    def w_avg(group):
        return group['weighted_val'].sum() / group['weight'].sum() if group['weight'].sum() != 0 else 0

    spatial_features = merged_neighbors.groupby(['territory_id_x', 'date', 'category']).apply(w_avg).reset_index(name='spatial_lag_1')
    spatial_features = spatial_features.rename(columns={'territory_id_x': 'territory_id'})

    print("6. Финальная сборка датасета...")
    final_df = consumption.merge(spatial_features, on=['territory_id', 'date', 'category'], how='left')

    # В первых месяцах лаги будут пустыми (NaN). Для базовой модели заполним их нулями.
    # В идеале (на будущее) их лучше заполнять средними, но 0 - надежный старт.
    final_df = final_df.fillna(0)

    # Удалим строки, где таргет (value) равен 0 или NaN (если такие есть), так как их нельзя предсказать
    final_df = final_df[final_df['value'] > 0]

    output_path = os.path.join(data_dir, 'train_features.parquet')
    final_df.to_parquet(output_path, index=False)
    
    print(f"\nУспешно. Датасет с признаками успешно создан и сохранен в:")
    print(output_path)
    print(f"Размер итоговой таблицы: {final_df.shape}")

if __name__ == "__main__":
    # Папка с данными
    
    prepare_features(DATA_DIRECTORY)
