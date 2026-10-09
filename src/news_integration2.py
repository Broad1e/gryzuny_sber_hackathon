from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "train_features_with_macro2.parquet"


def find_input(filename):
    for directory in (PROCESSED_DIR, RAW_DIR, ROOT):
        path = directory / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"Не найден {filename} в data/processed, data/raw или корне проекта")


def analyze_sentiment(text):
    positive_words = [
        "рост", "восстановление", "увеличение", "рекорд",
        "поддержка", "стабилизация", "успех", "прибыль",
    ]
    negative_words = [
        "инфляция", "санкции", "кризис", "падение", "дефицит",
        "угроза", "снижение", "риск", "замедление", "шок",
    ]
    text_lower = str(text).lower()
    score = 0
    for word in positive_words:
        score += text_lower.count(word)
    for word in negative_words:
        score -= text_lower.count(word)
    return score


def integrate_news_and_macro():
    macro_path = find_input("external_macro_data.csv")
    features_path = find_input("train_features.parquet")
    print("1. Загрузка макроэкономических данных...")
    df_macro = pd.read_csv(macro_path)
    df_macro["date"] = pd.to_datetime(df_macro["date"])
    print("2. Анализ тональности новостей...")
    df_macro["macro_news_sentiment"] = df_macro["news_text"].apply(analyze_sentiment)
    print("3. Загрузка основного датасета...")
    df_features = pd.read_parquet(features_path)
    df_features["date"] = pd.to_datetime(df_features["date"])
    print("4. Объединение данных...")
    macro_to_merge = df_macro[["date", "cbr_key_rate", "macro_news_sentiment"]]
    df_final = pd.merge(df_features, macro_to_merge, on="date", how="left")
    df_final = df_final.sort_values(["territory_id", "category", "date"]).reset_index(drop=True)
    grouped = df_final.groupby(["territory_id", "category"])
    df_final["cbr_rate_diff"] = grouped["cbr_key_rate"].diff().fillna(0)
    df_final["sentiment_diff"] = grouped["macro_news_sentiment"].diff().fillna(0)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_PATH, index=False)
    print(f"Готово. Файл сохранён: {OUTPUT_PATH}")
    return df_final


if __name__ == "__main__":
    integrate_news_and_macro()
