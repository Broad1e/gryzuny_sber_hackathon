
from pathlib import Path
import re
import traceback

import numpy as np
import pandas as pd


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

ROOT = Path(__file__).resolve().parent

MACRO_FILE = ROOT / "external_macro_data.csv"
FEATURES_FILE = ROOT / "train_features.parquet"

OUTPUT_FILE = ROOT / "train_features_with_macro.parquet"
REPORT_FILE = ROOT / "news_rubert_monthly_report.csv"

MODEL_NAME = "cointegrated/rubert-tiny-sentiment-balanced"

# Новости за январь используются с февраля
NEWS_DELAY_MONTHS = 1

BATCH_SIZE = 8


# ============================================================
# 2. ЭКОНОМИЧЕСКИЕ ТЕМЫ
# ============================================================

TOPICS = {
    "inflation": [
        "инфляц",
        "подорож",
        "рост цен",
        "ценов",
    ],
    "credit": [
        "кредит",
        "ключев",
        "ставк",
        "ипотек",
    ],
    "demand": [
        "спрос",
        "потреблен",
        "рознич",
        "торговл",
        "маркетплейс",
    ],
    "supply": [
        "производств",
        "логист",
        "импорт",
        "санкц",
        "дефицит",
        "поставк",
    ],
}


def topic_present(text, keywords):
    """Проверка наличия экономической темы."""

    if pd.isna(text) or not str(text).strip():
        return np.nan

    normalized = str(text).lower().replace("ё", "е")

    return float(
        any(
            re.search(r"\b" + re.escape(k), normalized)
            for k in keywords
        )
    )


# ============================================================
# 3. АНАЛИЗ ТОНАЛЬНОСТИ ЧЕРЕЗ RuBERT
# ============================================================

def score_news_with_rubert(texts):

    try:
        import torch

        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

    except ImportError as exc:
        raise RuntimeError(
            "Установите библиотеки: "
            "pip install pandas numpy pyarrow torch transformers"
        ) from exc

    print("\nЗагрузка RuBERT...")
    print("При первом запуске нужен интернет.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    # Получаем реальные имена классов модели
    id2label = {
        int(k): str(v).lower()
        for k, v in model.config.id2label.items()
    }

    expected = {
        "positive",
        "neutral",
        "negative",
    }

    if set(id2label.values()) != expected:
        raise ValueError(
            f"Неожиданные классы RuBERT: {id2label}"
        )

    # Пропускаем пустые новости
    clean = [
        (i, str(text).strip())
        for i, text in enumerate(texts)
        if pd.notna(text) and str(text).strip()
    ]

    result = pd.DataFrame(
        np.nan,
        index=range(len(texts)),
        columns=[
            "nlp_negative",
            "nlp_neutral",
            "nlp_positive",
            "macro_news_sentiment",
        ],
    )

    # Обрабатываем новости пакетами
    for start in range(0, len(clean), BATCH_SIZE):

        batch = clean[start:start + BATCH_SIZE]

        batch_texts = [
            text for _, text in batch
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.inference_mode():

            logits = model(**encoded).logits

            # По методике из описания модели
            scores = torch.sigmoid(logits)

            scores = scores.cpu().numpy()

        for (row_idx, _), vector in zip(batch, scores):

            classes = {
                id2label[j]: float(vector[j])
                for j in range(len(vector))
            }

            negative = classes["negative"]
            neutral = classes["neutral"]
            positive = classes["positive"]

            # Непрерывная оценка тональности
            sentiment = positive - negative

            result.loc[row_idx] = [
                negative,
                neutral,
                positive,
                sentiment,
            ]

        processed = min(
            start + BATCH_SIZE,
            len(clean)
        )

        print(
            f"RuBERT: {processed}/{len(clean)}"
        )

    return result


# ============================================================
# 4. ПОДГОТОВКА МЕСЯЧНЫХ ПРИЗНАКОВ
# ============================================================

def build_monthly_features(macro, feature_months):

    required = {
        "date",
        "news_text",
        "cbr_key_rate",
    }

    missing = required - set(macro.columns)

    if missing:
        raise ValueError(
            f"Отсутствуют колонки: {missing}"
        )

    macro = macro.copy()

    macro["date"] = pd.to_datetime(
        macro["date"],
        errors="raise",
    )

    macro["month_key"] = (
        macro["date"].dt.to_period("M")
    )

    # За один месяц должна быть одна сводка
    if macro["month_key"].duplicated().any():
        raise ValueError(
            "Найдены повторяющиеся месяцы. "
            "Сначала необходимо агрегировать новости."
        )

    if macro.empty:
        raise ValueError(
            "Файл с новостями пуст."
        )

    print("\nПериод новостей:")
    print(
        macro["month_key"].min(),
        "-",
        macro["month_key"].max()
    )

    # --------------------------------------------------------
    # NLP
    # --------------------------------------------------------

    print("\nАнализ тональности...")

    nlp = score_news_with_rubert(
        macro["news_text"].tolist()
    )

    for column in nlp.columns:
        macro[column] = nlp[column].to_numpy()

    # --------------------------------------------------------
    # Тематики
    # --------------------------------------------------------

    print("\nВыделение экономических тем...")

    for name, keywords in TOPICS.items():

        macro[f"news_topic_{name}"] = (
            macro["news_text"].apply(
                lambda text: topic_present(
                    text,
                    keywords,
                )
            )
        )

    macro["cbr_key_rate"] = pd.to_numeric(
        macro["cbr_key_rate"],
        errors="raise",
    )

    # --------------------------------------------------------
    # Полный календарь
    # --------------------------------------------------------

    # Важно: лаг должен означать календарный месяц,
    # даже если в исходном файле пропущены некоторые даты.

    all_months = pd.period_range(
        start=min(
            macro["month_key"].min(),
            feature_months.min(),
        ),
        end=max(
            macro["month_key"].max(),
            feature_months.max(),
        ),
        freq="M",
    )

    monthly = (
        macro.set_index("month_key")
        .reindex(all_months)
    )

    features = pd.DataFrame(
        index=all_months
    )

    shift = NEWS_DELAY_MONTHS

    # --------------------------------------------------------
    # Тональность за предыдущий месяц
    # --------------------------------------------------------

    features["macro_news_sentiment"] = (
        monthly["macro_news_sentiment"]
        .shift(shift)
    )

    # Оценки отдельных классов
    for column in [
        "nlp_negative",
        "nlp_neutral",
        "nlp_positive",
    ]:

        features[f"{column}_asof"] = (
            monthly[column].shift(shift)
        )

    # --------------------------------------------------------
    # Изменение новостного фона
    # --------------------------------------------------------

    sentiment = monthly["macro_news_sentiment"]

    features["sentiment_diff"] = (
        sentiment.diff().shift(shift)
    )

    # Средняя тональность за последние 3 доступных месяца
    features["sentiment_rolling_3"] = (
        sentiment.shift(shift)
        .rolling(
            window=3,
            min_periods=1,
        )
        .mean()
    )

    # --------------------------------------------------------
    # Ключевая ставка ЦБ
    # --------------------------------------------------------

    rate = monthly["cbr_key_rate"].ffill()

    features["cbr_key_rate"] = (
        rate.shift(shift)
    )

    features["cbr_rate_diff"] = (
        rate.diff().shift(shift)
    )

    # --------------------------------------------------------
    # Тематические признаки
    # --------------------------------------------------------

    for topic in TOPICS:

        column = f"news_topic_{topic}"

        features[column] = (
            monthly[column].shift(shift)
        )

    # --------------------------------------------------------
    # Флаги пропусков
    # --------------------------------------------------------

    features["news_missing"] = (
        features["macro_news_sentiment"]
        .isna()
        .astype("int8")
    )

    features["rate_missing"] = (
        features["cbr_key_rate"]
        .isna()
        .astype("int8")
    )

    # Отсутствующие новости оставляем NaN.
    # LightGBM умеет работать с пропусками.

    features.index.name = "month_key"

    # Отдельный отчёт для проверки результатов RuBERT
    report_columns = [
        "date",
        "news_text",
        "macro_news_sentiment",
        "nlp_negative",
        "nlp_neutral",
        "nlp_positive",
        "cbr_key_rate",
    ]

    report = (
        macro[report_columns]
        .sort_values("date")
        .reset_index(drop=True)
    )

    return features.reset_index(), report


# ============================================================
# 5. ОБЪЕДИНЕНИЕ С ДАННЫМИ СБЕРИНДЕКСА
# ============================================================

def main():

    print("=" * 60)
    print("ИНТЕГРАЦИЯ НОВОСТЕЙ С ПОМОЩЬЮ RuBERT")
    print("=" * 60)

    # Проверяем файлы
    for path in [MACRO_FILE, FEATURES_FILE]:

        if not path.is_file():
            raise FileNotFoundError(
                f"Не найден {path.name} в папке {ROOT}"
            )

    # --------------------------------------------------------
    # Загрузка
    # --------------------------------------------------------

    print("\n1. Загрузка данных...")

    macro = pd.read_csv(MACRO_FILE)
    df = pd.read_parquet(FEATURES_FILE)

    if "date" not in df.columns:
        raise ValueError(
            "В train_features.parquet отсутствует date"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    df["month_key"] = (
        df["date"].dt.to_period("M")
    )

    original_rows = len(df)

    print(
        f"Строк в исходном датасете: {original_rows:,}"
    )

    # --------------------------------------------------------
    # Макроэкономические признаки
    # --------------------------------------------------------

    print("\n2. Генерация новостных признаков...")

    monthly_features, report = build_monthly_features(
        macro,
        df["month_key"],
    )

    # Защита от повторного добавления колонок
    new_columns = (
        set(monthly_features.columns) - {"month_key"}
    )

    duplicates = new_columns & set(df.columns)

    if duplicates:
        raise ValueError(
            f"Во входном файле уже есть признаки: "
            f"{sorted(duplicates)}"
        )

    # --------------------------------------------------------
    # Объединение
    # --------------------------------------------------------

    print("\n3. Объединение по календарному месяцу...")

    result = df.merge(
        monthly_features,
        how="left",
        on="month_key",
        validate="many_to_one",
    )

    if len(result) != original_rows:
        raise AssertionError(
            "После объединения изменилось количество строк!"
        )

    result = result.drop(
        columns=["month_key"]
    )

    result = result.sort_values(
        [
            "territory_id",
            "category",
            "date",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Сохранение
    # --------------------------------------------------------

    print("\n4. Сохранение...")

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    report.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)

    print("Файл:", OUTPUT_FILE.name)
    print("Строк:", len(result))
    print("Столбцов:", len(result.columns))
    print("Отчёт RuBERT:", REPORT_FILE.name)

    print(
        "\nПримечание: новостные признаки "
        "используют лаг 1 месяц."
    )


# ============================================================
# 6. ЗАПУСК БЕЗ АРГУМЕНТОВ
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print("\nОШИБКА:", error)
        traceback.print_exc()

    finally:
        try:
            input("\nНажмите Enter для завершения...")
        except EOFError:
            pass
