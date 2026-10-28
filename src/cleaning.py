# src/cleaning.py
import logging
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def clean_column_names(df):
    """1. Приводим имена столбцов к нижнему регистру и маппим колонки geonames."""
    new_cols = []
    for col in df.columns:
        c = col.strip().lower()
        c = c.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        c = c.replace("²", "2").replace("вІ", "2")
        new_cols.append(c)
    df.columns = new_cols

    # Если на вход подан geonames.csv, переименовываем его столбцы под единый стандарт
    geonames_mapping = {
        "geonameid": "id",
        "name": "city",
        "asciiname": "city_ascii",
        "latitude": "lat",
        "longitude": "lng",
        "country_code": "iso2",
        "admin1_code": "admin_name",
        "feature_code": "capital",
    }
    df = df.rename(
        columns={k: v for k, v in geonames_mapping.items() if k in df.columns}
    )

    log.info("Имена столбцов приведены к единому виду: %s", list(df.columns))
    return df


def join_datasets(cities_df, pop_df, worldcities_path="data/raw/worldcities.csv"):
    """2. Соединяем таблицу городов и справочник стран по коду ISO3 / ISO2."""
    before_len = len(cities_df)
    pop_df["cca3"] = pop_df["cca3"].astype(str).str.strip().str.upper()
    pop_df = pop_df.drop_duplicates(subset=["cca3"])

    # Если в geonames есть iso2, маппим его в iso3 через worldcities.csv
    if "iso3" not in cities_df.columns and "iso2" in cities_df.columns:
        cities_df["iso2"] = cities_df["iso2"].astype(str).str.strip().str.upper()

        # Читаем через безопасное автоопределение разделителя
        with open(worldcities_path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        sep = (
            ";"
            if ";" in first_line and first_line.count(";") >= first_line.count(",")
            else ","
        )

        wc = pd.read_csv(worldcities_path, sep=sep, encoding="utf-8", low_memory=False)
        wc.columns = [c.strip().lower() for c in wc.columns]

        col_iso2 = next((c for c in wc.columns if "iso2" in c), None)
        col_iso3 = next((c for c in wc.columns if "iso3" in c), None)

        if col_iso2 and col_iso3:
            wc[col_iso2] = wc[col_iso2].astype(str).str.strip().str.upper()
            wc[col_iso3] = wc[col_iso3].astype(str).str.strip().str.upper()
            iso_map = wc.set_index(col_iso2)[col_iso3].to_dict()
            cities_df["iso3"] = cities_df["iso2"].map(iso_map)

    if "iso3" in cities_df.columns:
        cities_df["iso3"] = cities_df["iso3"].astype(str).str.strip().str.upper()
        merged_df = pd.merge(
            cities_df, pop_df, left_on="iso3", right_on="cca3", how="left"
        )
    else:
        merged_df = cities_df.copy()
        merged_df["cca3"] = None

    if "country" not in merged_df.columns:
        for cand in ["country_territory", "country_name", "name_y"]:
            if cand in merged_df.columns:
                merged_df = merged_df.rename(columns={cand: "country"})
                break

    after_len = len(merged_df)
    unmatched = merged_df["cca3"].isna().sum() if "cca3" in merged_df.columns else 0

    log.info(
        "Соединение таблиц: было %d строк, стало %d строк. Не сопоставилось: %d записей",
        before_len,
        after_len,
        unmatched,
    )
    return merged_df


def convert_types(df):
    """3. Явное приведение типов данных с экономией памяти (float32)."""
    num_cols = [
        "lat",
        "lng",
        "population",
        "2022_population",
        "area_km2",
        "growth_rate",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype("int64")

    log.info("Приведение числовых типов завершено. Строк: %d", len(df))
    return df


def replace_hidden_missing(df):
    """4. Замена скрытых пропусков на NaN."""
    hidden_values = ["", "-", "нет данных", "none", "null", "nan", "N/A"]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].replace(hidden_values, np.nan)

    if "population" in df.columns:
        zero_count = (df["population"] == 0).sum()
        df.loc[df["population"] == 0, "population"] = np.nan
        log.info("Заменено нулевых значений в population на NaN: %d", zero_count)

    return df


def handle_missing_values(df):
    """5. Обработка пропусков: заполнение медианой и 'unknown'."""
    before_len = len(df)
    df = df.dropna(subset=["city", "lat", "lng"]).copy()

    if "capital" in df.columns:
        df["capital"] = df["capital"].astype(str).fillna("not_capital")

    if "admin_name" in df.columns:
        df["admin_name"] = df["admin_name"].fillna("unknown")

    if "population" in df.columns:
        df["is_population_missing"] = df["population"].isna()
        median_pop = float(df["population"].median())
        if np.isnan(median_pop):
            median_pop = 1000.0
        df["population"] = df["population"].fillna(median_pop)

    log.info(
        "Обработка пропусков: было %d строк, стало %d (удалено %d)",
        before_len,
        len(df),
        before_len - len(df),
    )
    return df


def remove_duplicates(df):
    """6. Удаление полных дубликатов и дубликатов по ключу id."""
    before_len = len(df)
    df = df.drop_duplicates()
    after_full = len(df)

    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="last")
    after_key = len(df)

    log.info(
        "Дубликаты: было %d, после полных %d, после ключа id %d (удалено всего %d)",
        before_len,
        after_full,
        after_key,
        before_len - after_key,
    )
    return df


def normalize_categories_and_text(df):
    """7-8. Нормализация категорий и очистка текстовых полей."""
    text_cols = ["city", "city_ascii", "country", "admin_name", "capital", "continent"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "capital" in df.columns:
        # Для geonames коды PPLC — столица (primary), PPLA — админ. центр (admin)
        cap_map = {
            "pplc": "primary",
            "ppla": "admin",
            "ppla2": "minor",
            "primary": "primary",
            "admin": "admin",
            "minor": "minor",
            "not_capital": "not_capital",
        }
        df["capital"] = (
            df["capital"]
            .str.lower()
            .map(lambda x: cap_map.get(x, "not_capital"))
            .astype("category")
        )

    if "continent" in df.columns:
        df["continent"] = (
            df["continent"]
            .replace("nan", "Unknown")
            .fillna("Unknown")
            .astype("category")
        )

    log.info("Нормализация текста и категорий выполнена. Строк: %d", len(df))
    return df


def check_business_rules(df):
    """9-10. Проверка предметных ограничений (широта [-90, 90], долгота [-180, 180], население > 0)."""
    before_len = len(df)
    cond_lat = (df["lat"] >= -90) & (df["lat"] <= 90)
    cond_lng = (df["lng"] >= -180) & (df["lng"] <= 180)
    cond_pop = df["population"] > 0

    df = df[cond_lat & cond_lng & cond_pop].copy()
    log.info(
        "Проверка диапазонов: было %d, стало %d (удалено некорректных: %d)",
        before_len,
        len(df),
        before_len - len(df),
    )
    return df


def detect_outliers(df):
    """11. Выявление выбросов двумя способами (IQR и правило 3 сигм)."""
    pop = df["population"]
    q1 = pop.quantile(0.25)
    q3 = pop.quantile(0.75)
    iqr = q3 - q1
    upper_iqr = q3 + 1.5 * iqr
    outliers_iqr = (pop > upper_iqr).sum()

    mean_val = pop.mean()
    std_val = pop.std()
    upper_3sigma = mean_val + 3 * std_val
    outliers_3sigma = (pop > upper_3sigma).sum()

    df["is_megacity_outlier"] = pop > upper_3sigma
    log.info(
        "Выбросы по population: метод IQR нашел %d, метод 3 сигм нашел %d.",
        outliers_iqr,
        outliers_3sigma,
    )
    return df
