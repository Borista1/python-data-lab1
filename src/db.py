import os
import time
import sqlite3
import pandas as pd
import pyarrow.parquet as pq
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()


def prepare_3nf_tables(df):
    """Разбиваем батч датафрейма на 3 таблицы в 3-й нормальной форме (3НФ)."""
    if "continent" not in df.columns:
        df["continent"] = "Unknown"
    else:
        df["continent"] = df["continent"].fillna("Unknown")

    # 1. Таблица континентов
    continents = df[["continent"]].drop_duplicates().dropna().reset_index(drop=True)

    continent_map = {
        "Asia": 1,
        "Europe": 2,
        "Africa": 3,
        "North America": 4,
        "South America": 5,
        "Oceania": 6,
        "Antarctica": 7,
        "Unknown": 0,
    }

    def get_continent_id(name):
        return continent_map.get(str(name), abs(hash(str(name))) % 900 + 100)

    continents["continent_id"] = continents["continent"].map(get_continent_id)
    continents = continents.rename(columns={"continent": "continent_name"})
    continents = continents[["continent_id", "continent_name"]].drop_duplicates(
        subset=["continent_id"]
    )

    # ПРИНУДИТЕЛЬНО добавляем системный континент Unknown (ID = 0) в КАЖДЫЙ батч без исключений
    unknown_df = pd.DataFrame({"continent_id": [0], "continent_name": ["Unknown"]})
    continents = pd.concat([unknown_df, continents], ignore_index=True).drop_duplicates(
        subset=["continent_id"]
    )

    df_merged = pd.merge(
        df, continents, left_on="continent", right_on="continent_name", how="left"
    )
    df_merged["continent_id"] = df_merged["continent_id"].fillna(0).astype("int64")

    # 2. Таблица стран
    countries = (
        df_merged[
            ["iso3", "iso2", "country", "continent_id", "2022_population", "area_km2"]
        ]
        .drop_duplicates(subset=["iso3"])
        .copy()
    )
    countries = countries.rename(
        columns={"country": "country_name", "2022_population": "population_2022"}
    )
    countries["iso3"] = countries["iso3"].fillna("UNK")
    countries["country_name"] = countries["country_name"].fillna("Unknown Country")

    # 3. Таблица городов
    city_cap_col = "capital_x" if "capital_x" in df_merged.columns else "capital"
    if city_cap_col not in df_merged.columns:
        df_merged["capital_status"] = "not_capital"
        city_cap_col = "capital_status"

    cities = df_merged[
        ["id", "city", "lat", "lng", city_cap_col, "population", "iso3"]
    ].copy()
    cities = cities.rename(
        columns={"id": "city_id", "city": "city_name", city_cap_col: "capital_status"}
    )
    cities["iso3"] = cities["iso3"].fillna("UNK")

    return continents, countries, cities


def load_to_sqlite(
    clean_parquet_path="data/processed/clean.parquet",
    sqlite_path="data/processed/dataset.db",
    batch_size=100_000,
):
    """Потоковая загрузка больших данных в SQLite в 3НФ батчами с игнорированием дубликатов."""
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    try:
        cursor.executescript("""
            DROP TABLE IF EXISTS cities;
            DROP TABLE IF EXISTS countries;
            DROP TABLE IF EXISTS continents;

            CREATE TABLE continents (
                continent_id INTEGER PRIMARY KEY,
                continent_name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE countries (
                iso3 TEXT PRIMARY KEY,
                iso2 TEXT,
                country_name TEXT NOT NULL,
                continent_id INTEGER,
                population_2022 REAL,
                area_km2 REAL,
                FOREIGN KEY (continent_id) REFERENCES continents(continent_id)
            );
            CREATE TABLE cities (
                city_id INTEGER PRIMARY KEY,
                city_name TEXT NOT NULL,
                lat REAL,
                lng REAL,
                capital_status TEXT,
                population REAL,
                iso3 TEXT,
                FOREIGN KEY (iso3) REFERENCES countries(iso3)
            );
        """)

        # Гарантированно добавляем континент Unknown (ID = 0)
        cursor.execute(
            "INSERT OR IGNORE INTO continents (continent_id, continent_name) VALUES (?, ?)",
            (0, "Unknown"),
        )
        conn.commit()

        parquet_file = pq.ParquetFile(clean_parquet_path)
        total_rows = parquet_file.metadata.num_rows
        print(f"SQLite: Начинаем потоковую загрузку ({total_rows:,} строк)...")

        t0 = time.perf_counter()
        processed = 0

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df_batch = batch.to_pandas()
            continents, countries, cities = prepare_3nf_tables(df_batch)

            # Вставляем батчи через pandas to_sql, но с method написанным через INSERT OR IGNORE
            for table_name, df_data in [
                ("continents", continents),
                ("countries", countries),
                ("cities", cities),
            ]:
                if df_data.empty:
                    continue
                df_clean = df_data.where(pd.notnull(df_data), None)
                records = [tuple(x) for x in df_clean.to_numpy()]

                if table_name == "continents":
                    cursor.executemany(
                        "INSERT OR IGNORE INTO continents (continent_id, continent_name) VALUES (?, ?)",
                        records,
                    )
                elif table_name == "countries":
                    cursor.executemany(
                        "INSERT OR IGNORE INTO countries (iso3, iso2, country_name, continent_id, population_2022, area_km2) VALUES (?, ?, ?, ?, ?, ?)",
                        records,
                    )
                elif table_name == "cities":
                    cursor.executemany(
                        "INSERT OR IGNORE INTO cities (city_id, city_name, lat, lng, capital_status, population, iso3) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        records,
                    )

            conn.commit()
            processed += len(df_batch)
            print(f"SQLite: обработано {processed:,} / {total_rows:,} строк...")

        load_time = time.perf_counter() - t0
        print(f"SQLite: Все данные успешно загружены в 3НФ за {load_time:.2f} с")

    except Exception as e:
        conn.rollback()
        print(f"Ошибка при миграции в SQLite: {e}")
    finally:
        cursor.close()
        conn.close()


def load_to_postgres(
    clean_parquet_path="data/processed/clean.parquet", batch_size=100_000
):
    """Потоковая загрузка больших данных в PostgreSQL в 3НФ батчами с полной очисткой старых таблиц."""
    conn = psycopg2.connect(
        database=os.getenv("PG_DB", "analytics").strip(),
        user=os.getenv("PG_USER", "postgres").strip(),
        password=os.getenv("PG_PASSWORD", "2280").strip(),
        host=os.getenv("PG_HOST", "localhost").strip(),
        port=int(os.getenv("PG_PORT", "5432").strip()),
    )
    cursor = conn.cursor()

    try:
        # Сбрасываем старые таблицы, чтобы избежать конфликтов со старыми данными
        cursor.execute("""
            DROP TABLE IF EXISTS cities, countries, continents CASCADE;
            
            CREATE TABLE continents (
                continent_id INTEGER PRIMARY KEY,
                continent_name VARCHAR(100) NOT NULL UNIQUE
            );
            CREATE TABLE countries (
                iso3 VARCHAR(10) PRIMARY KEY,
                iso2 VARCHAR(10),
                country_name VARCHAR(150) NOT NULL,
                continent_id INTEGER REFERENCES continents(continent_id),
                population_2022 DOUBLE PRECISION,
                area_km2 DOUBLE PRECISION
            );
            CREATE TABLE cities (
                city_id BIGINT PRIMARY KEY,
                city_name VARCHAR(200) NOT NULL,
                lat DOUBLE PRECISION,
                lng DOUBLE PRECISION,
                capital_status VARCHAR(50),
                population DOUBLE PRECISION,
                iso3 VARCHAR(10) REFERENCES countries(iso3)
            );
        """)

        # Гарантированно добавляем системный континент 'Unknown' (ID = 0) в самую первую очередь
        cursor.execute(
            "INSERT INTO continents (continent_id, continent_name) VALUES (%s, %s) ON CONFLICT (continent_id) DO NOTHING",
            (0, "Unknown"),
        )
        conn.commit()

        parquet_file = pq.ParquetFile(clean_parquet_path)
        total_rows = parquet_file.metadata.num_rows
        print(f"PostgreSQL: Начинаем потоковую загрузку ({total_rows:,} строк)...")

        t0 = time.perf_counter()
        processed = 0

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df_batch = batch.to_pandas()
            continents, countries, cities = prepare_3nf_tables(df_batch)

            for table_df, query in [
                (
                    continents,
                    "INSERT INTO continents (continent_id, continent_name) VALUES %s ON CONFLICT (continent_id) DO NOTHING",
                ),
                (
                    countries,
                    "INSERT INTO countries (iso3, iso2, country_name, continent_id, population_2022, area_km2) VALUES %s ON CONFLICT (iso3) DO NOTHING",
                ),
                (
                    cities,
                    "INSERT INTO cities (city_id, city_name, lat, lng, capital_status, population, iso3) VALUES %s ON CONFLICT (city_id) DO NOTHING",
                ),
            ]:
                df_clean = table_df.where(pd.notnull(table_df), None)
                records = [tuple(x) for x in df_clean.to_numpy(dtype=object)]
                if records:
                    execute_values(cursor, query, records, page_size=10000)

            conn.commit()
            processed += len(df_batch)
            print(f"PostgreSQL: обработано {processed:,} / {total_rows:,} строк...")

        load_time = time.perf_counter() - t0
        print(f"PostgreSQL: Все данные успешно загружены в 3НФ за {load_time:.2f} с")

    except Exception as e:
        conn.rollback()
        print(f"Ошибка при миграции в PostgreSQL: {e}")
    finally:
        cursor.close()
        conn.close()
