# src/io_utils.py
import os
import time
from pathlib import Path
import pandas as pd
from src import config


def inspect_raw_file(filepath):
    """Шаг 6: Узнаем размер файла и читаем первые 5 строк без загрузки всего файла."""
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Файл: {filepath}")
    print(f"Размер: {size_mb:.2f} МБ")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for i in range(5):
            line = f.readline().strip()
            print(f"Строка {i}: {line[:120]}")
    print("-" * 50)


def make_sample_200k(input_path, output_path, num_rows=200000):
    """Шаг 6: Потоковое чтение первых 200 000 строк из большого файла geonames.csv."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    with (
        open(input_path, "r", encoding="utf-8", errors="replace") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            fout.write(line)
            count += 1
            # +1 строка на заголовок столбцов
            if count >= num_rows + 1:
                break

    print(f"Готово! Записано {count - 1} строк в файл {output_path}")


def read_csv_auto(filepath):
    """Чтение CSV с автоопределением разделителя и оптимизацией памяти для geonames.csv."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()

    if ";" in first_line and first_line.count(";") >= first_line.count(","):
        sep = ";"
    elif "\t" in first_line:
        sep = "\t"
    else:
        sep = ","

    # Если читаем большой geonames.csv, загружаем только нужные для пайплайна столбцы
    if "geonameid" in first_line.lower():
        needed = {
            "geonameid",
            "name",
            "asciiname",
            "latitude",
            "longitude",
            "feature class",
            "feature code",
            "country code",
            "admin1 code",
            "population",
        }
        return pd.read_csv(
            filepath,
            sep=sep,
            encoding="utf-8",
            usecols=lambda c: c.strip().lower() in needed,
            low_memory=False,
        )

    return pd.read_csv(filepath, sep=sep, encoding="utf-8", low_memory=False)


def benchmark_formats(csv_path):
    """Шаг 9: Сравнение CSV, Pickle, Feather и Parquet."""
    results = []

    # 1. Читаем исходный CSV
    t0 = time.perf_counter()
    df = read_csv_auto(csv_path)
    csv_read_time = time.perf_counter() - t0
    csv_size = os.path.getsize(csv_path) / (1024 * 1024)
    results.append(["CSV (исходный)", csv_size, 0.0, csv_read_time])

    # 2. Pickle
    t0 = time.perf_counter()

    from pathlib import Path

    Path(config.INTERIM_PICKLE).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    df.to_pickle(config.INTERIM_PICKLE)
    w_time = time.perf_counter() - t0
    df.to_pickle(config.INTERIM_PICKLE)
    w_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = pd.read_pickle(config.INTERIM_PICKLE)
    r_time = time.perf_counter() - t0
    size = os.path.getsize(config.INTERIM_PICKLE) / (1024 * 1024)
    results.append(["Pickle", size, w_time, r_time])

    # 3. Feather
    t0 = time.perf_counter()
    df.to_feather(config.INTERIM_FEATHER)
    w_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = pd.read_feather(config.INTERIM_FEATHER)
    r_time = time.perf_counter() - t0
    size = os.path.getsize(config.INTERIM_FEATHER) / (1024 * 1024)
    results.append(["Feather", size, w_time, r_time])

    # 4. Parquet (snappy)
    t0 = time.perf_counter()
    df.to_parquet(config.INTERIM_PARQUET, compression="snappy")
    w_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = pd.read_parquet(config.INTERIM_PARQUET)
    r_time = time.perf_counter() - t0
    size = os.path.getsize(config.INTERIM_PARQUET) / (1024 * 1024)
    results.append(["Parquet (snappy)", size, w_time, r_time])

    # 5. Parquet (только 3 первых столбца)
    cols_3 = list(df.columns[:3])
    t0 = time.perf_counter()
    _ = pd.read_parquet(config.INTERIM_PARQUET, columns=cols_3)
    r_time_3 = time.perf_counter() - t0
    results.append(["Parquet, 3 столбца", size, 0.0, r_time_3])

    res_df = pd.DataFrame(
        results, columns=["Формат", "Объём, МБ", "Запись, с", "Чтение, с"]
    )
    res_df["Отношение объёма к CSV"] = res_df["Объём, МБ"] / csv_size
    return res_df
