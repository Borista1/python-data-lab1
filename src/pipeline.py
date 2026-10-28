# src/pipeline.py
import os
import argparse
import logging
from src import config
from src.io_utils import read_csv_auto
from src.cleaning import (
    clean_column_names,
    join_datasets,
    convert_types,
    replace_hidden_missing,
    handle_missing_values,
    remove_duplicates,
    normalize_categories_and_text,
    check_business_rules,
    detect_outliers,
)


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def run_pipeline(cities_path, pop_path, output_path, sample=False):
    log = logging.getLogger(__name__)
    log.info("=== ЗАПУСК КОНВЕЙЕРА ОЧИСТКИ ДАННЫХ ===")

    if not os.path.exists(cities_path):
        raise FileNotFoundError(f"Файл не найден: {cities_path}")
    if not os.path.exists(pop_path):
        raise FileNotFoundError(f"Файл не найден: {pop_path}")

    cities_df = read_csv_auto(cities_path)
    pop_df = read_csv_auto(pop_path)

    if sample:
        cities_df = cities_df.head(5000)
        log.info("Включен режим среза (--sample): взято 5000 строк")

    initial_rows = len(cities_df)
    log.info("Исходное число строк: %d", initial_rows)

    # Применяем функции очистки по очереди
    cities_df = clean_column_names(cities_df)
    pop_df = clean_column_names(pop_df)

    df = join_datasets(cities_df, pop_df)
    df = convert_types(df)
    df = replace_hidden_missing(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = normalize_categories_and_text(df)
    df = check_business_rules(df)
    df = detect_outliers(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

    final_rows = len(df)
    lost_pct = (initial_rows - final_rows) / initial_rows * 100
    log.info(
        "=== ОЧИСТКА ЗАВЕРШЕНА: Было %d, Стало %d (Потеряно %.2f%%) ===",
        initial_rows,
        final_rows,
        lost_pct,
    )
    log.info("Результат сохранен в %s", output_path)
    return df


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Конвейер очистки данных ЛР1")
    parser.add_argument("--input", default=config.RAW_CITIES, help="Путь к исходному файлу")
    parser.add_argument("--pop", default=config.RAW_POPULATION, help="Путь к справочнику стран")
    parser.add_argument("--output", default=config.CLEAN_PARQUET, help="Путь к результату")
    parser.add_argument("--sample", action="store_true", help="Запуск на небольшом срезе")

    args = parser.parse_args()
    run_pipeline(args.input, args.pop, args.output, sample=args.sample)