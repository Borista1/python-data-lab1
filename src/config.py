# src/config.py
# Простой файл с настройками и путями к файлам

RAW_GEONAMES = "data/raw/geonames.csv"
RAW_CITIES = "data/raw/worldcities.csv"
RAW_POPULATION = "data/raw/world_population.csv"

SAMPLE_CSV = "data/interim/sample_200k.csv"
INTERIM_PARQUET = "data/interim/source.parquet"
INTERIM_FEATHER = "data/interim/source.feather"
INTERIM_PICKLE = "data/interim/source.pkl"
TYPED_PARQUET = "data/interim/source_typed.parquet"

CLEAN_PARQUET = "data/processed/clean.parquet"
LOG_FILE = "logs/cleaning.log"

# Оптимальная схема типов для таблицы городов (worldcities)
SCHEMA_CITIES = {
    "city": "string",
    "city_ascii": "string",
    "lat": "float32",
    "lng": "float32",
    "country": "category",
    "iso2": "category",
    "iso3": "category",
    "admin_name": "string",
    "capital": "category",
    "population": "float32",
    "id": "int64",
}