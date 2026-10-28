DROP TABLE IF EXISTS cities;
DROP TABLE IF EXISTS countries;
DROP TABLE IF EXISTS continents;

CREATE TABLE continents (
    continent_id INTEGER PRIMARY KEY,
    continent_name VARCHAR(64) NOT NULL UNIQUE
);

CREATE TABLE countries (
    iso3 VARCHAR(3) PRIMARY KEY,
    iso2 VARCHAR(2),
    country_name VARCHAR(128) NOT NULL,
    continent_id INTEGER REFERENCES continents(continent_id),
    population_2022 BIGINT,
    area_km2 NUMERIC(12, 2)
);

CREATE TABLE cities (
    city_id BIGINT PRIMARY KEY,
    city_name VARCHAR(128) NOT NULL,
    lat NUMERIC(8, 4) NOT NULL,
    lng NUMERIC(8, 4) NOT NULL,
    capital_status VARCHAR(32),
    population NUMERIC(12, 0),
    iso3 VARCHAR(3) REFERENCES countries(iso3)
);