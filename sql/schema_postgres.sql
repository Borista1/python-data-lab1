-- DDL-схема базы данных (PostgreSQL), выгруженная через DBeaver (Generate SQL -> DDL)

DROP TABLE IF EXISTS public.cities CASCADE;
DROP TABLE IF EXISTS public.countries CASCADE;
DROP TABLE IF EXISTS public.continents CASCADE;

CREATE TABLE public.continents (
    continent_id int4 NOT NULL,
    continent_name varchar(64) NOT NULL,
    CONSTRAINT continents_pkey PRIMARY KEY (continent_id),
    CONSTRAINT continents_continent_name_key UNIQUE (continent_name)
);

CREATE TABLE public.countries (
    iso3 varchar(10) NOT NULL,
    iso2 varchar(10) NULL,
    country_name varchar(128) NOT NULL,
    continent_id int4 NULL,
    population_2022 numeric(15, 2) NULL,
    area_km2 numeric(12, 2) NULL,
    CONSTRAINT countries_pkey PRIMARY KEY (iso3),
    CONSTRAINT countries_continent_id_fkey FOREIGN KEY (continent_id) REFERENCES public.continents(continent_id)
);

CREATE TABLE public.cities (
    city_id int8 NOT NULL,
    city_name varchar(128) NOT NULL,
    lat float4 NULL,
    lng float4 NULL,
    capital_status varchar(32) NULL,
    population float4 NULL,
    iso3 varchar(10) NULL,
    CONSTRAINT cities_pkey PRIMARY KEY (city_id),
    CONSTRAINT cities_iso3_fkey FOREIGN KEY (iso3) REFERENCES public.countries(iso3)
);

CREATE INDEX idx_cities_iso3 ON public.cities USING btree (iso3);
CREATE INDEX idx_cities_population ON public.cities USING btree (population);-- DDL-схема базы данных (PostgreSQL), выгруженная через DBeaver (Generate SQL -> DDL)

DROP TABLE IF EXISTS public.cities CASCADE;
DROP TABLE IF EXISTS public.countries CASCADE;
DROP TABLE IF EXISTS public.continents CASCADE;

CREATE TABLE public.continents (
    continent_id int4 NOT NULL,
    continent_name varchar(64) NOT NULL,
    CONSTRAINT continents_pkey PRIMARY KEY (continent_id),
    CONSTRAINT continents_continent_name_key UNIQUE (continent_name)
);

CREATE TABLE public.countries (
    iso3 varchar(10) NOT NULL,
    iso2 varchar(10) NULL,
    country_name varchar(128) NOT NULL,
    continent_id int4 NULL,
    population_2022 numeric(15, 2) NULL,
    area_km2 numeric(12, 2) NULL,
    CONSTRAINT countries_pkey PRIMARY KEY (iso3),
    CONSTRAINT countries_continent_id_fkey FOREIGN KEY (continent_id) REFERENCES public.continents(continent_id)
);

CREATE TABLE public.cities (
    city_id int8 NOT NULL,
    city_name varchar(128) NOT NULL,
    lat float4 NULL,
    lng float4 NULL,
    capital_status varchar(32) NULL,
    population float4 NULL,
    iso3 varchar(10) NULL,
    CONSTRAINT cities_pkey PRIMARY KEY (city_id),
    CONSTRAINT cities_iso3_fkey FOREIGN KEY (iso3) REFERENCES public.countries(iso3)
);

CREATE INDEX idx_cities_iso3 ON public.cities USING btree (iso3);
CREATE INDEX idx_cities_population ON public.cities USING btree (population);