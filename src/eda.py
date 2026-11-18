# src/eda.py
import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def target_summary(df, target_col="population"):
    """Шаг 3: Сводка распределения целевой переменной."""
    s = df[target_col]
    if "is_population_missing" in df.columns:
        missing_count = int(df["is_population_missing"].sum())
    else:
        missing_count = int(s.isna().sum())

    q1 = float(s.quantile(0.25))
    median_val = float(s.median())
    mean_val = float(s.mean())
    q3 = float(s.quantile(0.75))
    skew_dir = "Вправо (положительная)" if mean_val > median_val else "Влево (отрицательная)"

    summary_df = pd.DataFrame(
        [
            ["Число наблюдений", int(len(s))],
            ["Число пропусков", missing_count],
            ["Минимум", round(float(s.min()), 2)],
            ["Первый квартиль", round(q1, 2)],
            ["Медиана", round(median_val, 2)],
            ["Среднее", round(mean_val, 2)],
            ["Третий квартиль", round(q3, 2)],
            ["Максимум", round(float(s.max()), 2)],
            ["Скошенность", skew_dir],
        ],
        columns=["Показатель", "Значение"],
    )
    return summary_df


def build_time_series(df, target_col="population"):
    """Шаг 4: Сбор динамики по периодам (1970-2022) с сохранением пустых периодов."""
    census_years = [1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022]
    all_years = sorted(list(set(range(1970, 2025, 5)).union(census_years)))

    rows = []
    for y in all_years:
        date_str = f"{y}-01-01"
        col_name = f"{y}_population"
        if col_name in df.columns:
            valid = df.dropna(subset=[col_name, "2022_population"]).copy()
            ratio = valid[col_name] / valid["2022_population"]
            city_pop_y = valid[target_col] * ratio
            rows.append(
                {
                    "date": pd.to_datetime(date_str),
                    "Период": str(y),
                    "Число записей": int(len(city_pop_y)),
                    "Медиана цели": round(float(city_pop_y.median()), 2),
                    "Среднее цели": round(float(city_pop_y.mean()), 2),
                }
            )
        else:
            rows.append(
                {
                    "date": pd.to_datetime(date_str),
                    "Период": str(y),
                    "Число записей": 0,
                    "Медиана цели": np.nan,
                    "Среднее цели": np.nan,
                }
            )

    return pd.DataFrame(rows)


def group_aggregates(df, group_col="continent", target_col="population", rare_threshold=0.01):
    """Шаг 5: Расчёт агрегатов по группам и слияние редких категорий (< 1%) для диаграммы."""
    total_len = len(df)
    records = []

    for grp, gdf in df.groupby(group_col, observed=False):
        cnt = int(len(gdf))
        med = round(float(gdf[target_col].median()), 2)
        avg = round(float(gdf[target_col].mean()), 2)
        if "is_population_missing" in gdf.columns:
            miss_share = round(float(gdf["is_population_missing"].mean() * 100), 2)
        else:
            miss_share = round(float(gdf[target_col].isna().mean() * 100), 2)
        share = cnt / total_len
        records.append(
            {
                "Группа": str(grp),
                "Число": cnt,
                "Доля группы": share,
                "Медиана": med,
                "Среднее": avg,
                "Доля пропусков": f"{miss_share:.2f}%",
            }
        )

    table3_df = pd.DataFrame(records).sort_values("Число", ascending=False).reset_index(drop=True)

    df_plot = df[[group_col, target_col]].copy()
    df_plot[group_col] = df_plot[group_col].astype(str)
    rare_groups = table3_df.loc[table3_df["Доля группы"] < rare_threshold, "Группа"].tolist()
    df_plot["group_merged"] = df_plot[group_col].apply(
        lambda x: "Прочие" if x in rare_groups else x
    )

    plot_agg = (
        df_plot.groupby("group_merged", as_index=False)[target_col]
        .agg(Число="count", Медиана="median", Среднее="mean")
        .sort_values("Медиана", ascending=False)
        .reset_index(drop=True)
    )

    table3_clean = table3_df[["Группа", "Число", "Медиана", "Среднее", "Доля пропусков"]]
    return table3_clean, plot_agg, rare_groups


def build_pivot_table(df, row_col="continent", col_col="capital_x", target_col="population"):
    """Шаг 5: Построение сводной таблицы (строки — группа, столбцы — статус столицы, ячейка — медиана цели)."""
    temp = df[[row_col, col_col, target_col]].copy()
    temp[row_col] = temp[row_col].astype(str)
    temp[col_col] = temp[col_col].fillna("not_capital").astype(str)
    temp.loc[temp[col_col].isin(["None", "nan", ""]), col_col] = "not_capital"

    pivot_df = temp.pivot_table(
        index=row_col,
        columns=col_col,
        values=target_col,
        aggfunc="median",
        observed=False,
    )
    cols_order = [c for c in ["primary", "admin", "minor", "not_capital"] if c in pivot_df.columns]
    return pivot_df[cols_order].round(1)


def detect_anomalies_iqr(series):
    """Шаг 6: Первое правило аномалий — межквартильный размах (1.5 * IQR)."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    count = int(mask.sum())
    log.info("Правило IQR: отмечено %d строк (верхняя граница: %.2f)", count, upper)
    return mask, count, lower, upper


def detect_anomalies_zscore(series, threshold=3.0):
    """Шаг 6: Второе правило аномалий — модуль z-оценки больше 3 (|z| > 3)."""
    mean_val = float(series.mean())
    std_val = float(series.std())
    z_scores = (series - mean_val) / std_val
    mask = z_scores.abs() > threshold
    count = int(mask.sum())
    upper = mean_val + threshold * std_val
    log.info("Правило |z| > %.1f: отмечено %d строк (верхняя граница: %.2f)", threshold, count, upper)
    return mask, count, upper


def compare_anomalies(df, target_col="population"):
    """Шаг 6: Сопоставление двух правил поиска аномалий и формирование таблицы 4."""
    s = df[target_col]
    total = len(s)

    mask_iqr, cnt_iqr, _, upper_iqr = detect_anomalies_iqr(s)
    mask_z, cnt_z, upper_z = detect_anomalies_zscore(s, threshold=3.0)

    mask_both = mask_iqr & mask_z
    cnt_both = int(mask_both.sum())

    log.info(
        "Сравнение правил: IQR=%d, |z|>3=%d, пересечение=%d",
        cnt_iqr,
        cnt_z,
        cnt_both,
    )

    table4_df = pd.DataFrame(
        [
            [
                "Межквартильный размах",
                cnt_iqr,
                f"{cnt_iqr / total * 100:.2f}%",
                cnt_both,
            ],
            [
                "Второе правило (|z| > 3)",
                cnt_z,
                f"{cnt_z / total * 100:.2f}%",
                cnt_both,
            ],
            [
                "Пересечение",
                cnt_both,
                f"{cnt_both / total * 100:.2f}%",
                cnt_both,
            ],
        ],
        columns=["Правило", "Число отмеченных", "Доля", "Совпало со вторым"],
    )
    return table4_df, mask_iqr, mask_z, mask_both, upper_iqr, upper_z


def mean_loop(values):
    """Шаг 7: Расчёт среднего значения циклом Python."""
    total = 0.0
    count = 0
    for value in values:
        if value == value:  # пропуск не равен себе
            total += float(value)
            count += 1
    return total / count


def mean_numpy(values):
    """Шаг 7: Расчёт среднего значения средствами numpy."""
    array = np.asarray(values, dtype="float64")
    return float(np.nanmean(array))


def median_seconds(func, values):
    """Шаг 7: Холостой прогон и медиана времени из 3 повторов."""
    func(values)
    marks = []
    for _ in range(3):
        started = time.perf_counter()
        func(values)
        marks.append(time.perf_counter() - started)
    marks.sort()
    return marks[1]


def benchmark_vectorization(series):
    """Шаг 7: Сверка результатов и замер времени цикла Python против numpy."""
    values = series
    n_values = len(values)

    res_loop = mean_loop(values)
    res_np = mean_numpy(values)
    if not np.isclose(res_loop, res_np, rtol=1e-5):
        raise ValueError(f"Результаты не совпали: loop={res_loop}, numpy={res_np}")

    t_loop = median_seconds(mean_loop, values)
    t_np = median_seconds(mean_numpy, values)
    ratio = t_loop / t_np if t_np > 0 else np.nan

    table5_df = pd.DataFrame(
        [
            ["Цикл Python", f"{t_loop:.6f}", f"{res_loop:.2f}"],
            ["numpy", f"{t_np:.6f}", f"{res_np:.2f}"],
            ["Отношение цикл / numpy", f"{ratio:.1f} раз", f"{res_loop:.2f}"],
            ["Число значений", str(n_values), "—"],
        ],
        columns=["Способ", "Медиана времени, с", "Значение метрики"],
    )
    return table5_df, t_loop, t_np, ratio, res_loop


def generate_all_plots(df, plots_dir="plots"):
    """Шаг 8: Построение и сохранение 6 графиков разведочного анализа данных."""
    os.makedirs(plots_dir, exist_ok=True)
    plt.rcParams["font.size"] = 10

    s = df["population"]
    q95 = float(s.quantile(0.95))
    med_all = float(s.median())
    mean_all = float(s.mean())

    # Рисунок 1: Гистограмма целевой переменной (срез до 95-го перцентиля для читаемости ядра)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sample = s[s <= q95]
    ax.hist(sample / 1000.0, bins=40, color="#2b5c8f", edgecolor="white", alpha=0.9)
    ax.axvline(med_all / 1000.0, color="#d95f02", linestyle="--", linewidth=2, label=f"Медиана: {med_all:,.0f} чел.")
    ax.axvline(mean_all / 1000.0, color="#7570b3", linestyle=":", linewidth=2, label=f"Среднее: {mean_all:,.0f} чел.")
    ax.set_title("Рис. 1. Распределение численности населения городов (до 95-го перцентиля)")
    ax.set_xlabel("Численность населения города, тыс. чел.")
    ax.set_ylabel("Число записей (городов), шт.")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "01_hist.png"), dpi=150)
    plt.close(fig)

    # Рисунок 2: Целевая переменная в разрезе групп (Boxplot по континентам без экстремальных выбросов)
    _, _, rare_groups = group_aggregates(df)
    df_box = df[["continent", "population"]].copy()
    df_box["continent"] = df_box["continent"].astype(str)
    df_box["group"] = df_box["continent"].apply(lambda x: "Прочие" if x in rare_groups else x)
    order_groups = ["Asia", "South America", "Africa", "North America", "Europe", "Прочие"]
    data_by_grp = [df_box.loc[df_box["group"] == g, "population"].values / 1000.0 for g in order_groups]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data_by_grp, tick_labels=order_groups, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#80b1d3")
        patch.set_alpha(0.85)
    for median_line in bp["medians"]:
        median_line.set_color("#d95f02")
        median_line.set_linewidth(2)
    ax.set_title("Рис. 2. Численность населения городов в разрезе континентов (без выбросов)")
    ax.set_xlabel("Группа (континент)")
    ax.set_ylabel("Численность населения города, тыс. чел.")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "02_groups_box.png"), dpi=150)
    plt.close(fig)

    # Рисунок 3: Ряд по датам (с сохранением пустых периодов)
    ts_df = build_time_series(df)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    x_idx = np.arange(len(ts_df))
    bars = ax1.bar(x_idx, ts_df["Число записей"], width=0.55, color="#cbd5e8", label="Число записей (городов), шт.")
    ax1.set_xlabel("Период (год переписи)")
    ax1.set_ylabel("Число записей, шт.", color="#4a6fe3")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(ts_df["Период"])
    ax1.set_ylim(0, 62000)

    for i, cnt in enumerate(ts_df["Число записей"]):
        if cnt == 0:
            ax1.text(i, 2000, "Пусто\n(0 зап.)", ha="center", va="bottom", fontsize=8, color="#b2182b", fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(
        x_idx,
        ts_df["Медиана цели"] / 1000.0,
        color="#d95f02",
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="Медиана населения, тыс. чел.",
    )
    # Пунктиром покажем общую траекторию через пустые периоды для наглядности, но основная линия прерывается на NaN
    valid_mask = ts_df["Медиана цели"].notna()
    ax2.plot(
        x_idx[valid_mask],
        ts_df.loc[valid_mask, "Медиана цели"] / 1000.0,
        color="#d95f02",
        linestyle="--",
        alpha=0.5,
    )
    ax2.set_ylabel("Медиана населения города, тыс. чел.", color="#d95f02")
    ax2.set_ylim(8, 24)
    ax1.set_title("Рис. 3. Динамика числа записей и медианы населения по периодам (1970–2022 гг.)")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "03_time_series.png"), dpi=150)
    plt.close(fig)

    # Рисунок 4: Столбчатая диаграмма агрегата по группам (редкие < 1% слиты в «Прочие»)
    _, plot_agg, _ = group_aggregates(df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(plot_agg["group_merged"], plot_agg["Медиана"] / 1000.0, color="#386cb0", edgecolor="white", width=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.4, f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Рис. 4. Медианная численность населения города по континентам")
    ax.set_xlabel("Группа (континент, редкие < 1% объединены в «Прочие»)")
    ax.set_ylabel("Медиана населения города, тыс. чел.")
    ax.set_ylim(0, 28)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "04_group_bars.png"), dpi=150)
    plt.close(fig)

    # Рисунок 5: Тепловая карта сводной таблицы (с сохранением пустых ячеек NaN)
    piv = build_pivot_table(df)
    piv_thou = piv / 1000.0
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#eeeeee")
    # Используем логарифмическую цветовую шкалу или верхний порог для читаемости контраста, подписывая точные числа
    masked_vals = np.ma.masked_invalid(np.log10(piv_thou.values + 1.0))
    im = ax.imshow(masked_vals, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Статус населённого пункта (capital_x)")
    ax.set_ylabel("Группа (континент)")
    ax.set_title("Рис. 5. Тепловая карта медианы населения (тыс. чел.): континент и статус столицы", fontsize=10.5)

    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            val = piv_thou.iloc[i, j]
            if np.isnan(val):
                ax.text(j, i, "Пусто (NaN)", ha="center", va="center", color="#666666", fontsize=8.5, fontstyle="italic")
            else:
                txt_color = "white" if val > 250 else "black"
                ax.text(j, i, f"{val:,.1f}", ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "05_heatmap.png"), dpi=150)
    plt.close(fig)

    # Рисунок 6: Отметка аномалий на распределении (по готовым маскам из src/eda.py)
    _, mask_iqr, mask_z, _, upper_iqr, upper_z = compare_anomalies(df)
    only_iqr = mask_iqr & (~mask_z)
    normal = ~mask_iqr

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.scatter(
        df.loc[normal, "lng"],
        df.loc[normal, "population"] / 1_000_000.0,
        s=6,
        alpha=0.35,
        color="#8da0cb",
        label=f"Обычные записи ({normal.sum():,} шт.)",
    )
    ax.scatter(
        df.loc[only_iqr, "lng"],
        df.loc[only_iqr, "population"] / 1_000_000.0,
        s=12,
        alpha=0.6,
        color="#fc8d62",
        label=f"Только правило 1.5 IQR ({only_iqr.sum():,} шт.)",
    )
    ax.scatter(
        df.loc[mask_z, "lng"],
        df.loc[mask_z, "population"] / 1_000_000.0,
        s=28,
        alpha=0.9,
        color="#e41a1c",
        label=f"Оба правила: 1.5 IQR и |z| > 3 ({mask_z.sum():,} шт.)",
    )
    ax.axhline(upper_z / 1_000_000.0, color="#e41a1c", linestyle="--", linewidth=1.5, label=f"Порог |z| > 3 ({upper_z/1e6:.2f} млн чел.)")
    ax.set_title("Рис. 6. Отметка аномалий численности населения городов по двум правилам")
    ax.set_xlabel("Географическая долгота города (lng), градусы")
    ax.set_ylabel("Численность населения города, млн чел.")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "06_anomalies.png"), dpi=150)
    plt.close(fig)
