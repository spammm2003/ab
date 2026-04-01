import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from lxml import etree

st.set_page_config(page_title="Компанія під мікроскопом", page_icon="🔎", layout="wide")

SMIDA_FEED_URL = "https://smida.gov.ua/db/api/v1/feed-index.xml"
REQUEST_TIMEOUT = 25
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    )
}


# ---------------------------------
# Допоміжні функції
# ---------------------------------
def safe_get(url: str, params: Optional[dict] = None) -> requests.Response:
    response = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response



def extract_edrpou(user_input: str) -> Optional[str]:
    s = str(user_input).strip()
    match = re.fullmatch(r"\d{8,10}", s)
    if match:
        return match.group(0)
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def get_smida_reports_by_edrpou(edrpou: str, limit: int = 300) -> pd.DataFrame:
    params = {
        "edrpou": edrpou,
        "period": "y",
        "limit": limit,
    }

    response = safe_get(SMIDA_FEED_URL, params=params)
    root = etree.fromstring(response.content)

    items = []
    for item in root.findall(".//item"):
        row = {}
        for child in item:
            tag = etree.QName(child).localname
            row[tag] = (child.text or "").strip()
        items.append(row)

    df = pd.DataFrame(items)
    if df.empty:
        return df

    rename_map = {
        "title": "title",
        "link": "link",
        "pubDate": "pub_date",
        "description": "description",
        "guid": "guid",
        "edrpou": "edrpou",
        "period": "period",
        "date": "report_date",
        "fdate": "period_end",
        "sdate": "period_start",
        "name": "company_name",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["pub_date", "report_date", "period_end", "period_start"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df



def choose_latest_company_name(df: pd.DataFrame, fallback: str) -> str:
    if df.empty or "company_name" not in df.columns:
        return fallback
    values = df["company_name"].dropna().astype(str)
    if values.empty:
        return fallback
    return values.iloc[0]



def extract_report_links(df: pd.DataFrame) -> List[str]:
    if df.empty or "link" not in df.columns:
        return []
    return [x for x in df["link"].dropna().astype(str).tolist() if x.startswith("http")]



def parse_number(value) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().replace("\xa0", " ")
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", ".", "-", "-."}:
        return None
    try:
        return float(s)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def parse_financials_from_report_page(url: str) -> List[Dict]:
    keywords = {
        "Чистий дохід": ["чистий дохід", "дохід від реалізації", "net revenue"],
        "Чистий прибуток": ["чистий прибуток", "чистий збиток", "net profit"],
        "Активи": ["усього активів", "активи", "total assets"],
        "Зобов'язання": ["усього зобов'язань", "зобов'язання", "total liabilities"],
        "Власний капітал": ["власний капітал", "equity"],
        "Грошові кошти": ["грошові кошти", "cash"],
        "Поточні активи": ["оборотні активи", "поточні активи", "current assets"],
        "Поточні зобов'язання": ["поточні зобов'язання", "current liabilities"],
    }

    try:
        resp = safe_get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")
        results = []

        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
                if len(cells) < 2:
                    continue

                row_text = " ".join(cells).lower()
                for metric_name, variants in keywords.items():
                    if any(v in row_text for v in variants):
                        numbers = []
                        for c in cells[1:]:
                            found = re.findall(r"-?\d[\d\s,.]*", c)
                            numbers.extend(found)

                        value = None
                        if numbers:
                            value = parse_number(numbers[-1])

                        if value is not None:
                            results.append(
                                {
                                    "metric": metric_name,
                                    "value": value,
                                    "source_url": url,
                                }
                            )
        return results
    except Exception:
        return []



def build_financial_dataframe(report_df: pd.DataFrame, max_reports_to_scan: int = 12) -> pd.DataFrame:
    links = extract_report_links(report_df)[:max_reports_to_scan]
    collected = []

    for link in links:
        collected.extend(parse_financials_from_report_page(link))

    if not collected:
        return pd.DataFrame(columns=["metric", "value", "source_url"])

    fin_df = pd.DataFrame(collected)
    fin_df = fin_df.dropna(subset=["metric", "value"])
    fin_df = fin_df.groupby("metric", as_index=False).agg({"value": "last", "source_url": "last"})
    return fin_df



def calculate_risk_score(fin_df: pd.DataFrame) -> Tuple[int, str, List[str], Dict[str, Optional[float]]]:
    metrics = {row["metric"]: row["value"] for _, row in fin_df.iterrows()} if not fin_df.empty else {}

    revenue = metrics.get("Чистий дохід")
    net_profit = metrics.get("Чистий прибуток")
    assets = metrics.get("Активи")
    liabilities = metrics.get("Зобов'язання")
    equity = metrics.get("Власний капітал")
    current_assets = metrics.get("Поточні активи")
    current_liabilities = metrics.get("Поточні зобов'язання")
    cash = metrics.get("Грошові кошти")

    ratios = {
        "Рентабельність": round(net_profit / revenue, 4) if revenue not in (None, 0) and net_profit is not None else None,
        "Автономія": round(equity / assets, 4) if assets not in (None, 0) and equity is not None else None,
        "Борг/Активи": round(liabilities / assets, 4) if assets not in (None, 0) and liabilities is not None else None,
        "Поточна ліквідність": round(current_assets / current_liabilities, 4) if current_liabilities not in (None, 0) and current_assets is not None else None,
        "Cash ratio": round(cash / current_liabilities, 4) if current_liabilities not in (None, 0) and cash is not None else None,
    }

    score = 0
    reasons = []

    if net_profit is not None and net_profit < 0:
        score += 30
        reasons.append("Компанія має від'ємний чистий прибуток.")

    if ratios["Автономія"] is not None and ratios["Автономія"] < 0.2:
        score += 20
        reasons.append("Низька частка власного капіталу в активах.")

    if ratios["Борг/Активи"] is not None and ratios["Борг/Активи"] > 0.8:
        score += 20
        reasons.append("Високе боргове навантаження відносно активів.")

    if ratios["Поточна ліквідність"] is not None and ratios["Поточна ліквідність"] < 1:
        score += 15
        reasons.append("Недостатня поточна ліквідність.")

    if ratios["Cash ratio"] is not None and ratios["Cash ratio"] < 0.2:
        score += 10
        reasons.append("Низьке покриття короткострокових зобов'язань грошовими коштами.")

    if revenue is None:
        score += 10
        reasons.append("Не вдалося коректно визначити виручку зі звітів.")

    if score <= 20:
        level = "Низький"
    elif score <= 50:
        level = "Середній"
    else:
        level = "Високий"

    return score, level, reasons, ratios



def format_money(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.2f}".replace(",", " ")


# ---------------------------------
# UI
# ---------------------------------
st.title("🔎 OSINT-додаток: Компанія під мікроскопом")
st.caption("Пошук компанії за ЄДРПОУ, збір даних із SMIDA, DataFrame, графіки та оцінка ризику.")

with st.sidebar:
    st.header("Налаштування")
    company_query = st.text_input(
        "ЄДРПОУ компанії",
        placeholder="Наприклад: 14360570",
    )
    max_reports = st.slider("Скільки SMIDA-звітів аналізувати", 3, 30, 12)
    run_btn = st.button("Запустити аналіз", type="primary")

st.markdown(
    """
**Джерела:**
- **SMIDA**: офіційний API відкритих даних.
- Автоматичний пошук через сторонні сайти може блокуватися захистом від ботів, тому тут використовується лише ЄДРПОУ.

> Примітка: структура HTML-сторінок звітів може змінюватися. Якщо SMIDA змінить верстку,
> парсер фінансових таблиць може потребувати оновлення.
"""
)

if run_btn:
    edrpou = extract_edrpou(company_query)

    if not edrpou:
        st.error("Введіть коректний ЄДРПОУ: 8–10 цифр без пробілів.")
        st.stop()

    with st.spinner("Отримую річні звіти з SMIDA..."):
        try:
            report_df = get_smida_reports_by_edrpou(edrpou)
        except Exception as e:
            st.error(f"Помилка запиту до SMIDA: {e}")
            st.stop()

    company_name = choose_latest_company_name(report_df, edrpou)

    col1, col2 = st.columns(2)
    col1.metric("Компанія", company_name)
    col2.metric("ЄДРПОУ", edrpou)

    st.subheader("1) Реєстр звітів SMIDA")
    if report_df.empty:
        st.warning("SMIDA не повернула звітів для цього ЄДРПОУ.")
    else:
        view_cols = [c for c in ["title", "pub_date", "period_start", "period_end", "link"] if c in report_df.columns]
        st.dataframe(report_df[view_cols], use_container_width=True)
        csv_reports = report_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Завантажити таблицю SMIDA (CSV)",
            data=csv_reports,
            file_name=f"smida_reports_{edrpou}.csv",
            mime="text/csv",
        )

    with st.spinner("Парсю фінансові показники зі звітів..."):
        fin_df = build_financial_dataframe(report_df, max_reports_to_scan=max_reports)

    st.subheader("2) Фінансові показники (DataFrame)")
    if fin_df.empty:
        st.warning(
            "Не вдалося автоматично витягнути фінансові показники зі сторінок звітів. "
            "Це нормально: структура звітів у різних компаній може відрізнятися."
        )
    else:
        fin_df_display = fin_df.copy()
        fin_df_display["formatted_value"] = fin_df_display["value"].apply(format_money)
        st.dataframe(fin_df_display[["metric", "formatted_value", "source_url"]], use_container_width=True)

        csv_fin = fin_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Завантажити фінансові показники (CSV)",
            data=csv_fin,
            file_name=f"financial_metrics_{edrpou}.csv",
            mime="text/csv",
        )

        chart_df = fin_df.copy().sort_values("value", ascending=False)
        st.markdown("**Графік ключових фінансових показників**")
        st.bar_chart(chart_df.set_index("metric")["value"], use_container_width=True)

        positive_df = chart_df[chart_df["value"] > 0].copy()
        if not positive_df.empty:
            st.markdown("**Позитивні показники**")
            st.dataframe(positive_df[["metric", "value"]], use_container_width=True)

        score, level, reasons, ratios = calculate_risk_score(fin_df)

        st.subheader("3) Оцінка ризику")
        risk_col1, risk_col2 = st.columns([1, 2])
        risk_col1.metric("Risk score", score)
        risk_col2.metric("Рівень ризику", level)

        ratio_df = pd.DataFrame([{"ratio": k, "value": v} for k, v in ratios.items()])
        ratio_df["formatted"] = ratio_df["value"].apply(lambda x: "—" if x is None else f"{x:.2f}")

        st.markdown("**Розраховані коефіцієнти:**")
        st.dataframe(ratio_df[["ratio", "formatted"]], use_container_width=True)

        if reasons:
            st.markdown("**Причини такого рівня ризику:**")
            for reason in reasons:
                st.write(f"- {reason}")
        else:
            st.write("Суттєвих негативних індикаторів не виявлено.")

        st.subheader("4) Висновок")
        st.info(
            f"За доступними відкритими даними компанія **{company_name}** має рівень ризику: **{level}**. "
            f"Оцінка базується на доступних показниках зі звітності SMIDA."
        )
else:
    st.info("Введіть ЄДРПОУ в лівій панелі та натисніть 'Запустити аналіз'.")

    st.markdown(
        """
### Що робить цей застосунок
1. Приймає ЄДРПОУ компанії.
2. Через API SMIDA отримує список річних звітів.
3. Складає результати у `pandas.DataFrame`.
4. Автоматично намагається витягнути фінансові показники зі сторінок звітів.
5. Будує таблиці та графік.
6. Обчислює сукупний ризиковий бал.

### Запуск локально
```bash
pip install streamlit pandas requests beautifulsoup4 lxml
streamlit run app.py
```

### requirements.txt
```txt
streamlit
pandas
requests
beautifulsoup4
lxml
```
"""
)
