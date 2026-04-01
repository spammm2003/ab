import re
import io
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from bs4 import BeautifulSoup
from lxml import etree

st.set_page_config(page_title="Компанія під мікроскопом", page_icon="🔎", layout="wide")

SMIDA_FEED_URL = "https://smida.gov.ua/db/api/v1/feed-index.xml"
YOUCONTROL_SEARCH_URL = "https://youcontrol.com.ua/search/?q={query}"
YOUCONTROL_COMPANY_URL = "https://youcontrol.com.ua/catalog/company_details/{edrpou}/"
REQUEST_TIMEOUT = 25
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
}


# -----------------------------
# Допоміжні функції
# -----------------------------
def normalize_company_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name


def safe_get(url: str, params: Optional[dict] = None) -> requests.Response:
    response = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response


@st.cache_data(show_spinner=False, ttl=3600)
def search_youcontrol_company(query: str) -> Dict:
    """
    Пошук компанії через публічну пошукову сторінку YouControl.
    Повертає перший знайдений ЄДРПОУ, якщо він є в HTML.
    """
    query = normalize_company_name(query)
    url = YOUCONTROL_SEARCH_URL.format(query=quote(query))

    try:
        resp = safe_get(url)
        html = resp.text

        # Шукаємо перший company_details/<edrpou>
        match = re.search(r"company_details/(\d{8,10})/", html)
        if match:
            edrpou = match.group(1)
            return {
                "query": query,
                "edrpou": edrpou,
                "search_url": url,
                "company_url": YOUCONTROL_COMPANY_URL.format(edrpou=edrpou),
                "source": "YouControl search",
            }

        # fallback: спроба знайти код на сторінці
        code_match = re.search(r"Код\s+ЄДРПОУ\s*</?[^>]*>\s*(\d{8,10})", html, flags=re.IGNORECASE)
        if code_match:
            edrpou = code_match.group(1)
            return {
                "query": query,
                "edrpou": edrpou,
                "search_url": url,
                "company_url": YOUCONTROL_COMPANY_URL.format(edrpou=edrpou),
                "source": "YouControl search",
            }

        return {
            "query": query,
            "edrpou": None,
            "search_url": url,
            "company_url": None,
            "source": "YouControl search",
            "error": "Компанію не знайдено на публічній сторінці пошуку YouControl.",
        }
    except Exception as e:
        return {
            "query": query,
            "edrpou": None,
            "search_url": url,
            "company_url": None,
            "source": "YouControl search",
            "error": f"Помилка пошуку YouControl: {e}",
        }


@st.cache_data(show_spinner=False, ttl=3600)
def get_smida_reports_by_edrpou(edrpou: str, limit: int = 300) -> pd.DataFrame:
    params = {
        "edrpou": edrpou,
        "period": "y",  # річні звіти
        "limit": limit,
    }

    response = safe_get(SMIDA_FEED_URL, params=params)
    xml_bytes = response.content
    root = etree.fromstring(xml_bytes)

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

    # типові поля в feed-index.xml
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

    if "title" in df.columns:
        df["title"] = df["title"].astype(str)

    return df


def choose_latest_company_name(df: pd.DataFrame, fallback: str) -> str:
    if df.empty or "company_name" not in df.columns:
        return fallback
    values = df["company_name"].dropna().astype(str)
    if values.empty:
        return fallback
    return values.iloc[0]


def extract_report_links(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    links = []
    if "link" in df.columns:
        links.extend(df["link"].dropna().astype(str).tolist())
    return [x for x in links if x.startswith("http")]


@st.cache_data(show_spinner=False, ttl=3600)
def parse_financials_from_report_page(url: str) -> List[Dict]:
    """
    Універсальний парсер HTML-таблиць зі сторінки звіту.
    Шукає типові фінансові показники за ключовими словами.
    """
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
                            results.append({
                                "metric": metric_name,
                                "value": value,
                                "source_url": url,
                            })
        return results
    except Exception:
        return []



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



def build_financial_dataframe(report_df: pd.DataFrame, max_reports_to_scan: int = 12) -> pd.DataFrame:
    links = extract_report_links(report_df)[:max_reports_to_scan]
    collected = []

    for link in links:
        collected.extend(parse_financials_from_report_page(link))

    if not collected:
        return pd.DataFrame(columns=["metric", "value", "source_url"])

    fin_df = pd.DataFrame(collected)
    fin_df = fin_df.dropna(subset=["metric", "value"])

    # беремо останнє значення по кожному показнику
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
        reasons.append("Висока боргове навантаження відносно активів.")

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


# -----------------------------
# UI
# -----------------------------
st.title("🔎 OSINT-додаток: Компанія під мікроскопом")
st.caption("Пошук компанії, збір даних із відкритих джерел, DataFrame, графіки та оцінка ризику.")

with st.sidebar:
    st.header("Налаштування")
    company_query = st.text_input("Назва компанії", placeholder="Наприклад: ПриватБанк")
    max_reports = st.slider("Скільки SMIDA-звітів аналізувати", 3, 30, 12)
    run_btn = st.button("Запустити аналіз", type="primary")

st.markdown(
    """
    **Джерела:**
    - **SMIDA**: офіційний API відкритих даних.
    - **YouControl demo / public pages**: публічний пошук компаній та демо-досьє.

    > Примітка: структура HTML-сторінок може змінюватися. Якщо SMIDA/YouControl змінять верстку,
    > парсер може потребувати оновлення.
    """
)

if run_btn:
    if not company_query.strip():
        st.warning("Введіть назву компанії.")
        st.stop()

    with st.spinner("Шукаю компанію в YouControl..."):
        yc = search_youcontrol_company(company_query)

    if yc.get("error"):
        st.error(yc["error"])
        st.stop()

    edrpou = yc.get("edrpou")
    if not edrpou:
        st.error("Не вдалося визначити ЄДРПОУ компанії.")
        st.stop()

    st.success(f"Знайдено ЄДРПОУ: {edrpou}")

    with st.spinner("Отримую річні звіти з SMIDA..."):
        report_df = get_smida_reports_by_edrpou(edrpou)

    company_name = choose_latest_company_name(report_df, company_query)

    col1, col2, col3 = st.columns(3)
    col1.metric("Компанія", company_name)
    col2.metric("ЄДРПОУ", edrpou)
    col3.markdown(f"[Відкрити досьє YouControl]({yc['company_url']})")

    st.subheader("1) Реєстр звітів SMIDA")
    if report_df.empty:
        st.warning("SMIDA не повернула звітів для цієї компанії.")
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
            "Це нормально для частини емітентів: структура звітів може відрізнятися."
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
        fig_bar = px.bar(chart_df, x="metric", y="value", title="Ключові фінансові показники")
        st.plotly_chart(fig_bar, use_container_width=True)

        positive_df = chart_df.copy()
        positive_df = positive_df[positive_df["value"] > 0]
        if not positive_df.empty:
            fig_pie = px.pie(positive_df, names="metric", values="value", title="Структура позитивних показників")
            st.plotly_chart(fig_pie, use_container_width=True)

        score, level, reasons, ratios = calculate_risk_score(fin_df)

        st.subheader("3) Оцінка ризику")
        risk_col1, risk_col2 = st.columns([1, 2])
        risk_col1.metric("Risk score", score)
        risk_col2.metric("Рівень ризику", level)

        ratio_df = pd.DataFrame(
            [{"ratio": k, "value": v} for k, v in ratios.items()]
        )
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
            f"Оцінка базується на доступних показниках зі звітності SMIDA та відкритій інформації з YouControl."
        )

else:
    st.info("Введіть назву компанії в лівій панелі та натисніть 'Запустити аналіз'.")

    st.markdown(
        """
        ### Що робить цей застосунок
        1. Приймає назву компанії.
        2. Через публічний пошук YouControl намагається знайти ЄДРПОУ.
        3. Через API SMIDA отримує список річних звітів.
        4. Складає результати у `pandas.DataFrame`.
        5. Будує таблиці та графіки.
        6. Обчислює сукупний ризиковий бал.

        ### Запуск локально
        ```bash
        pip install streamlit pandas requests beautifulsoup4 lxml plotly numpy
        streamlit run streamlit_osint_app.py
        ```
        """
    )
