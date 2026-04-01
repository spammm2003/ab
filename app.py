import streamlit as st
import pandas as pd
import numpy as np
import time

# Налаштування сторінки
st.set_page_config(page_title='OSINT: Компанія під мікроскопом', page_icon='🔍', layout='centered')

st.title('🔍 OSINT-додаток "Компанія під мікроскопом"')
st.write("Цей додаток демонструє збір фінансових даних компанії та їх аналіз.")

# --- КРОК 1: Імітація запиту до відкритого API (SMIDA, YouControl demo) ---
def fetch_company_data(company_name):
    """
    Функція імітує отримання даних по API. 
    Для повторюваності результатів використовуємо довжину назви як seed.
    """
    time.sleep(1.5) # Імітація затримки мережі
    np.random.seed(len(company_name) + sum([ord(c) for c in company_name]))
    
    years = ['2021', '2022', '2023']
    data = {
        'Рік': years,
        'Дохід (млн грн)': np.random.randint(10, 200, size=3),
        'Прибуток (млн грн)': np.random.randint(-20, 50, size=3),
        'Борг (млн грн)': np.random.randint(0, 150, size=3)
    }
    
    # КРОК 2: Збереження результатів у DataFrame
    return pd.DataFrame(data)

# --- Функція для розрахунку ризику (КРОК 4) ---
def calculate_risk(df):
    """
    Оцінка ризику на основі останнього року (2023).
    Сукупні критерії: від'ємний прибуток та високий рівень боргу.
    """
    latest_data = df.iloc[-1]
    revenue = latest_data['Дохід (млн грн)']
    profit = latest_data['Прибуток (млн грн)']
    debt = latest_data['Борг (млн грн)']
    
    if profit < 0 and debt > revenue:
        return "Високий (Критичний стан)", "🚨", st.error, "Компанія збиткова, а борги перевищують річний дохід. Співпраця дуже ризикована."
    elif profit < 0 or debt > (revenue * 0.7):
        return "Середній (Потребує уваги)", "⚠️", st.warning, "Компанія має збитки за останній звітний період або підвищене боргове навантаження."
    else:
        return "Низький (Стабільна)", "✅", st.success, "Фінансові показники в нормі. Компанія генерує прибуток, боргове навантаження помірне."

# --- Інтерфейс користувача ---
st.markdown("### 1. Пошук компанії")
company_name = st.text_input("Введіть назву компанії або ЄДРПОУ:", placeholder="Наприклад: ТОВ 'Рога і Копита'")

if st.button("Аналізувати компанію", type="primary"):
    if company_name.strip():
        with st.spinner(f"Запит до API (YouControl/SMIDA demo) для '{company_name}'..."):
            df = fetch_company_data(company_name)
        
        st.divider()
        
        # --- КРОК 3: Відображення фінансових показників ---
        st.markdown(f"### 📊 Фінансовий звіт: **{company_name}**")
        
        st.write("#### Таблиця показників (DataFrame)")
        # Відображення таблиці з приховуванням індексів для краси
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.write("#### Динаміка Доходу та Прибутку")
        # Підготовка даних для графіка (Рік стає індексом)
        chart_data = df.set_index('Рік')
        
        # Стовпчикова діаграма для доходу та прибутку
        st.bar_chart(chart_data[['Дохід (млн грн)', 'Прибуток (млн грн)']])
        
        st.write("#### Динаміка Заборгованості")
        # Лінійний графік для боргу
        st.line_chart(chart_data[['Борг (млн грн)']], color="#ff4b4b")
        
        st.divider()
        
        # --- КРОК 4: Додай оцінку ризику за сукупними критеріями ---
        st.markdown("### 🎯 Оцінка ризиків (Scoring)")
        risk_level, icon, alert_type, description = calculate_risk(df)
        
        # Виведення результату відповідним кольором
        alert_type(f"{icon} **Рівень ризику:** {risk_level}")
        st.write(f"**Обґрунтування:** {description}")
        
    else:
        st.warning("Будь ласка, введіть назву компанії для початку аналізу.")
