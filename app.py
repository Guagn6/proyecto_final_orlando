import os
from dotenv import load_dotenv
import mysql.connector
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Cargar variables desde .env
# Comentario de prueba
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

def get_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )


# ========================================
# CARGA DE DATOS
# ========================================
@st.cache_data
def load_data():
    conn = get_connection()
    query = "SELECT * FROM fact_taxi_trips;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ========================================
# CARGA DE DATOS
# ========================================
@st.cache_data
def load_data():
    conn = get_connection()
    query = "SELECT * FROM fact_taxi_trips;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ========================================
# APP STREAMLIT
# ========================================
st.set_page_config(page_title="NYC Taxi Dashboard", layout="wide")

st.title("🚕 NYC Taxi Dashboard")
st.markdown("Visualización dinámica de la tabla `fact_taxi_trips` con **matplotlib/seaborn**")

# Cargar datos
df = load_data()

# ==============================
# FILTROS DINÁMICOS
# ==============================
col1, col2 = st.columns(2)

with col1:
    years = st.multiselect("Selecciona año(s):", options=df["year"].unique(), default=df["year"].unique())

with col2:
    payment_types = st.multiselect("Método de pago:", options=df["payment_type"].unique(), default=df["payment_type"].unique())

# Filtrar dataset
filtered_df = df[(df["year"].isin(years)) & (df["payment_type"].isin(payment_types))]

st.dataframe(filtered_df)

# ==============================
# GRÁFICAS
# ==============================


st.subheader("💵 Ingresos totales por día")
revenue_by_day = df.groupby("pickup_date")["total_amount"].sum().reset_index()

fig, ax = plt.subplots()
sns.lineplot(data=revenue_by_day, x="pickup_date", y="total_amount", marker="o", ax=ax)

ax.set_title("Ingresos Totales por Día")
ax.set_xlabel("Fecha")
ax.set_ylabel("Total ($)")

ax.xaxis.set_major_locator(mdates.AutoDateLocator())     # detecta mejor intervalos
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # formato YYYY-MM-DD
plt.xticks(rotation=45, ha="right")  # rota etiquetas

st.pyplot(fig)

st.subheader("💳 Propinas promedio por método de pago")
tips_by_payment = (
    filtered_df.groupby("payment_type")["tip_amount"].mean().reset_index()
)

fig, ax = plt.subplots()
sns.barplot(data=tips_by_payment, x="payment_type", y="tip_amount", ax=ax)
ax.set_title("Propina promedio por método de pago")
ax.set_xlabel("Método de pago")
ax.set_ylabel("Propina promedio ($)")
st.pyplot(fig)

st.subheader("🚏 Distancia promedio por día de la semana")
dist_by_day = (
    filtered_df.groupby("day_of_week")["trip_distance"].mean().reset_index()
)

fig, ax = plt.subplots()
sns.barplot(data=dist_by_day, x="day_of_week", y="trip_distance", ax=ax)
ax.set_title("Distancia promedio por día de la semana")
ax.set_xlabel("Día de la semana")
ax.set_ylabel("Distancia promedio (millas)")
st.pyplot(fig)
