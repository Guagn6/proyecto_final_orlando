import os
from dotenv import load_dotenv
import mysql.connector
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind
import json
# import boto3 # dependendcias agregadas
# from botocore.exceptions import ClientError

# lambda_client = boto3.client("lambda", region_name="us-east-1")
# s3 = boto3.client("s3")

# BUCKET = "xideralaws-curso-orlando"
# KEY = "student-performance.csv"
# LAMBDA_NAME = "lambda-function"

# Cargar variables desde .env
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

def get_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Datos",
    page_icon="📚",
    layout="wide"
)

# Inicialización de estado de sesión
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False

# Función AWS Lambda (en entorno real requiere las credenciales AWS)
@st.cache_data
def aws_lambda_processing(file_content):
    
    df = pd.read_csv(file_content)
    
    numeric_columns = ['StudentID', 'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 
                      'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 
                      'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GPA', 'GradeClass']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Eliminar filas con valores nulos solo después de conversión
    df_clean = df.dropna()

    if len(df_clean) == 0:
        st.error("❌ No quedaron datos válidos después de la limpieza")
        return pd.DataFrame()

    try:
        # Configuración MySQL (ajusta estos valores)
        connection = get_connection
        cursor = connection.cursor()
        
        # Insertar registros del DataFrame
        for _, row in df_clean.iterrows():
            cursor.execute("""
                INSERT INTO student_performance 
                (StudentID, Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, 
                 Absences, Tutoring, ParentalSupport, Extracurricular, Sports, Music, 
                 Volunteering, GPA, GradeClass)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, tuple(row))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        st.success(f"✅ {len(df_clean)} registros guardados exitosamente")
        
    except Exception as e:
        st.error(f"Error guardando en MySQL: {e}")
    
    return df_clean
    
    # mode = st.sidebar.radio("Modo de carga", ["json", "parquet"])
    # if st.sidebar.button("Procesar archivo"):
        # payload = {"bucket": BUCKET, "key": KEY, "mode": mode}
        
        # Invocar Lambda
        # response = lambda_client.invoke(
        #     FunctionName=LAMBDA_NAME,
        #     InvocationType="RequestResponse",
        #     Payload=json.dumps(payload)
        # )
        # result = json.loads(response["Payload"].read())

        # if result["statusCode"] == 200:
        #     body = json.loads(result["body"])
            
        #     if mode == "json":
        #         st.success("Datos obtenidos como JSON")
        #         df = pd.DataFrame(body["data"])
        #         st.write("Vista previa:")
        #         st.dataframe(df)

        #     else:
        #         st.success("Archivo limpio guardado en S3")
        #         st.write("Ruta en S3:", body["s3_key"])

        #         # Descargar Parquet procesado desde S3
        #        obj = s3.get_object(Bucket=BUCKET, Key=body["s3_key"])
        #         df = pd.read_parquet(obj["Body"])
        #         st.write("Vista previa:")
        #         st.dataframe(df.head())

            # Ejemplo de gráfico
        #     st.subheader("Promedios de columnas numéricas")
        #     means = body["promedios_columnas"]
        #     means_df = pd.DataFrame(list(means.items()), columns=["Columna", "Promedio"])
        #     st.bar_chart(means_df.set_index("Columna"))
        # else:
        #     st.error("Error al invocar Lambda") """

# Funcion AWS para recuperar datos
@st.cache_data
def load_data():
    conn = get_connection()
    query = "SELECT * FROM student_performance;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Pantalla de inicio
def show_home_screen():
    st.title("📚 Sistema de Análisis de Rendimiento Estudiantil")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🔄 Procesar Nuevo Archivo")
        st.markdown("""
        **Carga un archivo nuevo para análizarlo:**
        - El archivo será procesado automáticamente
        - Los datos serán limpiados y transformados
        - Se guardará una copia para futuras revisiones
        - Se mostrarán los análisis estadísticos
        """)
        
        uploaded_file = st.file_uploader(
            "Selecciona tu archivo CSV",
            type=['csv'],
            key="new_file"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Procesar con AWS Lambda", type="primary"):
                processed_df = aws_lambda_processing(uploaded_file)
                if processed_df is not None:
                    st.session_state.df = processed_df
                    st.session_state.data_loaded = True
                    st.session_state.show_analysis = True
                    st.rerun()
    
    with col2:
        st.header("📁 Cargar Análisis Previo")
        st.markdown("""
        **Carga un análisis previamente procesado:**
        - Archivos ya limpiados en S3
        - Análisis instantáneo
        - Datos previamente validados
        """)
        
        if st.button("📊 Cargar Análisis", type="secondary"):
            saved_df = load_data()
            if saved_df is not None:
                st.session_state.df = saved_df
                st.session_state.data_loaded = True
                st.session_state.show_analysis = True
                st.rerun()

    
    # Información del formato de datos esperado
    st.markdown("---")
    st.subheader("📋 Formato de Datos Esperado")

    st.code("""
StudentID,Age,Gender,Ethnicity,ParentalEducation,StudyTimeWeekly,Absences,Tutoring,
ParentalSupport,Extracurricular,Sports,Music,Volunteering,GPA,GradeClass
    """)
    
    st.markdown("**Descripción de columnas:**")
    cols_info = {
        "StudentID": "Identificador único del estudiante",
        "Age": "Edad del estudiante",
        "Gender": "Género (0/1)",
        "Ethnicity": "Etnia (categórica)",
        "ParentalEducation": "Nivel educativo de los padres",
        "StudyTimeWeekly": "Horas de estudio semanales",
        "Absences": "Número de ausencias",
        "Tutoring": "Recibe tutoría (0/1)",
        "ParentalSupport": "Nivel de apoyo parental",
        "Extracurricular": "Actividades extracurriculares (0/1)",
        "Sports": "Practica deportes (0/1)",
        "Music": "Estudia música (0/1)",
        "Volunteering": "Hace voluntariado (0/1)",
        "GPA": "Promedio de calificaciones",
        "GradeClass": "Clasificación de grado"
    }
    
    for col, desc in cols_info.items():
        st.markdown(f"- **{col}**: {desc}")

# Función para mostrar análisis
def show_analysis():
    df = st.session_state.df
    
    # ✅ AGREGAR ESTA VALIDACIÓN
    if df is None or len(df) == 0:
        st.error("No hay datos para mostrar")
        return
    
    # Verificar que las columnas existan y tengan datos válidos
    if 'Age' not in df.columns or df['Age'].isna().all():
        st.error("La columna 'Age' no existe o no tiene datos válidos")
        return
        
    # Título principal
    st.title("📊 Análisis de Rendimiento Estudiantil")
    
    # Sidebar con controles
    st.sidebar.title("🎛️ Panel de Control")
    st.sidebar.markdown("---")
    
    # Botón para volver al inicio
    if st.sidebar.button("🏠 Volver al Inicio"):
        st.session_state.show_analysis = False
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros de Datos")
    
    # Filtros en sidebar
    age_range = st.sidebar.slider(
        "Rango de Edad:",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max())),
        step=1
    )
    
    gpa_range = st.sidebar.slider(
        "Rango de GPA:",
        min_value=float(df['GPA'].min()),
        max_value=float(df['GPA'].max()),
        value=(float(df['GPA'].min()), float(df['GPA'].max())),
        step=0.1
    )
    
    tutoring_filter = st.sidebar.selectbox(
        "Filtrar por Tutoría:",
        options=["Todos", "Con tutoría (1)", "Sin tutoría (0)"]
    )
    
    gender_filter = st.sidebar.selectbox(
        "Filtrar por Género:",
        options=["Todos", "Género 0", "Género 1"]
    )
    
    # Aplicar filtros
    df_filtered = df.copy()
    df_filtered = df_filtered[
        (df_filtered['Age'] >= age_range[0]) & 
        (df_filtered['Age'] <= age_range[1]) &
        (df_filtered['GPA'] >= gpa_range[0]) & 
        (df_filtered['GPA'] <= gpa_range[1])
    ]
    
    if tutoring_filter == "Con tutoría (1)":
        df_filtered = df_filtered[df_filtered['Tutoring'] == 1]
    elif tutoring_filter == "Sin tutoría (0)":
        df_filtered = df_filtered[df_filtered['Tutoring'] == 0]
    
    if gender_filter == "Género 0":
        df_filtered = df_filtered[df_filtered['Gender'] == 0]
    elif gender_filter == "Género 1":
        df_filtered = df_filtered[df_filtered['Gender'] == 1]
    
    # Información del dataset en sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Info del Dataset")
    st.sidebar.metric("Estudiantes originales", len(df))
    st.sidebar.metric("Estudiantes filtrados", len(df_filtered))
    st.sidebar.metric("GPA promedio", f"{df_filtered['GPA'].mean():.2f}")
    st.sidebar.metric("Edad promedio", f"{df_filtered['Age'].mean():.1f}")
    
    # KPIs principales
    st.header("📊 Métricas Principales")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Estudiantes Total", len(df_filtered))
    with col2:
        st.metric("GPA Promedio", f"{df_filtered['GPA'].mean():.2f}")
    with col3:
        st.metric("Horas Estudio/Semana", f"{df_filtered['StudyTimeWeekly'].mean():.1f}")
    with col4:
        st.metric("Ausencias Promedio", f"{df_filtered['Absences'].mean():.1f}")
    with col5:
        tutoring_pct = (df_filtered['Tutoring'].sum() / len(df_filtered) * 100)
        st.metric("% Con Tutoría", f"{tutoring_pct:.1f}%")
    
    st.markdown("---")
    
    # Tabs de análisis
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Estadísticas", "📈 Distribuciones", "📊 Comparativas", "🔗 Correlaciones", "🎯 Análisis Específicos"])
    
    with tab1:
            st.subheader("Estadísticas Descriptivas")
            
            # Estadísticas generales
            st.write("**Estadísticas del dataset filtrado:**")
            st.dataframe(df_filtered.describe())
            
            # Distribución por categorías
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Distribución por Género:**")
                gender_dist = df_filtered['Gender'].value_counts()
                st.dataframe(gender_dist)
                
                st.write("**Distribución por Tutoría:**")
                tutoring_dist = df_filtered['Tutoring'].value_counts()
                st.dataframe(tutoring_dist)
            
            with col2:
                st.write("**Distribución por Apoyo Parental:**")
                support_dist = df_filtered['ParentalSupport'].value_counts()
                st.dataframe(support_dist)
                
                if 'GradeClass' in df_filtered.columns:
                    st.write("**Distribución por Clase de Calificación:**")
                    grade_dist = df_filtered['GradeClass'].value_counts()
                    st.dataframe(grade_dist)

    with tab2:
        st.subheader("Distribución de Variables Clave")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Histograma del GPA
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df_filtered['GPA'], bins=30, kde=True, color='skyblue', ax=ax)
            plt.title('Distribución del GPA')
            plt.xlabel('GPA')
            plt.ylabel('Frecuencia')
            st.pyplot(fig, clear_figure=True)
        
        with col2:
            # 9. Distribución de Ausencias
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df_filtered['Absences'], bins=30, kde=True, color='salmon', ax=ax)
            plt.title("Distribución de Ausencias")
            plt.xlabel("Número de Ausencias")
            plt.ylabel("Frecuencia")
            st.pyplot(fig, clear_figure=True)
        
        # Distribución de tiempo de estudio
        st.subheader("Distribución de Tiempo de Estudio Semanal")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df_filtered['StudyTimeWeekly'], bins=25, kde=True, color='lightgreen', ax=ax)
        plt.title("Distribución del Tiempo de Estudio Semanal")
        plt.xlabel("Horas de Estudio por Semana")
        plt.ylabel("Frecuencia")
        st.pyplot(fig, clear_figure=True)
    
    with tab3:
        st.subheader("Análisis Comparativo por Categorías")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 2. Boxplot de GPA por clase de calificación
            if 'GradeClass' in df_filtered.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x='GradeClass', y='GPA', data=df_filtered, palette='Set2', ax=ax)
                plt.title('GPA por Clase de Calificación')
                plt.xlabel('Clase de Calificación')
                plt.ylabel('GPA')
                st.pyplot(fig, clear_figure=True)
        
        with col2:
            # 6. Boxplot de GPA por apoyo parental
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='ParentalSupport', y='GPA', data=df_filtered, palette='pastel', ax=ax)
            plt.title('GPA por Nivel de Apoyo Parental')
            plt.xlabel('Apoyo Parental')
            plt.ylabel('GPA')
            st.pyplot(fig, clear_figure=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # 11. GPA por Nivel Educativo de los Padres
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='ParentalEducation', y='GPA', data=df_filtered, palette='Set3', ax=ax)
            plt.title("GPA por Nivel Educativo de los Padres")
            plt.xlabel("Nivel Educativo de los Padres")
            plt.ylabel("GPA")
            st.pyplot(fig, clear_figure=True)
        
        with col4:
            # 13. GPA por Actividades Extracurriculares
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Extracurricular', y='GPA', data=df_filtered, palette='Set2', ax=ax)
            plt.title("GPA por Participación en Actividades Extracurriculares")
            plt.xlabel("Participación en Actividades Extracurriculares (0=No, 1=Sí)")
            plt.ylabel("GPA")
            st.pyplot(fig, clear_figure=True)
        
        # Comparaciones adicionales
        col5, col6 = st.columns(2)
        
        with col5:
            # GPA por participación en deportes
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Sports', y='GPA', data=df_filtered, palette='Set1', ax=ax)
            plt.title("GPA por Participación en Deportes")
            plt.xlabel("Participación en Deportes (0=No, 1=Sí)")
            plt.ylabel("GPA")
            st.pyplot(fig, clear_figure=True)
        
        with col6:
            # 14. GPA por Participación en Música
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Music', y='GPA', data=df_filtered, palette='Set1', ax=ax)
            plt.title("GPA por Participación en Música")
            plt.xlabel("Participación en Música (0=No, 1=Sí)")
            plt.ylabel("GPA")
            st.pyplot(fig, clear_figure=True)
    
    with tab4:
        st.subheader("Análisis de Correlaciones")
        
        # 4 y 5. Matriz de correlación y mapa de calor
        correlation_cols = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA', 'ParentalSupport', 'Tutoring', 
                          'ParentalEducation', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
        
        # Filtrar solo las columnas que existen en el DataFrame
        available_cols = [col for col in correlation_cols if col in df_filtered.columns]
        correlation_matrix = df_filtered[available_cols].corr()
        
        # Mapa de calor
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title('Mapa de Calor de Correlaciones')
        st.pyplot(fig, clear_figure=True)
        
        # Mostrar correlaciones más fuertes
        st.subheader("Correlaciones más Significativas")
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        corr_values = correlation_matrix.mask(mask).stack().reset_index()
        corr_values.columns = ['Variable 1', 'Variable 2', 'Correlación']
        corr_values = corr_values[corr_values['Variable 1'] != corr_values['Variable 2']]
        strongest_corr = corr_values.reindex(corr_values['Correlación'].abs().sort_values(ascending=False).index).head(10)
        
        for _, row in strongest_corr.iterrows():
            correlation = row['Correlación']
            if abs(correlation) > 0.1:  # Solo mostrar correlaciones relevantes
                strength = "Fuerte" if abs(correlation) > 0.6 else "Moderada" if abs(correlation) > 0.3 else "Débil"
                direction = "positiva" if correlation > 0 else "negativa"
                st.write(f"**{row['Variable 1']}** vs **{row['Variable 2']}**: {correlation:.3f} ({strength} {direction})")
    
    with tab5:
        st.subheader("Análisis Específicos")
        
        # 3. Prueba t de Student: GPA entre estudiantes con y sin tutoría
        st.write("**Análisis Estadístico: Efecto de la Tutoría en el GPA**")
        
        if len(df_filtered[df_filtered['Tutoring'] == 1]) > 0 and len(df_filtered[df_filtered['Tutoring'] == 0]) > 0:
            gpa_tutoring = df_filtered[df_filtered['Tutoring'] == 1]['GPA']
            gpa_no_tutoring = df_filtered[df_filtered['Tutoring'] == 0]['GPA']
            t_stat, p_value = ttest_ind(gpa_tutoring, gpa_no_tutoring)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estadístico t", f"{t_stat:.2f}")
            with col2:
                st.metric("Valor p", f"{p_value:.4e}")
            with col3:
                significance = "Significativo" if p_value < 0.05 else "No significativo"
                st.metric("Resultado", significance)
            
            st.write(f"**Interpretación:** {'Existe una diferencia significativa' if p_value < 0.05 else 'No existe diferencia significativa'} en el GPA entre estudiantes con y sin tutoría.")
        
        # 7. Scatterplot de GPA vs. tiempo de estudio
        st.subheader("Relación entre Variables Continuas")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x='StudyTimeWeekly', y='GPA', data=df_filtered, alpha=0.6, ax=ax)
        plt.title('Relación entre Tiempo de Estudio Semanal y GPA')
        plt.xlabel('Horas de Estudio Semanal')
        plt.ylabel('GPA')
        st.pyplot(fig, clear_figure=True)
        
        # Análisis de GPA por múltiples factores
        st.subheader("Análisis Multifactorial del GPA")
        
        factors = ['Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music']
        available_factors = [f for f in factors if f in df_filtered.columns]
        
        if len(available_factors) > 0:
            fig, axes = plt.subplots(1, len(available_factors), figsize=(5*len(available_factors), 6))
            if len(available_factors) == 1:
                axes = [axes]
            
            for i, factor in enumerate(available_factors):
                sns.boxplot(x=factor, y='GPA', data=df_filtered, ax=axes[i])
                axes[i].set_title(f'GPA por {factor}')
            
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

# Lógica principal
def main():
    if not st.session_state.show_analysis:
        show_home_screen()
    else:
        show_analysis()

if __name__ == "__main__":
    main()