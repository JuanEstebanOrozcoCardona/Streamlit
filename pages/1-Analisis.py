import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title("📊 Análisis de Datos")

# Cargar datos
df = pd.read_csv("data/datos.csv")

st.header("Valores del DataSet")
st.dataframe(df)

# Crear pestañas correctamente
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Detalles",
       "Información del DataSet",
    "Valores nulos",
    "Valores duplicados",
    "Estadísticas"
])

# 📌 TAB 1
with tab1:
    st.subheader("Detalles")
    #Texto introductorio;
    st.markdown("""
    El DataSet contiene información sobre los estudiantes de una universidad.
    El DataSet incluye las siguientes columnas:
    - **Nombre**: El nombre del estudiante.
    - **Edad**: La edad del estudiante.
    - **Sexo**: El género del estudiante.
    - **Calificación**: La calificación del estudiante en una asignatura.
    """)

# 📌 TAB 2
with tab2:
    st.subheader("Información del DataSet")
    st.write("Primeros registros")
    st.dataframe(df.head())

    st.write("Últimos registros")
    st.dataframe(df.tail())

    st.write("Tipos de datos")
    st.write(df.dtypes)

# 📌 TAB 2
with tab3:
    st.subheader("Valores nulos")
    st.write(df.isnull().sum())

# 📌 TAB 3
with tab4:
    st.subheader("Valores duplicados")
    st.write("Cantidad de duplicados:", df.duplicated().sum())
    st.dataframe(df[df.duplicated()])

# 📌 TAB 4
with tab5:
    st.subheader("Estadísticas")
    st.dataframe(df.describe())

st.markdown("---")
st.markdown("© Juan Esteban Orozco ")