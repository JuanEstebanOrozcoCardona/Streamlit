import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configuración
st.title("📊 Gráficos")

# Menú interactivo
opcion = st.sidebar.selectbox(
    "Selecciona el tipo de gráfico",
    ["Tipos de Grafico", "Barras", "Líneas", "Pie"]
)

# Cargar datos
df = pd.read_csv("data/datos.csv")

st.header("Graficos de Datos")
st.write("Los gráficos son una manera de representar los datos de manera visual. Los tipos de gráficos incluyen barras, líneas, pie y dispersión.")
#Mas informacion sobre los graficos en markdown
st.markdown("""
- **Barras**: Muestra la cantidad de datos en categorías.
- **Líneas**: Muestra la tendencia de los datos a lo largo del tiempo.
- **Pie**: Muestra la proporción de datos en una categoría.
- **Disppersión**: Muestra la relación entre dos variables.
""")


# -------------------------------
# 📊 GRÁFICO DE BARRAS
# -------------------------------
if opcion == "Barras":
    st.subheader("Gráfico de Barras")

    conteo = df["Curso"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(conteo.index, conteo.values)

    ax.set_xlabel("Curso")
    ax.set_ylabel("Cantidad")

    st.pyplot(fig)

# -------------------------------
# 📈 GRÁFICO DE LÍNEAS
# -------------------------------
elif opcion == "Líneas":
    st.subheader("Gráfico de Líneas")

    fig, ax = plt.subplots()
    ax.plot(df["Edad"])

    ax.set_ylabel("Edad")

    st.pyplot(fig)

# -------------------------------
# 🥧 GRÁFICO DE PIE
# -------------------------------
elif opcion == "Pie":
    st.subheader("Gráfico de Pie")

    conteo = df["Curso"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(conteo.values, labels=conteo.index, autopct="%1.1f%%")

    st.pyplot(fig)

st.markdown("---")
st.markdown("© Juan Esteban Orozco ")