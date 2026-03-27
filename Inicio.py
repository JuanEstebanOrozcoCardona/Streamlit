import streamlit as st

st.title("🚀 Curso de Streamlit")

st.header("¿Qué es Streamlit?")
st.write("""
    Streamlit es un framework de Python que permite crear aplicaciones web interactivas
    de forma rápida, ideal para visualización de datos y prototipos de machine learning.
    """)

st.header("Componentes básicos")
st.markdown("""
    - Títulos (`st.title`)
    - Texto (`st.write`, `st.text`)
    - Imágenes (`st.image`)
    - Inputs (`st.selectbox`, `st.radio`)
    """)

st.subheader("Imagen de ejemplo")
st.image("images/logo.webp", caption="Imagen de ejemplo", use_column_width=True)

st.subheader("Interacción")
opcion = st.radio("Selecciona una opción:", ["Python", "Streamlit", "Machine Learning"])

if opcion == "Python":
    st.success("Python es el lenguaje base 🐍")
elif opcion == "Streamlit":
    st.info("Streamlit facilita crear apps web ⚡")
else:
    st.warning("ML permite crear modelos inteligentes 🤖")

st.markdown("---")
st.markdown("© Juan Esteban Orozco ")