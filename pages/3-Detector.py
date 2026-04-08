
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from deepface import DeepFace
import logging
import av

# Configuración de la página
st.set_page_config(page_title="IA Facial - Reconocimiento", page_icon="👤", layout="wide")

st.title("👤 Detector Facial en Tiempo Real")
st.sidebar.header("Configuración de Modelos")

mode = st.sidebar.selectbox("Selecciona lo que quieres detectar:", 
                            ["Todo", "Emociones", "Edad y Género"])

st.markdown("""
Este sistema utiliza **DeepFace** para el análisis biométrico y **WebRTC** para la transmisión de video.
Analiza: Emociones (7), Género y Edad aproximada.
""")

class FaceAnalyzer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.last_results = None
        self.analysis_interval = 15  # Analiza cada 15 frames (aprox 4 veces por segundo a 60fps)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Solo analizamos cada N frames para no trabar el video
        if self.frame_count % self.analysis_interval == 0 or self.last_results is None:
            try:
                # Reducir aún más la resolución para análisis más rápido (1/4 del tamaño original)
                small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
                # Usar backend 'ssd' para mayor velocidad
                self.last_results = DeepFace.analyze(
                    small_img,
                    actions=['emotion', 'age', 'gender'],
                    enforce_detection=False,
                    detector_backend='ssd',
                    silent=True
                )
                # Ajustar coordenadas a la imagen original
                for res in self.last_results:
                    for key in ['x', 'w']:
                        res['region'][key] = int(res['region'][key] * 4)
                    for key in ['y', 'h']:
                        res['region'][key] = int(res['region'][key] * 4)
            except Exception as e:
                logging.error(f"Error en análisis: {e}")
                self.last_results = None

        # Dibujar la última predicción en todos los frames
        if self.last_results is not None:
            # Diccionarios de traducción
            emociones_es = {
                'angry': 'Enojo',
                'disgust': 'Asco',
                'fear': 'Miedo',
                'happy': 'Feliz',
                'sad': 'Triste',
                'surprise': 'Sorpresa',
                'neutral': 'Neutral'
            }
            generos_es = {
                'Man': 'Hombre',
                'Woman': 'Mujer'
            }
            for res in self.last_results:
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                emotion = emociones_es.get(res.get('dominant_emotion', ''), 'Desconocido')
                gender = generos_es.get(res.get('dominant_gender', ''), 'Desconocido')
                age = res.get('age', '?')
                label = f"Género: {gender}, Edad: {age} años"
                if mode == "Emociones" or mode == "Todo":
                    label = f"Emoción: {emotion} | " + label
                # Eliminar caracteres extraños y asegurar codificación
                label = label.encode('utf-8', errors='ignore').decode('utf-8')
                cv2.putText(img, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Contenedor para el video
st.subheader("Cámara en Vivo")
ctx = webrtc_streamer(
    key="facial-analysis",
    video_transformer_factory=FaceAnalyzer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_transformer:
    st.success("Analizando flujo de video...")

# Sección informativa sobre los modelos
with st.expander("ℹ️ Información sobre los modelos utilizados"):
    st.write("""
    - **Reconocimiento Facial:** Basado en arquitecturas VGG-Face.
    - **Emociones:** El modelo clasifica entre: *enojado, asco, miedo, feliz, triste, sorprendido y neutral*.
    - **Rendimiento:** Se implementó un skip-frame (procesa 1 de cada 5 frames) para garantizar que la app no se bloquee en dispositivos con pocos recursos.
    """)

st.markdown("---")
st.markdown("© Juan Esteban Orozco ")