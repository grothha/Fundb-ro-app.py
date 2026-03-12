import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# --- PFAD-KONFIGURATION ---
# Wir suchen einfach im aktuellen Verzeichnis nach dem Ordner "model"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "keras_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "model", "labels.txt")

# --- FUNKTIONEN ---
@st.cache_resource
def load_model_file():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modell nicht gefunden. Erwarteter Pfad: {MODEL_PATH}")
        return None
    try:
        # compile=False ist wichtig für Teachable Machine Modelle
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None

def load_labels():
    if not os.path.exists(LABEL_PATH):
        st.error(f"Labels nicht gefunden unter: {LABEL_PATH}")
        return []
    
    labels = []
    try:
        with open(LABEL_PATH, "r", encoding="utf-8") as f:
            for line in f.readlines():
                # Formate wie "0 Jacke" zu "Jacke" umwandeln
                parts = line.strip().split(" ", 1)
                labels.append(parts[1] if len(parts) > 1 else parts[0])
    except Exception:
        st.error("Fehler beim Lesen der labels.txt")
    return labels

# --- UI SETUP ---
st.set_page_config(page_title="Schul-Fundbüro KI", page_icon="🏫")
st.title("🏫 Schul-Fundbüro KI-App")
st.write("Lade ein Bild eines verlorenen Gegenstands hoch, um ihn automatisch zu kategorisieren.")

# Modell und Labels laden
model = load_model_file()
labels = load_labels()

# --- BILD-UPLOAD & VERARBEITUNG ---
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is None:
        st.error("KI-Modell konnte nicht geladen werden. Bitte Dateistruktur prüfen.")
    else:
        # Bild anzeigen
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

        # 1. Bildvorbereitung (Preprocessing)
        # Teachable Machine nutzt meist 224x224 Pixel
        size = (224, 224)
        img = image.resize(size)
        img_array = np.asarray(img)
        
        # 2. Normalisierung (Standard für Teachable Machine)
        normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
        data = np.expand_dims(normalized_image_array, axis=0)

        # 3. Vorhersage
        with st.spinner('KI analysiert das Fundstück...'):
            prediction = model.predict(data)
            index = np.argmax(prediction)
            confidence_score = prediction[0][index]

        # 4. Ergebnis anzeigen
        st.divider()
        st.subheader("🔎 Analyse-Ergebnis")
        
        if labels and index < len(labels):
            label_name = labels[index]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Gegenstand", value=label_name)
            with col2:
                st.metric(label="Sicherheit", value=f"{confidence_score * 100:.1f}%")
            
            st.progress(float(confidence_score))
            
            if confidence_score < 0.6:
                st.warning("Hinweis: Die KI ist sich unsicher. Bitte manuell prüfen.")
        else:
            st.error("Analyse abgeschlossen, aber Zuordnung zum Namen fehlgeschlagen.")

# --- FOOTER ---
st.sidebar.info("Tipp: Achte auf gute Beleuchtung und einen neutralen Hintergrund.")
